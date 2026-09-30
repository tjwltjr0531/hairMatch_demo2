"""
HairMatch 웹 백엔드 — 하이브리드 분류기 + 확률 기반 다중 추천

09_predict.py 의 FaceShapePredictor 를 그대로 사용한다 (Top-1 74.5% / Top-2 91.0%).
단일 라벨을 단정하지 않고 상위 2개 얼굴형의 스타일을 함께 제시한다.

사전 준비
    1. 09_predict.py 를 face_shape_cnn.py 로 복사 (숫자로 시작하면 import 불가)
       copy 09_predict.py face_shape_cnn.py
    2. 같은 폴더에 hybrid_model.joblib, CNN 모델 파일 배치
    3. db_setup_v2.py 로 스키마 생성 완료

실행
    pip install fastapi uvicorn python-multipart sqlalchemy psycopg2-binary joblib httpx
    set PGCLIENTENCODING=UTF8
    set DATABASE_URL=postgresql+psycopg2://postgres:비번@localhost:5432/hairmatch_db
    uvicorn web_main:app --reload

설계 결정
    - 회원가입 없음. 업로드 사진은 메모리에서만 처리하고 디스크에 저장하지 않는다.
    - 합성은 별도 inference 서버(8001, venv_hf)가 한다. 여기서는 /synthesize 를 넘겨 줄 뿐이다.
      (API_CONTRACT.md 2서버 구조. TensorFlow 와 PyTorch 가상환경이 달라 한 프로세스에 못 합친다)

[변경 이력 — a.html 연동]
    + /recommend               a.html 이 얼굴형 하나 기준으로 스타일 배열을 받는다
    + /analyze 응답 호환 키     face_shape, confidence, ambiguous, top2 (기존 키는 그대로)
    + 라우터 3개                /salons, /me/quota·/generate, /synthesize
    + / 에서 static/index.html  파일 맨 끝에 마운트 (위에 두면 API 가 가려진다)
    + predictor 미로드 시 503
"""

import io
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, joinedload

from db_setup_v2 import (DATABASE_URL, DemoFace, FaceShape, Hairstyle,
                         Recommendation, SynthResult)
from salons_api import router as salons_router
from gemini_api import router as gemini_router
from synth_proxy import router as synth_router, infer_available

# 1순위와 2순위 확률 격차가 이 값보다 작으면 두 얼굴형을 모두 제시한다.
# Top-1 74.5% / Top-2 91.0% 이므로 애매한 경우 2개를 보여주는 편이 정직하다.
AMBIGUOUS_GAP = float(os.getenv("AMBIGUOUS_GAP", "0.15"))

STATIC_DIR = os.getenv("STATIC_DIR", "static")

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
Session = sessionmaker(bind=engine, future=True)

predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 로드는 무거우므로 서버 기동 시 한 번만 수행한다.
    global predictor
    from face_shape_cnn import FaceShapePredictor
    predictor = FaceShapePredictor()
    print("분류기 로드 완료:", predictor.classes)
    ok = await infer_available()
    print("합성 서버:", f"연결됨 {sorted(ok)}" if ok is not None else "연결 안 됨 (8001 을 먼저 켤 것)")
    yield


app = FastAPI(title="HairMatch API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(salons_router)   # GET  /salons
app.include_router(gemini_router)   # GET  /me/quota, POST /generate
app.include_router(synth_router)    # POST /synthesize → inference 8001

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def fetch_styles(session, face_shape_code: str, gender: str):
    """해당 얼굴형·성별의 추천 스타일을 rank 순으로 반환"""
    q = (select(Recommendation)
         .options(joinedload(Recommendation.hairstyle))
         .where(Recommendation.face_shape_code == face_shape_code,
                Recommendation.gender == gender)
         .order_by(Recommendation.rank))
    out = []
    for r in session.scalars(q):
        hs = r.hairstyle
        # 사전 생성된 예시 이미지가 있으면 하나 붙인다 (없으면 None)
        example = session.scalar(
            select(SynthResult.image_path)
            .join(DemoFace, SynthResult.demo_face_id == DemoFace.id)
            .where(SynthResult.hairstyle_id == hs.id,
                   DemoFace.gender == gender)
            .limit(1)
        )
        out.append({
            "rank": r.rank,
            "slug": hs.slug,
            "name_ko": hs.name_ko,
            "advice": r.advice,
            "example_image": example,
        })
    return out


@app.get("/health")
def health():
    with Session() as s:
        n_style = len(list(s.scalars(select(Hairstyle))))
        n_synth = len(list(s.scalars(select(SynthResult))))
    return {
        "model_classes": predictor.classes if predictor else None,
        "hairstyles": n_style,
        "synth_results": n_synth,
        "ambiguous_gap": AMBIGUOUS_GAP,
    }


@app.get("/styles")
def list_styles(gender: str | None = None):
    with Session() as s:
        q = select(Hairstyle).order_by(Hairstyle.id)
        if gender:
            q = q.where(Hairstyle.gender.in_([gender, "unisex"]))
        return [{"slug": h.slug, "name_ko": h.name_ko, "gender": h.gender,
                 "has_reference": bool(h.ref_image_path)}
                for h in s.scalars(q)]


@app.get("/recommend")
async def recommend(shape: str, gender: str, all: bool = False):
    """a.html 용. 얼굴형 하나 기준 추천을 rank 순 배열로 준다.
    합성 서버에 레퍼런스가 없는 슬러그는 뺀다 — '추천했는데 합성이 안 되는' 카드를 막는다.
    ?all=1 이면 거르지 않고 DB 그대로 준다 (점검용)."""
    with Session() as s:
        styles = fetch_styles(s, shape, gender)
    ok = await infer_available()           # 합성 서버가 꺼져 있으면 None → 거르지 않는다
    if ok is not None and not all:
        dropped = [x["slug"] for x in styles if x["slug"] not in ok]
        if dropped:
            # 카드가 적게 나오면 여기부터 본다: DB 슬러그와 ref_{slug}.png 이름이 다른 경우
            print(f"[recommend] {shape}/{gender}: 레퍼런스 없음 → 제외 {dropped}")
        styles = [x for x in styles if x["slug"] in ok]
    return styles


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    gender: str = Form(...),
    consent: str = Form("no"),
):
    if predictor is None:
        raise HTTPException(503, "분류기가 아직 로드되지 않았습니다.")
    if gender not in ("male", "female"):
        raise HTTPException(400, "gender 는 'male' 또는 'female' 이어야 합니다.")
    if consent.lower() not in ("yes", "true", "1", "on"):
        raise HTTPException(400, "사진 이용 동의가 필요합니다.")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    # 원본은 메모리에서만 다루고 여기서 참조를 끊는다. 디스크에 저장하지 않는다.
    del contents
    if img is None:
        raise HTTPException(400, "이미지를 읽을 수 없습니다.")

    result = predictor.predict_bgr(img)
    del img
    if result is None:
        raise HTTPException(
            422, "얼굴을 인식하지 못했습니다. 정면을 바라본 밝은 사진을 올려주세요.")

    probs = result["probs"]
    order = sorted(probs.items(), key=lambda kv: -kv[1])
    gap = order[0][1] - order[1][1]
    is_ambiguous = gap < AMBIGUOUS_GAP

    with Session() as s:
        ko = {fs.code: fs.name_ko for fs in s.scalars(select(FaceShape))}

        distribution = [
            {"code": c, "name_ko": ko.get(c, c), "probability": round(v, 4)}
            for c, v in order
        ]

        # 애매하면 상위 2개, 명확하면 1개의 얼굴형에 대해 스타일을 제시
        recommendations = []
        for code, prob in order[: (2 if is_ambiguous else 1)]:
            styles = fetch_styles(s, code, gender)
            if not styles:
                continue
            recommendations.append({
                "face_shape": code,
                "face_shape_ko": ko.get(code, code),
                "probability": round(prob, 4),
                "styles": styles,
            })

    if not recommendations:
        raise HTTPException(404, "추천 데이터가 없습니다. db_setup_v2.py 를 확인하세요.")

    return {
        "status": "success",
        "gender": gender,
        "primary": distribution[0],
        "is_ambiguous": is_ambiguous,
        "confidence_gap": round(gap, 4),
        "distribution": distribution,
        "recommendations": recommendations,
        "notice": ("얼굴형은 경계가 뚜렷하지 않아 두 유형의 특징이 함께 나타날 수 있습니다."
                   if is_ambiguous else None),
        # ↓ a.html 호환용. 위 키는 그대로 두므로 기존 화면(view2)은 영향 없다
        "face_shape": order[0][0],
        "confidence": round(order[0][1], 4),
        "ambiguous": is_ambiguous,
        "top2": [{"code": c, "prob": round(v, 4)} for c, v in order[:2]],
    }


# ★ 반드시 파일 맨 끝. 위에 두면 /analyze, /salons, /synthesize 까지 파일로 찾으려 해서 404 가 난다.
#   / 로 들어오면 static/index.html 을 보여 준다.
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="root")
