"""
infer_server.py — HairFastGAN 합성 서버 (포트 8001, GPU PC 전용)

API_CONTRACT.md §2·3·5 를 구현한다.
    GET  /infer/health      서버·GPU 상태, warm-up 여부
    GET  /infer/styles      실제로 합성 가능한 슬러그 (ref_{slug}.png 가 있는 것)
    POST /infer/synthesize  file, style_slug, keep_color → image/png

web 서버(8000)의 /synthesize 가 여기로 넘겨 준다. 브라우저가 직접 부르지 않는다.

폴더
    HAIRMATCH/
      HairFastGAN/          hair_swap.py, pretrained_models/, input/ref_*.png
      inference/infer_server.py   ← 이 파일
      venv_hf/

실행 (새 CMD 창)
    D:\\hairmatch\\setup_env.bat            ← CUDA 환경. 기존 합성 실험 때와 같다
    cd /d <HAIRMATCH>\\inference
    ..\\venv_hf\\Scripts\\activate
    pip install fastapi uvicorn python-multipart
    python infer_server.py
    → "warm-up 완료" 가 뜬 뒤부터 합성이 약 2.3초

설계
    - 한 번에 1건만 (VRAM 실측 7,210 / 8,192MB). 대기 3건 넘으면 503 GPU_BUSY
    - 원본 사진은 저장하지 않는다. HairFastGAN 이 파일 경로를 받아서
      정렬본과 원본을 임시 파일로 잠깐 쓰고, 합성이 끝나면 바로 지운다
    - 오류는 {"error": {"code", "message"}} 형식만 쓴다
"""

import asyncio
import io
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

# ── HairFastGAN 은 pretrained_models 를 상대경로로 찾는다. import 전에 그 폴더로 이동한다
HERE = Path(__file__).resolve().parent
HAIRFAST_DIR = Path(os.getenv("HAIRFAST_DIR", HERE.parent / "HairFastGAN")).resolve()
if not (HAIRFAST_DIR / "hair_swap.py").exists():
    sys.exit(f"[FAIL] HairFastGAN 폴더를 찾지 못했다: {HAIRFAST_DIR}\n"
             f"       set HAIRFAST_DIR=D:\\hairmatch\\HairFastGAN 처럼 지정할 것")
os.chdir(HAIRFAST_DIR)
sys.path.insert(0, str(HAIRFAST_DIR))

import torch                                                     # noqa: E402
import uvicorn                                                   # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile              # noqa: E402
from fastapi.responses import JSONResponse, Response             # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError          # noqa: E402

# Pillow 10 부터 ANTIALIAS 가 없다. HairFastGAN 내부가 참조한다
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

REF_DIR = HAIRFAST_DIR / "input"
BALD_REF = REF_DIR / "ref_bald.png"
PREDICTOR_PATH = HAIRFAST_DIR / "pretrained_models" / "ShapeAdaptor" / "shape_predictor_68_face_landmarks.dat"

# 대머리를 거쳐야 결과가 나은 슬러그만 여기에 적는다 (two_stage.py 실험 결과 기준).
# 비어 있으면 모두 1단계. 2단계는 시간이 두 배다.
TWO_STAGE = set(filter(None, os.getenv("TWO_STAGE_SLUGS", "").split(",")))

MAX_UPLOAD = 15 * 1024 * 1024
MAX_WAITING = 3                     # 이보다 많이 줄 서면 503

STATE = {"model": None, "predictor": None, "warm": False, "waiting": 0}
GPU: asyncio.Semaphore = None      # 이벤트 루프 안에서 만든다 (lifespan). Python 3.9 이하 호환


# ───────────────────────── 공통 ─────────────────────────
MESSAGES = {
    "NO_FACE_DETECTED": "사진에서 얼굴을 찾지 못했습니다.",
    "MULTIPLE_FACES": "사진에 얼굴이 여러 명 있습니다. 한 명만 나온 사진을 올려 주세요.",
    "UNKNOWN_STYLE": "없는 스타일입니다.",
    "GPU_BUSY": "다른 합성이 진행 중입니다.",
    "BAD_IMAGE": "이미지를 읽을 수 없습니다.",
    "NOT_READY": "모델을 불러오는 중입니다.",
    "INFERENCE_FAILED": "합성 중 오류가 발생했습니다.",
}


def err(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": MESSAGES.get(code, code)}})


def ref_path(slug: str) -> Path:
    return REF_DIR / f"ref_{slug}.png"


def available_slugs() -> list[str]:
    return sorted(p.stem[4:] for p in REF_DIR.glob("ref_*.png") if p.stem != "ref_bald")


def to_pil(t: torch.Tensor) -> Image.Image:
    """two_stage.py 와 같은 변환. float[0,1] / uint8 둘 다 받는다"""
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dtype == torch.uint8:
        arr = t.permute(1, 2, 0).numpy()
    else:
        arr = (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def vram_peak_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_reserved() / 1024 ** 2)


def _tmp_png(img: Image.Image) -> Path:
    """Windows 는 열린 NamedTemporaryFile 을 다른 코드가 못 연다. 닫고 경로만 넘긴다"""
    fd, name = tempfile.mkstemp(suffix=".png", prefix="hm_")
    os.close(fd)
    img.save(name)
    return Path(name)


def _rm(*paths):
    for p in paths:
        try:
            if p:
                os.remove(p)
        except OSError:
            pass


# ───────────────────────── 합성 (스레드에서 실행) ─────────────────────────
def align_upload(raw: bytes) -> Image.Image:
    """업로드 → FFHQ 1024 정렬본. align_simple.py 에서 검증된 호출 방식 그대로 쓴다"""
    from utils.shape_predictor import align_face

    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")   # 휴대폰 사진 회전 보정
    src = _tmp_png(img)
    try:
        faces = align_face(str(src), predictor=STATE["predictor"],
                           is_filepath=True, return_tensors=False)
    finally:
        _rm(src)                                        # 원본 임시 파일은 바로 지운다
    if len(faces) == 0:
        raise ValueError("NO_FACE_DETECTED")
    if len(faces) > 1:
        raise ValueError("MULTIPLE_FACES")
    out = faces[0]
    if out.size != (1024, 1024):
        out = out.resize((1024, 1024), Image.LANCZOS)
    return out


def run_synthesis(raw: bytes, slug: str, keep_color: bool) -> tuple[bytes, dict]:
    model = STATE["model"]
    aligned = align_upload(raw)
    face = _tmp_png(aligned)
    mid = None
    shape = ref_path(slug)
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        # ★ keep_color=False 면 color 를 shape 와 '같은 객체'로 넘긴다.
        #   hair_swap.py 의 `shape is not color` 분기가 그 경우를 따로 처리한다.
        if slug in TWO_STAGE and BALD_REF.exists():
            bald = BALD_REF
            mid = _tmp_png(to_pil(model.swap(face, bald, bald)))     # 1차: 대머리
            color = face if keep_color else shape
            out = model.swap(mid, shape, color)                      # 2차: 스타일
        else:
            color = face if keep_color else shape
            out = model.swap(face, shape, color)

        ms = round((time.perf_counter() - t0) * 1000)
        buf = io.BytesIO()
        to_pil(out).save(buf, "PNG")
        return buf.getvalue(), {"elapsed_ms": ms, "vram_peak_mb": vram_peak_mb()}
    finally:
        _rm(face, mid)                                  # 정렬본·중간물도 남기지 않는다


def warm_up():
    """첫 합성은 CUDA 커널 컴파일로 수 분 걸린다. 기동 때 레퍼런스 두 장으로 미리 한 번 돌린다"""
    slugs = available_slugs()
    if len(slugs) < 2:
        print("[warm-up 생략] ref_*.png 가 2장 미만")
        return
    a, b = ref_path(slugs[0]), ref_path(slugs[1])
    t0 = time.perf_counter()
    STATE["model"].swap(a, b, b)
    STATE["warm"] = True
    print(f"[warm-up 완료] {time.perf_counter() - t0:.1f}s  ({a.name} + {b.name})")


# ───────────────────────── 서버 ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import dlib
    from hair_swap import HairFast, get_parser

    if not torch.cuda.is_available():
        sys.exit("[FAIL] CUDA 를 못 쓴다. setup_env.bat 을 먼저 실행할 것")
    if not PREDICTOR_PATH.exists():
        sys.exit(f"[FAIL] 랜드마크 모델 없음: {PREDICTOR_PATH}")

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"레퍼런스: {available_slugs()}")
    print(f"2단계 슬러그: {sorted(TWO_STAGE) or '(없음)'}")
    global GPU
    GPU = asyncio.Semaphore(1)
    print("모델 로딩 중...")
    STATE["predictor"] = dlib.shape_predictor(str(PREDICTOR_PATH))
    STATE["model"] = HairFast(get_parser().parse_args([]))
    await asyncio.to_thread(warm_up)
    yield


app = FastAPI(title="HairMatch inference", lifespan=lifespan)


@app.get("/infer/health")
def health():
    ok = torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info() if ok else (0, 0)
    return {
        "status": "ok" if STATE["model"] else "loading",
        "gpu": torch.cuda.get_device_name(0) if ok else None,
        "vram_total_mb": round(total / 1024 ** 2),
        "vram_free_mb": round(free / 1024 ** 2),
        "warm": STATE["warm"],
        "mock": False,
        "waiting": STATE["waiting"],
    }


@app.get("/infer/styles")
def styles():
    return {"available": available_slugs(), "two_stage": sorted(TWO_STAGE)}


@app.post("/infer/synthesize")
async def synthesize(
    file: UploadFile = File(...),
    style_slug: str = Form(...),
    keep_color: str = Form("true"),
):
    if STATE["model"] is None:
        return err(503, "NOT_READY")
    # 슬러그는 파일 경로가 되므로 목록에 있는 것만 받는다 (../ 같은 입력 차단)
    if style_slug not in available_slugs():
        return err(404, "UNKNOWN_STYLE")

    raw = await file.read(MAX_UPLOAD + 1)
    if not raw or len(raw) > MAX_UPLOAD:
        return err(400, "BAD_IMAGE")

    if STATE["waiting"] >= MAX_WAITING:
        return err(503, "GPU_BUSY")

    # 대기 수는 '세마포어를 기다리는 동안'만 센다. 취소돼도 새지 않게 finally 에서 뺀다
    STATE["waiting"] += 1
    try:
        await GPU.acquire()                              # ★ 한 번에 1건
    finally:
        STATE["waiting"] -= 1

    try:
        png, info = await asyncio.to_thread(
            run_synthesis, raw, style_slug, keep_color.lower() == "true")
    except (UnidentifiedImageError, OSError):
        return err(400, "BAD_IMAGE")
    except ValueError as e:
        code = str(e)
        return err(422, code) if code in ("NO_FACE_DETECTED", "MULTIPLE_FACES") else err(500, "INFERENCE_FAILED")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return err(503, "GPU_BUSY")
    except Exception as e:
        print(f"[synthesize] {type(e).__name__}: {e}")
        return err(500, "INFERENCE_FAILED")
    finally:
        GPU.release()
    del raw

    return Response(content=png, media_type="image/png", headers={
        "X-Elapsed-Ms": str(info["elapsed_ms"]),
        "X-Style-Slug": style_slug,
        "X-Vram-Peak-Mb": str(info["vram_peak_mb"]),
        "Cache-Control": "no-store",
    })


if __name__ == "__main__":
    # 외부에서 직접 부를 일이 없다. web 서버만 localhost 로 부른다
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("INFER_PORT", "8001")))
