"""
synth_proxy.py — 브라우저의 /synthesize 를 inference 서버(8001)로 넘긴다

왜 프록시인가
    - web 서버(venv_web, TensorFlow)와 합성 서버(venv_hf, PyTorch+CUDA)는 가상환경이 다르다.
      한 프로세스에 합치면 의존성이 충돌한다 → API_CONTRACT.md 의 2서버 구조를 그대로 따른다.
    - a.html 은 같은 주소(8000)만 부르므로 CORS·포트 문제가 없다.

web_main.py 에 붙이는 법
    from synth_proxy import router as synth_router, infer_available
    app.include_router(synth_router)

필요
    pip install httpx
    set INFERENCE_URL=http://localhost:8001      (기본값이 이것이라 같은 PC면 생략 가능)
"""

import os
import time

import httpx
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

router = APIRouter()

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001").rstrip("/")
# warm-up 전 첫 합성은 수 분 걸릴 수 있다. 너무 짧으면 멀쩡한 합성을 끊는다
TIMEOUT = httpx.Timeout(connect=3.0, read=300.0, write=30.0, pool=5.0)
PASS_HEADERS = ("x-elapsed-ms", "x-style-slug", "x-vram-peak-mb")

_cache = {"at": 0.0, "slugs": None}


def _err(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code}})


async def infer_available() -> set | None:
    """합성 서버가 실제로 만들 수 있는 슬러그. 60초 캐시. 서버가 꺼져 있으면 None(= 거르지 않음)"""
    if _cache["slugs"] is not None and time.time() - _cache["at"] < 60:
        return _cache["slugs"]
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{INFERENCE_URL}/infer/styles")
        slugs = set(r.json().get("available", [])) if r.status_code == 200 else None
    except httpx.HTTPError:
        slugs = None
    _cache.update(at=time.time(), slugs=slugs)
    return slugs


@router.post("/synthesize")
async def synthesize(
    file: UploadFile = File(...),
    style_slug: str = Form(...),
    keep_color: str = Form("true"),
):
    data = await file.read()
    files = {"file": (file.filename or "upload.jpg", data, file.content_type or "image/jpeg")}
    form = {"style_slug": style_slug, "keep_color": keep_color}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.post(f"{INFERENCE_URL}/infer/synthesize", files=files, data=form)
    except httpx.ConnectError:
        # 합성 서버를 안 켰을 때. a.html 은 503 이면 재시도하므로 503 으로 준다
        return _err(503, "INFERENCE_DOWN")
    except httpx.TimeoutException:
        return _err(504, "INFERENCE_TIMEOUT")
    finally:
        del data                                          # 원본을 오래 들고 있지 않는다

    # 성공(이미지)·실패(JSON) 모두 상태 코드와 본문을 그대로 전달한다
    headers = {k: v for k, v in r.headers.items() if k.lower() in PASS_HEADERS}
    headers["Cache-Control"] = "no-store"
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/octet-stream"),
                    headers=headers)


@router.get("/synthesize/health")
async def synth_health():
    """발표·부스 직전 확인용: warm=true, mock=false 인지"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{INFERENCE_URL}/infer/health")
        return r.json()
    except httpx.HTTPError:
        return _err(503, "INFERENCE_DOWN")
