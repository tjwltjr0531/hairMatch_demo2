"""
gemini_api.py — 화면 4 '프롬프트로 직접 만들기' (Gemini 이미지 편집 프록시)

붙이는 법 (salons_api.py 와 같다)
    from gemini_api import router as gemini_router
    app.include_router(gemini_router)

필요한 것
    pip install -U google-genai pillow python-multipart
    setx GEMINI_API_KEY "AI Studio 에서 발급한 키"     ← 새 CMD 창부터 적용

a.html 과 맞춘 계약
    GET  /me/quota   → {"logged_in": true, "remaining": 2}
    POST /generate   (multipart: file, prompt, consent=true) → 이미지 바이트
    에러             → {"error": {"code": "..."}}
        QUOTA_EXCEEDED / CONTENT_BLOCKED / INVALID_PROMPT / CONSENT_REQUIRED /
        BAD_IMAGE / GEMINI_KEY_MISSING / GEN_FAILED

사용자 식별
    ★ 로그인이 아직 없어서, 지금은 브라우저마다 임의 ID 쿠키(hm_uid)를 발급해 센다.
      쿠키를 지우면 횟수가 초기화된다 → 부스 체험용으로는 충분하지만 '계정당 2회'는 아니다.
      로그인이 붙으면 current_user_id() 한 곳만 바꾸면 된다.
"""

import asyncio
import base64
import io
import os
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError

router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")   # 1K 기준 약 $0.0336/장
FREE_LIMIT = 2
MAX_PROMPT = 120
MAX_UPLOAD = 8 * 1024 * 1024          # 8MB
MAX_SIDE = 1024                        # 보내기 전에 줄인다(비용·속도)
DB_PATH = Path(__file__).with_name("quota.db")
COOKIE = "hm_uid"

# 서버에만 있는 실제 템플릿. a.html 의 '실제로 전송되는 내용 보기'는 이 문장의 사본이다.
TEMPLATE = (
    "Change only the hairstyle of the person in this photo to:\n"
    '  "{user}"\n\n'
    "Keep the face, identity, skin tone, expression and background unchanged.\n"
    "Photorealistic, front facing, same lighting as the original."
)

# 헤어와 무관한 요청을 1차로 거른다. 최종 판단은 Gemini 안전 필터가 한다.
BLOCK_WORDS = re.compile(
    # ★ 한 글자 단어는 넣지 않는다. '칼'은 칼단발, '피'는 피부톤에 걸린다
    r"(nude|naked|nsfw|sexy|누드|노출|야한|알몸|벗은)", re.IGNORECASE
)

_lock = asyncio.Lock()      # 쿼터 증감이 겹치지 않게
_client = None


# ───────────────────────── 공통 ─────────────────────────
def _err(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code}})


def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS quota (uid TEXT PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0)")
    return con


def _used(uid: str) -> int:
    with _db() as con:
        row = con.execute("SELECT used FROM quota WHERE uid=?", (uid,)).fetchone()
    return row[0] if row else 0


def _add(uid: str, delta: int) -> None:
    with _db() as con:
        con.execute(
            "INSERT INTO quota(uid, used) VALUES(?, MAX(0, ?)) "
            "ON CONFLICT(uid) DO UPDATE SET used = MAX(0, used + ?)",
            (uid, delta, delta),
        )


def current_user_id(request: Request) -> Optional[str]:
    """★ 로그인 붙으면 여기만 바꾼다. 예) return request.session.get('user_id')"""
    return request.cookies.get(COOKIE)




def _get_client():
    global _client
    if _client is None:
        from google import genai            # 키가 없을 때도 서버는 뜨도록 지연 import
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _prepare_image(raw: bytes) -> bytes:
    """EXIF(위치 등)를 버리고, 긴 변 1024px JPEG 로 다시 만든다."""
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=92)
    return out.getvalue()


def _call_gemini(prompt: str, jpeg: bytes):
    client = _get_client()
    return client.interactions.create(
        model=MODEL,
        input=[
            {"type": "text", "text": prompt},
            {"type": "image", "data": base64.b64encode(jpeg).decode("utf-8"), "mime_type": "image/jpeg"},
        ],
    )


# ───────────────────────── 라우트 ─────────────────────────
@router.get("/me/quota")
async def me_quota(request: Request):
    uid = current_user_id(request)
    is_new = not uid
    if is_new:
        uid = secrets.token_urlsafe(16)
    resp = JSONResponse(content={"logged_in": True, "remaining": max(0, FREE_LIMIT - _used(uid))})
    if is_new:
        resp.set_cookie(COOKIE, uid, max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax")
    return resp


@router.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    consent: str = Form("false"),
):
    if not GEMINI_API_KEY:
        return _err(500, "GEMINI_KEY_MISSING")

    uid = current_user_id(request)
    if not uid:                                   # /me/quota 를 먼저 거쳐야 쿠키가 생긴다
        return _err(401, "NOT_LOGGED_IN")

    # 화면의 체크박스만 믿지 않는다. 서버에서도 동의 여부를 확인한다
    if consent.lower() != "true":
        return _err(400, "CONSENT_REQUIRED")

    text = " ".join(prompt.split())               # 줄바꿈·연속 공백 정리
    if not (2 <= len(text) <= MAX_PROMPT) or BLOCK_WORDS.search(text):
        return _err(400, "INVALID_PROMPT")

    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        return _err(413, "BAD_IMAGE")
    try:
        jpeg = _prepare_image(raw)
    except (UnidentifiedImageError, OSError):
        return _err(400, "BAD_IMAGE")

    # 먼저 1회 차감하고, 실패하면 되돌린다(동시에 두 번 눌러 3회 쓰는 걸 막는다)
    async with _lock:
        if _used(uid) >= FREE_LIMIT:
            return _err(429, "QUOTA_EXCEEDED")
        _add(uid, +1)

    try:
        full_prompt = TEMPLATE.replace("{user}", text)    # format() 은 사용자가 { } 를 쓰면 깨진다
        interaction = await asyncio.to_thread(_call_gemini, full_prompt, jpeg)
        out = getattr(interaction, "output_image", None)
        data = getattr(out, "data", None) if out else None
        if not data:
            # 이미지 없이 끝남 = 대개 안전 필터 차단. 횟수는 돌려준다
            async with _lock:
                _add(uid, -1)
            return _err(422, "CONTENT_BLOCKED")
        img = base64.b64decode(data)
        mime = getattr(out, "mime_type", None) or "image/png"
        return Response(content=img, media_type=mime, headers={"Cache-Control": "no-store"})
    except Exception as e:                         # 네트워크·키·쿼터 등 → 차감 취소
        async with _lock:
            _add(uid, -1)
        # 키 값이 로그에 섞이지 않도록 예외 종류와 코드만 남긴다
        print(f"[gemini] {type(e).__name__} code={getattr(e, 'code', None)}")
        return _err(502, "GEN_FAILED")
