"""
salons_api.py — 근처 미용실 조회 (카카오 로컬 API 프록시)

web_main.py 에 붙이는 법 (둘 중 하나)
  A) 이 파일을 web/ 폴더에 두고 web_main.py 에서
        from salons_api import router as salons_router
        app.include_router(salons_router)
  B) 아래 함수 본문을 web_main.py 에 그대로 복사

필요한 것
  pip install httpx
  setx KAKAO_REST_KEY "카카오_REST_API_키"      ← 새 CMD 창부터 적용된다

왜 서버에서 부르나
  카카오 로컬 API 는 'REST API 키'를 쓴다. 브라우저 JS 에 넣으면 누구나 볼 수 있고,
  남이 그 키로 호출하면 우리 앱의 일일 쿼터가 깎인다.
  지도 표시에 쓰는 'JavaScript 키'는 등록한 도메인에서만 동작하도록 돼 있어서 브라우저에 둬도 된다.

a.html 이 기대하는 응답 (배열)
  [{ place_name, road_address_name, address_name, phone,
     distance(미터, 문자열), x(경도), y(위도), place_url, category_name }, ...]
  → 카카오 응답의 documents 필드 이름을 그대로 쓴다. 변환할 게 없다.

에러는 a.html 과 맞춘 형식으로 돌려준다:  {"error": {"code": "..."}}
"""

import os

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter()

KAKAO_REST_KEY = os.getenv("KAKAO_REST_KEY")
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

# a.html 로 넘길 필드만 추린다
KEEP = ("place_name", "road_address_name", "address_name", "phone",
        "distance", "x", "y", "place_url", "category_name")


def _err(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code}})


@router.get("/salons")
async def salons(
    lat: float = Query(..., ge=33.0, le=39.0),        # 대한민국 범위 밖 좌표는 거절
    lng: float = Query(..., ge=124.0, le=132.0),
    radius: int = Query(3000, ge=100, le=20000),       # 카카오 허용 범위: 0 ~ 20000m
):
    if not KAKAO_REST_KEY:
        return _err(500, "KAKAO_KEY_MISSING")

    params = {
        "query": "미용실",
        "x": lng,            # ★ 카카오는 x = 경도, y = 위도. 순서를 바꾸면 바다 한가운데가 나온다
        "y": lat,
        "radius": radius,
        "sort": "distance",
        "size": 15,          # 한 페이지 최대 15
    }
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(KAKAO_KEYWORD_URL, params=params, headers=headers)
    except httpx.HTTPError:
        return _err(502, "MAP_API_UNREACHABLE")

    if r.status_code != 200:
        # 401: 키 오류 / 403: 카카오맵 API 비활성화(OPEN_MAP_AND_LOCAL) 가 흔한 원인
        return _err(502, f"MAP_API_{r.status_code}")

    docs = r.json().get("documents", [])

    # '미용실' 키워드에 네일·피부관리 등이 섞여 나올 수 있어 카테고리로 한 번 더 거른다.
    # 걸러서 하나도 안 남으면 원래 결과를 그대로 쓴다(카테고리 표기가 예상과 다를 때 대비).
    only_hair = [d for d in docs if "미용실" in (d.get("category_name") or "")]
    docs = only_hair or docs

    return [{k: d.get(k) for k in KEEP} for d in docs]
