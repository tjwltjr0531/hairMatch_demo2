"""
map_test_server.py — 카카오 지도/미용실 API 만 따로 시험하는 최소 서버

폴더 구성
  web/  map_test_server.py, salons_api.py, gemini_api.py
  web/static/  index.html, a.html, img/

실행 (CMD)
  pip install fastapi uvicorn httpx python-multipart pillow google-genai
  setx KAKAO_REST_KEY "REST_API_키"       ← 실행 후 CMD 창을 새로 열어야 적용
  python map_test_server.py
  → 브라우저에서 http://localhost:8000/ (메인) 또는 /a.html

file:// 로 열면 지도 SDK 가 동작하지 않는다. 반드시 이 서버 주소로 연다.
"""
import os

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from salons_api import router as salons_router
from gemini_api import router as gemini_router

app = FastAPI()
app.include_router(salons_router)                      # GET /salons
app.include_router(gemini_router)                      # GET /me/quota, POST /generate
# 화면 파일은 static/ 폴더에만 둔다. 이 파일이 있는 폴더를 통째로 공개하면
# web_main.py, gemini_api.py, quota.db 같은 서버 파일까지 브라우저로 내려받을 수 있다.
BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
# 정적 파일은 라우터 뒤에 마운트한다. 먼저 마운트하면 /salons 도 파일로 찾으려 한다
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

if __name__ == "__main__":
    print("KAKAO_REST_KEY 설정됨:", bool(os.getenv("KAKAO_REST_KEY")))   # 키 값 자체는 출력하지 않는다
    print("GEMINI_API_KEY 설정됨:", bool(os.getenv("GEMINI_API_KEY")))
    uvicorn.run(app, host="127.0.0.1", port=8000)
