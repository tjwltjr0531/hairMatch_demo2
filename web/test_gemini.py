"""
test_gemini.py — 서버 없이 Gemini 키가 동작하는지 한 장만 확인한다 (약 $0.03 과금)

    python test_gemini.py 내사진.jpg
    → gemini_out.png 가 생기면 성공
"""
import base64
import os
import sys

from google import genai

if not os.getenv("GEMINI_API_KEY"):
    sys.exit("GEMINI_API_KEY 가 없습니다. setx 후 CMD 창을 새로 여세요.")
if len(sys.argv) < 2:
    sys.exit("사용법: python test_gemini.py 사진파일")

with open(sys.argv[1], "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

client = genai.Client()     # GEMINI_API_KEY 를 자동으로 읽는다
interaction = client.interactions.create(
    model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image"),
    input=[
        {"type": "text", "text": "Change only the hairstyle of the person in this photo to: "
                                 "\"chin-length bob\". Keep the face, identity and background unchanged."},
        {"type": "image", "data": data, "mime_type": "image/jpeg"},
    ],
)
out = getattr(interaction, "output_image", None)
if not out or not out.data:
    sys.exit("이미지가 오지 않았습니다(안전 필터 차단 가능). 다른 사진으로 시도해 보세요.")
with open("gemini_out.png", "wb") as f:
    f.write(base64.b64decode(out.data))
print("성공: gemini_out.png")
