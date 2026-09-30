"""
HairMatch DB 스키마 (PostgreSQL / SQLAlchemy)

기존 db_setup.py 대비 바뀐 점
    1. users 테이블 제거 — 회원가입 없는 구조로 결정됨
    2. 얼굴형 코드를 실제 모델 클래스명으로 수정
       기존 'Long Face' -> 실제는 'Oblong'. 이 불일치로 DB 조회가 항상 실패했음
    3. 얼굴형당 스타일 1개 -> 여러 개 (rank 순) — 확률 기반 다중 추천 대응
    4. synth_results 테이블 추가 — 사전 생성한 합성 이미지 인덱스
    5. psycopg2 직접 호출 -> SQLAlchemy. DATABASE_URL 만 바꾸면 SQLite 로도 동작

[스타일 교체 — 실제 레퍼런스 사진 기준]
    - HAIRSTYLES 를 HairFastGAN/input/ref_{slug}.png 가 있는 10개로 교체 (성별당 5개)
    - 얼굴형마다 같은 5개를 순서만 다르게 → 어떤 결과든 추천 카드 5장
    - seed() 가 추천을 '교체'하도록 변경 (이전엔 없을 때만 추가해서 옛 추천이 남았음)
    - ref_image_path 자동 입력

사용:
    pip install sqlalchemy psycopg2-binary
    set DATABASE_URL=postgresql+psycopg2://postgres:비밀번호@localhost:5432/hairmatch_db

    python db_setup_v2.py            # 테이블 생성 + 시드 데이터 (추천은 새 목록으로 교체)
    python db_setup_v2.py --drop     # 전부 삭제 후 재생성
    python db_setup_v2.py --show     # 현재 내용 확인

SQLite 로 쓰려면:
    set DATABASE_URL=sqlite:///hairmatch.db
"""

import argparse
import os
from datetime import datetime

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String, Text,
                        UniqueConstraint, create_engine, select)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:changeme@localhost:5432/hairmatch_db",
)

Base = declarative_base()

# 모델이 출력하는 클래스명 (dataset_cropped 폴더명 = 알파벳 정렬 순서)
# 이 순서가 예측 인덱스 순서이며, 절대 임의로 바꾸면 안 된다.
FACE_SHAPES = [
    ("Heart",  "하트형"),
    ("Oblong", "긴형"),
    ("Oval",   "계란형"),
    ("Round",  "둥근형"),
    ("Square", "사각형"),
]


class FaceShape(Base):
    """얼굴형 마스터. code 는 모델 클래스명과 정확히 일치해야 한다."""
    __tablename__ = "face_shapes"

    code = Column(String(20), primary_key=True)     # Heart, Oblong, ...
    name_ko = Column(String(20), nullable=False)    # 하트형, 긴형, ...
    description = Column(Text)

    recommendations = relationship("Recommendation", back_populates="face_shape")


class Hairstyle(Base):
    """스타일 마스터. ref_image_path 는 합성 시 shape 레퍼런스로 쓰인다."""
    __tablename__ = "hairstyles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    slug = Column(String(40), nullable=False, unique=True)   # bob, layered, ...
    name_ko = Column(String(40), nullable=False)
    gender = Column(String(10), nullable=False)              # male / female / unisex
    ref_image_path = Column(String(255))                     # 1024x1024 정렬 이미지
    description = Column(Text)

    recommendations = relationship("Recommendation", back_populates="hairstyle")
    synth_results = relationship("SynthResult", back_populates="hairstyle")


class Recommendation(Base):
    """(얼굴형 x 성별) 조합별 추천 스타일. rank 1 이 1순위."""
    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint("face_shape_code", "gender", "rank", name="uq_reco_rank"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    face_shape_code = Column(String(20), ForeignKey("face_shapes.code"), nullable=False)
    gender = Column(String(10), nullable=False)
    hairstyle_id = Column(Integer, ForeignKey("hairstyles.id"), nullable=False)
    rank = Column(Integer, nullable=False, default=1)
    advice = Column(Text)

    face_shape = relationship("FaceShape", back_populates="recommendations")
    hairstyle = relationship("Hairstyle", back_populates="recommendations")


class DemoFace(Base):
    """사전 생성용 데모 얼굴. 실시간 추론 대신 미리 만들어 둔 결과를 보여주기 위함."""
    __tablename__ = "demo_faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(40), nullable=False)          # 팀원A, 데모1 등
    gender = Column(String(10), nullable=False)
    face_shape_code = Column(String(20), ForeignKey("face_shapes.code"))
    image_path = Column(String(255), nullable=False)    # 1024x1024 정렬 원본
    consent = Column(String(10), default="yes")         # 초상권 사용 동의 기록

    synth_results = relationship("SynthResult", back_populates="demo_face")


class SynthResult(Base):
    """HairFastGAN 배치 생성 결과 인덱스. 웹은 이 경로를 정적 서빙한다."""
    __tablename__ = "synth_results"
    __table_args__ = (
        UniqueConstraint("demo_face_id", "hairstyle_id", name="uq_synth"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    demo_face_id = Column(Integer, ForeignKey("demo_faces.id"), nullable=False)
    hairstyle_id = Column(Integer, ForeignKey("hairstyles.id"), nullable=False)
    image_path = Column(String(255), nullable=False)
    keep_color = Column(String(10), default="yes")      # color=face 여부
    elapsed_sec = Column(Integer)                        # 생성 소요 (실측 기록용)
    created_at = Column(DateTime, default=datetime.utcnow)

    demo_face = relationship("DemoFace", back_populates="synth_results")
    hairstyle = relationship("Hairstyle", back_populates="synth_results")


# ---------------------------------------------------------------
# 시드 데이터
# ---------------------------------------------------------------
# ★ 레퍼런스 사진(HairFastGAN/input/ref_{slug}.png)이 실제로 있는 스타일만 둔다.
#   슬러그 = 파일명. 여기와 파일명이 다르면 추천에서 빠진다 (/recommend 가 거른다).
# (slug, name_ko, gender)
HAIRSTYLES = [
    # --- female ---
    ("tucked_bob",  "턱선 단발",   "female"),
    ("hush",        "허쉬컷",      "female"),
    ("long_wave",   "롱 웨이브",   "female"),
    ("full_bang",   "풀뱅 롱",     "female"),
    ("gh",          "글램 펌",     "female"),
    # --- male ---
    ("regent",      "리젠트 컷",   "male"),
    ("dandy",       "댄디컷",      "male"),
    ("asperm",      "애즈펌",      "male"),
    ("leaf",        "리프 컷",     "male"),
    ("hippie",      "히피펌",      "male"),
]

REF_DIR = "input"   # HairFastGAN 폴더 기준 상대경로 (ref_image_path 표기용)

# (face_shape_code, gender, style_slug, rank, advice)
# 얼굴형마다 같은 5개를 순서만 다르게 — 어떤 결과가 나와도 카드가 5장 나온다.
# 순서 기준(일반 스타일링 원칙, 팀 검토용 초안):
#   Oval  : 대부분 어울림        Round : 세로선, 옆 볼륨은 뒤로
#   Oblong: 앞머리·옆 볼륨으로 길이 감소
#   Square: 곡선으로 턱선 완화   Heart : 이마는 가리고 턱 쪽 볼륨
RECOMMENDATIONS = [
    # --- female ---
    ("Oval",   "female", "tucked_bob", 1, "균형 잡힌 얼굴형이라 턱선 단발의 또렷한 라인이 잘 살아납니다."),
    ("Oval",   "female", "hush",       2, "가벼운 레이어가 얼굴선을 자연스럽게 감쌉니다."),
    ("Oval",   "female", "long_wave",  3, "긴 웨이브로 부드럽고 여성스러운 인상을 줍니다."),
    ("Oval",   "female", "gh",         4, "풍성한 컬로 화려한 분위기를 더합니다."),
    ("Oval",   "female", "full_bang",  5, "앞머리로 어려 보이는 인상을 연출할 수 있습니다."),

    ("Round",  "female", "long_wave",  1, "긴 세로선이 얼굴을 갸름하고 길어 보이게 합니다."),
    ("Round",  "female", "hush",       2, "옆 레이어가 볼 라인을 가려 윤곽이 정리돼 보입니다."),
    ("Round",  "female", "tucked_bob", 3, "턱 아래로 떨어지는 길이를 고르면 무난합니다."),
    ("Round",  "female", "full_bang",  4, "무거운 앞머리는 얼굴을 짧아 보이게 할 수 있어 가볍게 내려 주세요."),
    ("Round",  "female", "gh",         5, "옆 볼륨이 커지면 얼굴이 넓어 보일 수 있어 정수리 볼륨을 권합니다."),

    ("Oblong", "female", "full_bang",  1, "앞머리가 이마를 덮어 얼굴 길이를 짧아 보이게 합니다."),
    ("Oblong", "female", "gh",         2, "옆으로 퍼지는 컬이 가로 폭을 더해 균형을 잡아 줍니다."),
    ("Oblong", "female", "tucked_bob", 3, "턱선 길이의 단발이 세로 길이를 끊어 줍니다."),
    ("Oblong", "female", "hush",       4, "얼굴 옆 레이어로 가로 볼륨을 살려 보세요."),
    ("Oblong", "female", "long_wave",  5, "긴 기장은 세로선을 강조하므로 웨이브를 크게 넣는 것이 좋습니다."),

    ("Square", "female", "long_wave",  1, "부드러운 웨이브가 각진 턱선을 자연스럽게 완화합니다."),
    ("Square", "female", "gh",         2, "둥근 컬이 직선적인 윤곽을 부드럽게 보이게 합니다."),
    ("Square", "female", "hush",       3, "턱 주변 레이어가 각을 가려 시선을 분산시킵니다."),
    ("Square", "female", "full_bang",  4, "앞머리 끝을 가볍게 하면 이마·턱 폭의 대비를 줄일 수 있습니다."),
    ("Square", "female", "tucked_bob", 5, "턱선에서 끝나는 기장은 각을 강조할 수 있어 길이 조절이 필요합니다."),

    ("Heart",  "female", "tucked_bob", 1, "턱 주변 볼륨이 좁은 턱을 보완해 균형을 맞춥니다."),
    ("Heart",  "female", "full_bang",  2, "앞머리가 넓은 이마를 가려 이마·턱 비율을 맞춰 줍니다."),
    ("Heart",  "female", "gh",         3, "아래쪽에 풍성한 컬이 뾰족한 턱 끝을 보완합니다."),
    ("Heart",  "female", "long_wave",  4, "어깨 아래 웨이브로 얼굴 하단에 무게를 실어 보세요."),
    ("Heart",  "female", "hush",       5, "윗부분 볼륨은 줄이고 아래 레이어를 살리는 것이 좋습니다."),

    # --- male ---
    ("Oval",   "male", "regent",       1, "이마를 드러내 시원하고 깔끔한 인상을 줍니다."),
    ("Oval",   "male", "dandy",        2, "내린 앞머리로 부드럽고 단정한 느낌을 줍니다."),
    ("Oval",   "male", "asperm",       3, "적당한 볼륨의 컬로 자연스러운 분위기를 냅니다."),
    ("Oval",   "male", "leaf",         4, "가벼운 가르마 라인으로 부드러운 인상을 강조합니다."),
    ("Oval",   "male", "hippie",       5, "풍성한 컬로 개성 있는 스타일을 시도해 볼 수 있습니다."),

    ("Round",  "male", "regent",       1, "윗머리를 세우고 옆은 짧게 해 얼굴이 길어 보입니다."),
    ("Round",  "male", "asperm",       2, "정수리 볼륨으로 세로 비율을 살려 줍니다."),
    ("Round",  "male", "leaf",         3, "가르마로 이마를 일부 드러내 답답함을 줄입니다."),
    ("Round",  "male", "dandy",        4, "앞머리를 모두 내리면 얼굴이 둥글어 보일 수 있어 가볍게 넘겨 주세요."),
    ("Round",  "male", "hippie",       5, "옆 볼륨이 커지면 얼굴이 넓어 보일 수 있습니다."),

    ("Oblong", "male", "dandy",        1, "내린 앞머리가 얼굴 길이를 짧아 보이게 합니다."),
    ("Oblong", "male", "hippie",       2, "옆으로 퍼지는 볼륨이 가로 폭을 더해 줍니다."),
    ("Oblong", "male", "leaf",         3, "앞머리를 일부 내려 이마 노출을 줄여 보세요."),
    ("Oblong", "male", "asperm",       4, "윗볼륨은 낮추고 옆 볼륨을 살리는 방향이 좋습니다."),
    ("Oblong", "male", "regent",       5, "윗머리를 세우면 얼굴이 더 길어 보일 수 있습니다."),

    ("Square", "male", "asperm",       1, "부드러운 컬이 각진 턱선을 완화합니다."),
    ("Square", "male", "leaf",         2, "자연스러운 가르마 곡선으로 직선적인 인상을 덜어 줍니다."),
    ("Square", "male", "dandy",        3, "앞머리로 시선을 위로 모아 턱선 강조를 줄입니다."),
    ("Square", "male", "hippie",       4, "컬이 윤곽을 부드럽게 하지만 옆 볼륨은 조절이 필요합니다."),
    ("Square", "male", "regent",       5, "남성적인 인상이 강해지므로 옆머리를 너무 짧게 하지 않는 것이 좋습니다."),

    ("Heart",  "male", "dandy",        1, "앞머리가 넓은 이마를 가려 균형을 맞춰 줍니다."),
    ("Heart",  "male", "leaf",         2, "가벼운 앞머리로 이마 폭을 조절합니다."),
    ("Heart",  "male", "hippie",       3, "아래쪽 볼륨이 좁은 턱을 보완합니다."),
    ("Heart",  "male", "asperm",       4, "윗볼륨을 과하게 넣지 않는 선에서 자연스럽게 연출하세요."),
    ("Heart",  "male", "regent",       5, "이마를 모두 드러내면 이마 폭이 강조될 수 있습니다."),
]


def seed(session):
    for code, ko in FACE_SHAPES:
        if not session.get(FaceShape, code):
            session.add(FaceShape(code=code, name_ko=ko))
    session.flush()

    slug_to_id = {}
    for slug, ko, gender in HAIRSTYLES:
        ref = f"{REF_DIR}/ref_{slug}.png"
        row = session.scalar(select(Hairstyle).where(Hairstyle.slug == slug))
        if row is None:
            row = Hairstyle(slug=slug, name_ko=ko, gender=gender, ref_image_path=ref)
            session.add(row)
            session.flush()
        else:                                   # 이름·경로가 바뀌었으면 갱신
            row.name_ko, row.gender, row.ref_image_path = ko, gender, ref
        slug_to_id[slug] = row.id

    # ★ 추천은 매번 이 목록으로 '교체'한다.
    #   예전 방식(없을 때만 추가)은 옛 추천(wave_bob 등)이 남아 새 순서가 반영되지 않았다.
    #   예전 스타일 행(hairstyles)은 synth_results 가 참조할 수 있어 지우지 않는다.
    session.query(Recommendation).delete()
    session.flush()
    for code, gender, slug, rank, advice in RECOMMENDATIONS:
        session.add(Recommendation(
            face_shape_code=code, gender=gender,
            hairstyle_id=slug_to_id[slug], rank=rank, advice=advice))

    session.commit()


def show(session):
    print("=== face_shapes ===")
    for fs in session.scalars(select(FaceShape).order_by(FaceShape.code)):
        print(f"  {fs.code:8s} {fs.name_ko}")

    print("\n=== hairstyles ===")
    for hs in session.scalars(select(Hairstyle).order_by(Hairstyle.id)):
        ref = hs.ref_image_path or '(레퍼런스 미지정)'
        print(f"  [{hs.id:2d}] {hs.slug:14s} {hs.name_ko:16s} {hs.gender:7s} {ref}")

    print("\n=== recommendations ===")
    q = (select(Recommendation)
         .order_by(Recommendation.face_shape_code, Recommendation.gender,
                   Recommendation.rank))
    for r in session.scalars(q):
        print(f"  {r.face_shape_code:8s} {r.gender:7s} #{r.rank} "
              f"{r.hairstyle.name_ko:16s} {(r.advice or '')[:30]}")

    n_face = session.scalar(select(DemoFace).limit(1))
    n_synth = len(list(session.scalars(select(SynthResult))))
    print(f"\n=== 사전 생성 ===")
    print(f"  demo_faces  : {'있음' if n_face else '없음 (아직 등록 안 됨)'}")
    print(f"  synth_results: {n_synth}건")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop', action='store_true', help='모든 테이블 삭제 후 재생성')
    ap.add_argument('--show', action='store_true', help='현재 내용만 출력')
    args = ap.parse_args()

    masked = DATABASE_URL
    if '@' in masked:
        head, tail = masked.split('@', 1)
        masked = head.split('//')[0] + '//***@' + tail
    print(f"DB: {masked}\n")

    engine = create_engine(DATABASE_URL, echo=False, future=True)
    Session = sessionmaker(bind=engine, future=True)

    if args.show:
        with Session() as s:
            show(s)
        return

    if args.drop:
        if input("모든 테이블을 삭제합니다. 계속할까요? [y/N] ").strip().lower() != 'y':
            return
        Base.metadata.drop_all(engine)
        print("기존 테이블 삭제 완료")

    Base.metadata.create_all(engine)
    print("테이블 생성 완료:", ", ".join(Base.metadata.tables.keys()))

    with Session() as s:
        seed(s)
        print("\n시드 데이터 입력 완료\n")
        show(s)

    print("\n다음 단계")
    print("  1. 추천 카드가 적게 나오면 web_main 창의 '[recommend] ... 제외' 로그로 슬러그 확인")
    print("  2. 스타일을 추가하려면 ref_{slug}.png 를 넣고 HAIRSTYLES / RECOMMENDATIONS 에 같은 슬러그로 추가")


if __name__ == "__main__":
    main()