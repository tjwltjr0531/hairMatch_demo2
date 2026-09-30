"""
FFHQ 얼굴 정렬 — GPU / CUDA 없이 동작하는 버전

기존 scripts/align_face.py 는 utils.image_utils 를 import 하는데,
그 모듈이 models.Net -> models.stylegan2 를 끌고 와서 StyleGAN CUDA 커널을
JIT 컴파일하려다 CUDA_HOME 오류로 죽는다.

정렬에 실제로 필요한 건 utils.shape_predictor.align_face (dlib + PIL + scipy)
뿐이므로, 파일 목록 함수만 직접 구현해서 그 의존성을 끊었다.
결과는 기존 스크립트와 동일하다 (같은 align_face 를 호출).

사용 (반드시 HairFastGAN 폴더에서):
    cd /d D:\\hairmatch\\HairFastGAN
    ..\\venv_hf\\Scripts\\activate
    python align_simple.py
    python align_simple.py -unprocessed_dir unprocessed -output_dir input
    python align_simple.py --replace        # 이미 있는 것도 덮어쓰기

필요:
    pretrained_models/ShapeAdaptor/shape_predictor_68_face_landmarks.dat
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pillow 10 부터 Image.ANTIALIAS 가 제거됐다. shape_predictor 내부에서
# 아주 큰 원본(qsize >= 4096)일 때만 쓰이지만, 만나면 죽으므로 미리 채워둔다.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

import dlib  # noqa: E402
from utils.shape_predictor import align_face  # noqa: E402

PREDICTOR = os.path.join("pretrained_models", "ShapeAdaptor",
                         "shape_predictor_68_face_landmarks.dat")
EXTS = (".jpg", ".jpeg", ".png")


def list_images(d):
    """utils.image_utils.list_image_files 와 동일한 동작 (import 회피용)"""
    return [f for f in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(EXTS)]


def report(path):
    """정렬 전 원본이 얼마나 확대/축소되는지 참고용으로 출력"""
    try:
        w, h = Image.open(path).size
        return f"{w}x{h}"
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser(description="Align faces (CPU only)")
    ap.add_argument("-unprocessed_dir", type=Path, default=Path("unprocessed"))
    ap.add_argument("-output_dir", type=Path, default=Path("input"))
    ap.add_argument("--replace", action="store_true",
                    help="output 에 같은 이름이 있어도 덮어쓴다")
    args = ap.parse_args()

    if not args.unprocessed_dir.is_dir():
        raise SystemExit(f"폴더 없음: {args.unprocessed_dir}")
    if not os.path.isfile(PREDICTOR):
        raise SystemExit(
            f"랜드마크 모델 없음: {PREDICTOR}\n"
            "HairFastGAN 폴더에서 실행하고 있는지 확인하세요.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = dlib.shape_predictor(PREDICTOR)
    files = list_images(args.unprocessed_dir)
    if not files:
        raise SystemExit(f"{args.unprocessed_dir} 에 이미지가 없습니다.")

    print(f"대상 {len(files)}장\n")
    ok = skip = fail = 0

    for name in files:
        src = args.unprocessed_dir / name
        dst = args.output_dir / name

        if dst.is_file() and not args.replace:
            print(f"  [건너뜀] {name} (이미 있음, 덮어쓰려면 --replace)")
            skip += 1
            continue

        try:
            imgs = align_face(str(src), predictor=predictor,
                              is_filepath=True, return_tensors=False)
        except Exception as e:
            print(f"  [실패] {name}: {type(e).__name__}: {e}")
            fail += 1
            continue

        if len(imgs) == 0:
            print(f"  [실패] {name}: 얼굴 미검출")
            fail += 1
            continue
        if len(imgs) > 1:
            print(f"  [주의] {name}: 얼굴 {len(imgs)}개 검출 — 첫 번째만 저장")

        out = imgs[0]
        if out.size != (1024, 1024):
            out = out.resize((1024, 1024), Image.LANCZOS)
        out.save(dst)
        print(f"  [완료] {name}  ({report(src)} -> 1024x1024)")
        ok += 1

    print(f"\n성공 {ok} / 건너뜀 {skip} / 실패 {fail}")
    print(f"출력: {args.output_dir}/")
    if ok:
        print("\n다음: test_hairfast.py 로 합성 (GPU 필요)")


if __name__ == "__main__":
    main()
