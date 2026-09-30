"""
Gemini 헤어 합성 검증 스크립트

prompts.py 의 템플릿으로 실제 API 를 호출해 결과를 뽑고,
기존 HairFastGAN 결과와 나란히 비교할 격자를 만든다.

준비:
    pip install google-genai pillow opencv-python
    set GEMINI_API_KEY=발급받은키

    ※ 이미지 생성 모델은 무료 티어가 없다. 결제 설정이 되어 있어야 한다.

사용:
    # SDK 호출 형태부터 확인 (API 호출 안 함, 과금 없음)
    python gemini_test.py --probe

    # 프롬프트만 출력해서 눈으로 확인 (API 호출 안 함, 과금 없음)
    python gemini_test.py --face ..\\HairFastGAN\\input\\9.jpeg ^
        --styles hush shadow_perm regent --dry-run

    # 실제 호출
    python gemini_test.py --face ..\\HairFastGAN\\input\\9.jpeg ^
        --styles hush shadow_perm regent

    # HairFastGAN 결과와 비교 격자까지
    python gemini_test.py --face ..\\HairFastGAN\\input\\9.jpeg ^
        --styles hush shadow_perm regent ^
        --compare ..\\HairFastGAN\\perf_out

산출물:
    gemini_out/<face>__<slug>.png
    gemini_log.csv
    gemini_grid_<face>.png      (--compare 지정 시)
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import STYLE_PROMPTS, build_prompt

# 모델별 1장당 단가 (USD, 1K 해상도 기준). 공식 가격표가 바뀌면 여기만 고친다.
PRICE_PER_IMAGE = {
    "gemini-3.1-flash-image": 0.067,
    "gemini-3.1-flash-lite-image": 0.0336,
    "gemini-2.5-flash-image": 0.039,
}
DEFAULT_MODEL = "gemini-3.1-flash-image"


def probe():
    """설치된 SDK 가 어떤 호출 형태를 지원하는지 출력한다. API 호출은 하지 않는다."""
    try:
        import google.genai as genai
    except ImportError:
        raise SystemExit("google-genai 미설치: pip install google-genai")

    print("google-genai version:", getattr(genai, "__version__", "unknown"))
    client_cls = genai.Client
    attrs = [a for a in dir(client_cls) if not a.startswith("_")]
    print("Client 속성:", attrs)
    print()
    print("models.generate_content 존재:", hasattr(client_cls, "models"))
    print("interactions 존재:", hasattr(client_cls, "interactions"))
    print()
    print("둘 중 무엇이 있는지 확인 후 generate() 를 맞추면 된다.")


def generate(client, model, prompt, image_bytes, mime):
    """이미지 1장 생성. 성공 시 PNG/JPEG 바이트를 반환."""
    from google.genai import types

    resp = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    for part in resp.parts:
        if getattr(part, "inline_data", None) is not None:
            return part.inline_data.data

    # 이미지가 안 왔으면 텍스트 응답에 이유가 담겨 있는 경우가 많다 (안전 필터 등)
    texts = [p.text for p in resp.parts if getattr(p, "text", None)]
    raise RuntimeError("이미지 미반환" + (f" — 응답: {' '.join(texts)[:300]}" if texts else ""))


def mime_of(path):
    ext = os.path.splitext(path)[1].lower()
    return {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")


def build_grid(face_path, styles, out_dir, compare_dir, tile=300):
    """왼쪽부터 원본 / HairFastGAN 결과 / Gemini 결과."""
    import cv2
    import numpy as np

    def tile_of(path, label, fg=(0, 255, 255)):
        im = cv2.imread(path) if path and os.path.exists(path) else None
        if im is None:
            im = np.full((tile, tile, 3), 40, np.uint8)
            cv2.putText(im, "N/A", (tile // 3, tile // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
        else:
            im = cv2.resize(im, (tile, tile))
        cv2.rectangle(im, (0, 0), (tile, 20), (20, 20, 20), -1)
        cv2.putText(im, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, .38, fg, 1)
        return im

    face_stem = os.path.splitext(os.path.basename(face_path))[0]
    rows = [np.hstack([
        tile_of(face_path, f"FACE {os.path.basename(face_path)}"),
        np.full((tile, tile, 3), 40, np.uint8),
        np.full((tile, tile, 3), 40, np.uint8),
    ])]

    for slug in styles:
        ref = os.path.join(os.path.dirname(face_path), f"ref_{slug}.png")
        hfg = os.path.join(compare_dir, f"{face_stem}__ref_{slug}.png") if compare_dir else None
        gem = os.path.join(out_dir, f"{face_stem}__{slug}.png")
        rows.append(np.hstack([
            tile_of(ref, f"REF {slug}"),
            tile_of(hfg, "HairFastGAN", (0, 200, 255)),
            tile_of(gem, "Gemini", (0, 255, 120)),
        ]))

    path = f"gemini_grid_{face_stem}.png"
    cv2.imwrite(path, np.vstack(rows))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", help="얼굴 이미지 경로")
    ap.add_argument("--styles", nargs="+", default=["hush", "shadow_perm", "regent"],
                    help=f"slug. 가능: {', '.join(sorted(STYLE_PROMPTS))}")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="gemini_out")
    ap.add_argument("--repeat", type=int, default=1,
                    help="같은 조합을 n회 반복 (출력 편차 확인용)")
    ap.add_argument("--compare", default=None,
                    help="HairFastGAN 결과 폴더. 지정 시 비교 격자 생성")
    ap.add_argument("--dry-run", action="store_true", help="프롬프트만 출력, 호출 안 함")
    ap.add_argument("--probe", action="store_true", help="SDK 호출 형태만 확인")
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    if not args.face:
        raise SystemExit("--face 가 필요합니다 (--probe 제외)")
    if not os.path.exists(args.face):
        raise SystemExit(f"파일 없음: {args.face}")

    bad = [s for s in args.styles if s not in STYLE_PROMPTS]
    if bad:
        raise SystemExit(f"등록되지 않은 slug: {bad}\n가능: {sorted(STYLE_PROMPTS)}")

    jobs = [(s, r) for s in args.styles for r in range(args.repeat)]
    unit = PRICE_PER_IMAGE.get(args.model)
    est = f"약 ${unit * len(jobs):.2f}" if unit else "단가 미등록"
    print(f"모델   : {args.model}")
    print(f"얼굴   : {args.face}")
    print(f"스타일 : {args.styles} x {args.repeat}회 = {len(jobs)}장")
    print(f"예상 비용: {est}\n")

    if args.dry_run:
        for slug in args.styles:
            print("=" * 60)
            print(f"[{slug}]")
            print(build_prompt(slug))
        print("=" * 60)
        print("\n--dry-run 이므로 호출하지 않았습니다.")
        return

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("환경변수 GEMINI_API_KEY 가 없습니다.")

    try:
        from google import genai
    except ImportError:
        raise SystemExit("google-genai 미설치: pip install google-genai")

    client = genai.Client(api_key=key)
    os.makedirs(args.out, exist_ok=True)

    face_stem = os.path.splitext(os.path.basename(args.face))[0]
    image_bytes = open(args.face, "rb").read()
    mime = mime_of(args.face)

    rows = []
    for i, (slug, rep) in enumerate(jobs, 1):
        suffix = "" if args.repeat == 1 else f"_r{rep + 1}"
        out_path = os.path.join(args.out, f"{face_stem}__{slug}{suffix}.png")
        rec = {"face": os.path.basename(args.face), "slug": slug, "rep": rep + 1,
               "model": args.model, "status": "fail", "sec": 0.0,
               "out_path": "", "error": ""}

        t = time.time()
        try:
            data = generate(client, args.model, build_prompt(slug), image_bytes, mime)
            with open(out_path, "wb") as fh:
                fh.write(data)
            rec.update(status="ok", out_path=out_path)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["sec"] = round(time.time() - t, 2)

        rows.append(rec)
        print(f"[{i}/{len(jobs)}] {slug}{suffix} -> {rec['status']} ({rec['sec']:.1f}s)"
              + (f"  {rec['error'][:120]}" if rec["error"] else ""))

    with open("gemini_log.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n성공 {len(ok)}/{len(rows)}")
    if ok:
        secs = [r["sec"] for r in ok]
        print(f"평균 {sum(secs)/len(secs):.1f}초 | 최소 {min(secs):.1f} | 최대 {max(secs):.1f}")
    if unit:
        print(f"실제 과금 대상: {len(ok)}장 (약 ${unit * len(ok):.2f})")

    if args.compare and args.repeat == 1:
        try:
            path = build_grid(args.face, args.styles, args.out, args.compare)
            print(f"비교 격자: {path}")
        except Exception as e:
            print(f"[경고] 격자 생성 실패: {type(e).__name__}: {e}")

    print(f"저장: gemini_log.csv, {args.out}/")


if __name__ == "__main__":
    main()
