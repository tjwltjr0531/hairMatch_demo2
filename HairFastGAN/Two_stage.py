"""
two_stage.py — 1단계 합성 vs 2단계(대머리 경유) 합성 비교 실험

목적
  경로 A (1단계) : swap(face=원본,       shape=스타일, color=원본)
  경로 B (2단계) : swap(face=원본,       shape=대머리, color=대머리)  -> 중간물
                   swap(face=중간물,     shape=스타일, color=원본)

  B가 A보다 확실히 나을 때만 채택한다. 비슷하면 A가 절반 시간에 절반 리스크다.

실행
  1) 새 CMD 창에서
       D:\\hairmatch\\setup_env.bat
  2) 정렬본 준비 (unprocessed\\ -> input\\)
       python align_simple.py
  3) 실험
       python test_hairfast.py --faces  --shapes 

전제
  - D:\\hairmatch\\HairFastGAN 에서 실행할 것 (pretrained_models 상대경로 때문)
  - 모든 입력은 input\\ 안의 1024x1024 정렬본이어야 한다
  - ref_bald.png 가 아직 없으면 --make-bald-only 로 1차만 먼저 확인할 것

출력 (기본 two_stage_out\\)
  bald_<face>.png        1차 결과 (대머리 중간물) -- 이것부터 눈으로 확인
  A_<style>.png          경로 A 결과
  B_<style>.png          경로 B 결과
  compare_grid.png       비교 격자
  two_stage_log.csv      시간 / VRAM / 신원 유사도
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# Pillow 12에서 ANTIALIAS가 제거됨. HairFastGAN 내부가 참조하는 경우가 있어 먼저 채워둔다.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

from hair_swap import HairFast, get_parser

IN_DIR = Path("input")
DEFAULT_OUT = Path("two_stage_out")


# ---------------------------------------------------------------- 유틸

def die(msg):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def check_1024(path: Path):
    """test_hairfast.py의 valid_1024와 같은 검사. --shapes가 이 검사를 우회해서
    2048 이미지가 들어갔다가 텐서 크기 불일치로 죽은 적이 있다."""
    if not path.exists():
        die(f"파일이 없다: {path}\n       align_simple.py 를 먼저 돌렸는지 확인할 것")
    with Image.open(path) as im:
        w, h = im.size
    if (w, h) != (1024, 1024):
        die(f"{path.name} 이 {w}x{h} 다. 1024x1024 여야 한다.\n"
            f"       원본을 unprocessed\\ 에 넣고 align_simple.py 로 다시 정렬할 것.\n"
            f"       (직접 리사이즈하면 랜드마크가 틀어져서 결과가 망가진다)")
    return path


def to_pil(t: torch.Tensor) -> Image.Image:
    """HairFast 출력 텐서를 PIL로. float[0,1] / uint8 둘 다 받는다."""
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dtype == torch.uint8:
        arr = t.permute(1, 2, 0).numpy()
    else:
        arr = (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def vram_reset():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def vram_peak():
    """(allocated, reserved) MB. nvidia-smi 수치에 가까운 건 reserved 쪽이다."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() / 1024 ** 2,
            torch.cuda.max_memory_reserved() / 1024 ** 2)


def run_swap(model, face, shape, color, label):
    """1회 합성. 시간/VRAM을 재고 OOM을 잡는다."""
    print(f"  [{label}] face={Path(str(face)).name}  "
          f"shape={Path(str(shape)).name}  color={Path(str(color)).name}")
    vram_reset()
    t0 = time.perf_counter()
    try:
        out = model.swap(face, shape, color)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  [{label}] !! CUDA OOM -- 서로 다른 이미지 3장이 8GB를 넘겼다")
        return None, {"label": label, "sec": None, "alloc_mb": None,
                      "reserved_mb": None, "error": "OOM"}
    sec = time.perf_counter() - t0
    alloc, reserved = vram_peak()
    print(f"  [{label}] {sec:.2f}s   VRAM alloc {alloc:.0f}MB / reserved {reserved:.0f}MB")
    return out, {"label": label, "sec": round(sec, 2),
                 "alloc_mb": round(alloc), "reserved_mb": round(reserved), "error": ""}


# ---------------------------------------------------------------- ArcFace (선택)

class Identity:
    """원본 대비 신원 유사도. ArcFace 모듈을 못 찾으면 조용히 비활성화된다.
    실험의 핵심 판단(1차가 깔끔한 대머리를 만드는가)은 눈으로 하는 것이므로
    이게 없어도 실험 자체는 성립한다."""

    CKPT = Path("pretrained_models/ArcFace/ir_se50.pth")
    CANDIDATES = [
        "models.encoder4editing.models.encoders.model_irse",
        "models.encoders.model_irse",
        "encoder4editing.models.encoders.model_irse",
        "models.face_parsing.model_irse",
    ]

    def __init__(self):
        self.net = None
        if not self.CKPT.exists():
            print(f"[skip] ArcFace 체크포인트 없음 ({self.CKPT}) -- 신원 유사도 생략")
            return
        Backbone = None
        for mod in self.CANDIDATES:
            try:
                Backbone = __import__(mod, fromlist=["Backbone"]).Backbone
                print(f"[ok] ArcFace 백본: {mod}")
                break
            except Exception:
                continue
        if Backbone is None:
            print("[skip] ArcFace 백본 모듈을 못 찾음 -- 신원 유사도 생략")
            print("       (시간/VRAM/육안 비교는 그대로 진행된다)")
            return
        try:
            net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se")
            net.load_state_dict(torch.load(self.CKPT, map_location="cpu"))
            self.net = net.eval().cuda() if torch.cuda.is_available() else net.eval()
        except Exception as e:
            print(f"[skip] ArcFace 로드 실패: {type(e).__name__}: {e}")
            self.net = None

    @torch.no_grad()
    def feat(self, pil: Image.Image):
        if self.net is None:
            return None
        import torchvision.transforms.functional as TF
        x = TF.to_tensor(pil.resize((256, 256), Image.BILINEAR)).unsqueeze(0)
        x = x[:, :, 35:223, 32:220]                       # pSp id_loss와 같은 크롭
        x = torch.nn.functional.adaptive_avg_pool2d(x, (112, 112))
        x = x * 2 - 1                                     # [0,1] -> [-1,1]
        if torch.cuda.is_available():
            x = x.cuda()
        return torch.nn.functional.normalize(self.net(x), dim=1)

    def sim(self, a, b):
        fa, fb = self.feat(a), self.feat(b)
        if fa is None or fb is None:
            return None
        return round(float((fa * fb).sum()), 4)


# ---------------------------------------------------------------- 격자

def label_tile(img: Image.Image, text: str, size=320):
    tile = img.convert("RGB").resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, size, 26], fill=(20, 20, 20))
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:                                     # Pillow < 10.1
        font = ImageFont.load_default()
    d.text((6, 5), text[:38], fill=(255, 220, 0), font=font)
    return tile


def build_grid(rows, out_path, size=320):
    """rows: [[(PIL, label), ...], ...]  행 길이가 달라도 된다."""
    rows = [r for r in rows if r]
    if not rows:
        return
    cols = max(len(r) for r in rows)
    canvas = Image.new("RGB", (cols * size, len(rows) * size), (12, 12, 12))
    for ri, row in enumerate(rows):
        for ci, (img, lab) in enumerate(row):
            canvas.paste(label_tile(img, lab, size), (ci * size, ri * size))
    canvas.save(out_path)
    print(f"\n[격자] {out_path}")


# ---------------------------------------------------------------- 본체

def main():
    ap = argparse.ArgumentParser(description="1단계 vs 2단계 합성 비교")
    ap.add_argument("--face", required=True,
                    help="사용자 사진 (input\\ 안의 파일명)")
    ap.add_argument("--bald", default="ref_bald.png",
                    help="대머리 레퍼런스 (input\\ 안의 파일명)")
    ap.add_argument("--styles", nargs="+", default=[],
                    help="스타일 레퍼런스들 (input\\ 안의 파일명)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--make-bald-only", action="store_true",
                    help="1차(대머리)만 만들고 끝낸다. 제일 먼저 이걸로 확인할 것")
    ap.add_argument("--skip-a", action="store_true",
                    help="경로 A를 건너뛴다 (이미 돌려봤을 때)")
    ap.add_argument("--bald-color-from-face", action="store_true",
                    help="1차의 color를 원본에서 가져온다. 기본은 대머리 레퍼런스에서 "
                         "가져오는데, 그래야 blend 단계가 헤어색을 되살리지 않는다")
    args = ap.parse_args()

    if not Path("hair_swap.py").exists():
        die("HairFastGAN 폴더에서 실행할 것 (예: cd /d D:\\hairmatch\\HairFastGAN)")
    if not torch.cuda.is_available():
        die("CUDA를 못 쓴다. setup_env.bat 을 먼저 실행할 것")

    face = check_1024(IN_DIR / args.face)
    bald = check_1024(IN_DIR / args.bald)
    styles = [check_1024(IN_DIR / s) for s in args.styles]

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    print(f"\nGPU  : {torch.cuda.get_device_name(0)}")
    print(f"얼굴  : {face.name}")
    print(f"대머리: {bald.name}")
    print(f"스타일: {[s.name for s in styles] or '(없음)'}")
    print("\n모델 로딩 중... (첫 실행이면 CUDA 커널 컴파일로 수 분 걸린다)")

    model = HairFast(get_parser().parse_args([]))
    ident = Identity()
    log = []

    orig_pil = Image.open(face).convert("RGB")

    # ---- 1차: 대머리 ----------------------------------------------------
    # color를 대머리 레퍼런스로 두면 hair_swap.py의 `shape is not color` 분기가
    # 걸리지 않아서 blend가 align_shape를 그대로 쓴다. 원본에서 color를 가져오면
    # 없앤 머리색이 되살아날 수 있다.
    print("\n=== 1차: 대머리 만들기 ===")
    bald_color = face if args.bald_color_from_face else bald
    bald_t, rec = run_swap(model, face, bald, bald_color, "1차-대머리")
    log.append(rec)
    if bald_t is None:
        die("1차 합성이 OOM으로 실패했다. 여기서 멈춘다")

    bald_pil = to_pil(bald_t)
    bald_path = out_dir / f"bald_{face.stem}.png"
    bald_pil.save(bald_path)                 # 파일로 저장해서 2차에 경로로 넘긴다
    print(f"  -> {bald_path}")               # (텐서를 직접 넘기면 dtype이 섞인다)

    if ident.net is not None:
        s = ident.sim(orig_pil, bald_pil)
        print(f"  신원 유사도(원본 vs 대머리): {s}")
        log[-1]["identity"] = s

    if args.make_bald_only or not styles:
        build_grid([[(orig_pil, "ORIGINAL"), (bald_pil, "STAGE1 BALD")]],
                   out_dir / "compare_grid.png")
        print("\n" + "=" * 62)
        print("  먼저 bald_*.png 를 열어서 확인할 것:")
        print("   - 머리카락이 실제로 사라졌는가")
        print("   - 이마와 두피가 자연스러운가 (피부톤, 경계)")
        print("   - 얼굴이 본인으로 보이는가")
        print("  여기서 결과가 나쁘면 2단계 구조는 성립하지 않는다.")
        print("=" * 62)
        write_log(out_dir, log)
        return

    # ---- 2차 + 경로 A ---------------------------------------------------
    row_a = [(orig_pil, "ORIGINAL")]
    row_b = [(bald_pil, "STAGE1 BALD")]

    for sp in styles:
        name = sp.stem.replace("ref_", "")
        print(f"\n=== {name} ===")

        if not args.skip_a:
            # A: 원본에서 바로. color=face 라서 원래 머리색이 유지된다.
            a_t, rec = run_swap(model, face, sp, face, f"A-{name}")
            log.append(rec)
            if a_t is not None:
                a_pil = to_pil(a_t)
                a_pil.save(out_dir / f"A_{name}.png")
                row_a.append((a_pil, f"A (1-stage) {name}"))
                if ident.net is not None:
                    s = ident.sim(orig_pil, a_pil)
                    print(f"  신원 유사도: {s}")
                    log[-1]["identity"] = s

        # B: 대머리 중간물 위에. color는 반드시 '원본'이어야 한다.
        #    face(대머리)에서 색을 가져오면 머리카락이 없어서 색이 엉뚱하게 나온다.
        #    -- 이미지 3장이 전부 달라서 shape_module이 한 번 더 돈다 (느리고 VRAM↑)
        b_t, rec = run_swap(model, bald_path, sp, face, f"B-{name}")
        log.append(rec)
        if b_t is not None:
            b_pil = to_pil(b_t)
            b_pil.save(out_dir / f"B_{name}.png")
            row_b.append((b_pil, f"B (2-stage) {name}"))
            if ident.net is not None:
                s = ident.sim(orig_pil, b_pil)
                print(f"  신원 유사도: {s}")
                log[-1]["identity"] = s

    build_grid([row_a, row_b], out_dir / "compare_grid.png")
    write_log(out_dir, log)
    summarize(log)


def write_log(out_dir, log):
    path = out_dir / "two_stage_log.csv"
    cols = ["label", "sec", "alloc_mb", "reserved_mb", "identity", "error"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in log:
            w.writerow(r)
    print(f"[로그] {path}")


def summarize(log):
    ok = [r for r in log if r.get("sec") is not None]
    if not ok:
        return
    a = [r for r in ok if r["label"].startswith("A-")]
    b = [r for r in ok if r["label"].startswith("B-")]
    peak = max(r["reserved_mb"] for r in ok)

    print("\n" + "=" * 62)
    print("  요약")
    if a:
        print(f"   A 평균 {sum(r['sec'] for r in a) / len(a):.2f}s")
    if b:
        print(f"   B 평균 {sum(r['sec'] for r in b) / len(b):.2f}s  (+1차 1회)")
    print(f"   VRAM 최대 reserved {peak:.0f}MB / 8192MB", end="")
    print("   <-- 한계 근접, 주의" if peak > 7600 else "")

    ids_a = [r["identity"] for r in a if r.get("identity") is not None]
    ids_b = [r["identity"] for r in b if r.get("identity") is not None]
    if ids_a and ids_b:
        ma, mb = sum(ids_a) / len(ids_a), sum(ids_b) / len(ids_b)
        print(f"   신원 유사도  A {ma:.4f}   B {mb:.4f}   차이 {mb - ma:+.4f}")
        if mb < ma - 0.05:
            print("   -> B에서 신원이 뚜렷하게 흐려졌다. 누적 열화 의심")

    print("\n   판단 기준: compare_grid.png 에서 B가 A보다 '확실히' 나을 때만 채택.")
    print("   비슷하면 A가 절반 시간에 절반 리스크다.")
    print("=" * 62)


if __name__ == "__main__":
    main()