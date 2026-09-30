"""
StyleGAN2(FFHQ)로 헤어 레퍼런스 후보 얼굴 생성

HairFastGAN 이 이미 쓰고 있는 pretrained_models/StyleGAN/ffhq.pt 를 그대로 사용한다.
생성 결과는 실존 인물이 아니므로 초상권/저작권 문제가 없고,
StyleGAN2 FFHQ 출력은 원래부터 1024x1024 FFHQ 정렬이라 align_face.py 를 거칠 필요가 없다.

사용:
    cd D:\\hairmatch\\HairFastGAN
    venv_hf\\Scripts\\activate

    python gen_refs.py                          # 시드 0~99, 100장 + 컨택트시트
    python gen_refs.py --n 200 --start-seed 0   # 200장
    python gen_refs.py --seeds 7 23 41          # 특정 시드만 다시 뽑기
    python gen_refs.py --trunc 0.5              # 더 평균적인 얼굴 (다양성 감소)

산출물:
    gen_out/seed_0007.png ...   개별 1024x1024
    gen_contact.png             인덱스가 찍힌 격자 (여기서 골라라)

고른 뒤:
    copy gen_out\\seed_0023.png input\\ref_wave_bob.png

주의:
    - truncation 이 낮을수록 얼굴이 평범해지고 헤어 다양성이 줄어든다. 0.7 권장.
    - CPU 로도 동작하지만 1024 생성은 장당 수십 초 이상 걸린다. GPU 권장.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
except ImportError:
    raise SystemExit("opencv 미설치: pip install opencv-python")

from models.stylegan2.model import Generator

CKPT = os.path.join("pretrained_models", "StyleGAN", "ffhq.pt")


def build_generator(device):
    if not os.path.exists(CKPT):
        raise SystemExit(f"체크포인트 없음: {CKPT}")

    # hair_swap.get_parser() 의 기본값과 동일하게 맞춘다.
    # size=1024, latent=512, n_mlp=8, channel_multiplier=2
    g = Generator(1024, 512, 8, channel_multiplier=2)

    print(f"체크포인트 로드: {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu")
    g.load_state_dict(ckpt["g_ema"])
    g.eval().to(device)
    for p in g.parameters():
        p.requires_grad = False
    return g


def to_bgr(tensor):
    """[-1,1] float CHW -> uint8 BGR HWC"""
    x = tensor.clamp(-1, 1).add(1).div(2).mul(255)
    x = x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="생성 장수")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="지정하면 이 시드들만 생성 (--n, --start-seed 무시)")
    ap.add_argument("--trunc", type=float, default=0.7,
                    help="truncation psi. 낮을수록 평범, 높을수록 다양 (0.5~0.9)")
    ap.add_argument("--batch", type=int, default=4, help="VRAM 부족하면 줄여라")
    ap.add_argument("--out", default="gen_out")
    ap.add_argument("--contact", default="gen_contact.png")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"device: {device}")
    if device == "cpu":
        print("★ CPU 모드 — 1024 생성은 장당 수십 초 이상 걸립니다.")

    seeds = args.seeds if args.seeds else list(
        range(args.start_seed, args.start_seed + args.n))

    g = build_generator(device)

    # truncation 기준점. rosinality Generator 의 mean_latent 는 [1, 512] 를 돌려준다.
    print("mean latent 계산 중...")
    with torch.no_grad():
        mean_latent = g.mean_latent(4096)

    os.makedirs(args.out, exist_ok=True)
    saved = []

    for i in range(0, len(seeds), args.batch):
        chunk = seeds[i:i + args.batch]

        # 시드별로 개별 생성해야 시드 -> 이미지가 재현 가능하다.
        zs = []
        for s in chunk:
            gen = torch.Generator(device="cpu").manual_seed(int(s))
            zs.append(torch.randn(1, 512, generator=gen))
        z = torch.cat(zs).to(device)

        with torch.no_grad():
            imgs, _ = g([z], truncation=args.trunc,
                        truncation_latent=mean_latent, randomize_noise=False)

        for s, img in zip(chunk, imgs):
            path = os.path.join(args.out, f"seed_{s:04d}.png")
            cv2.imwrite(path, to_bgr(img))
            saved.append((s, path))

        print(f"  {min(i + args.batch, len(seeds))}/{len(seeds)}")

        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- 컨택트시트: 시드 번호를 찍어서 고르기 쉽게 ----
    cols = 10
    cell = 200
    rows = (len(saved) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 30, np.uint8)

    for idx, (s, path) in enumerate(saved):
        im = cv2.resize(cv2.imread(path), (cell, cell))
        cv2.rectangle(im, (0, 0), (cell, 20), (20, 20, 20), -1)
        cv2.putText(im, f"{s}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 255, 255), 1)
        r, c = divmod(idx, cols)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = im

    cv2.imwrite(args.contact, sheet)

    print(f"\n완료: {len(saved)}장 -> {args.out}/")
    print(f"컨택트시트: {args.contact}")
    print("\n다음 단계")
    print("  1. 컨택트시트에서 스타일별로 맞는 시드 번호를 고른다")
    print("  2. copy gen_out\\seed_XXXX.png input\\ref_<슬러그>.png")
    print("  3. hairstyles.ref_image_path 에 그 경로를 UPDATE")
    print("\n필요한 슬러그 12개")
    print("  female: wave_bob layered side_bang c_curl full_bang hush")
    print("  male  : leaf regent pomade shadow_perm side_volume two_block")


if __name__ == "__main__":
    main()
