"""
HairFastGAN 파라미터 스윕 — mixing x smooth 격자 비교

기존 파일(test_hairfast.py, hair_swap.py, models/*)은 수정하지 않는다.
모델은 한 번만 로드하고, 조합마다 값만 바꿔서 재실행한다.

[코드 확인 결과 — 왜 재로드가 필요 없는가]
  mixing : models/Embedding.py:92 에서 self.opts.mixing 을 '호출 시점'에 읽는다.
           (embedding_images() 안. __init__ 에 캐시되지 않음)
           -> hair_fast.args.mixing 을 바꾸면 다음 swap() 부터 바로 반영된다.
           단, len(images_to_name) > 1 (face/shape/color 가 전부 같은 이미지가
           아닐 때)에만 mixing 분기가 실행된다.
  smooth : models/Alignment.py:40, models/Blending.py:32 에서
           __init__ 시점에 DilateErosion(dilate_erosion=self.opts.smooth) 로
           고정된다. -> args.smooth 만 바꾸면 효과가 없다.
           다만 DilateErosion 은 self.dilate_erosion(반복 횟수 int)와
           고정 3x3 커널만 들고 있으므로(utils/image_utils.py:27-55),
           그 정수 필드를 직접 갈아끼우면 재로드 없이 동일한 효과가 난다.
  둘 다 **kwargs 로는 전달되지 않는다. kwargs 는 exp_name 정도만 소비된다.

사용:
    python param_sweep.py --face 9.jpeg --shape ref_regent.png --keep-color

    # --shape 는 여러 개 가능. 스타일마다 격자를 따로 만든다 (모델은 여전히 1회 로드)
    python param_sweep.py --face 9.jpeg --keep-color \
        --shape ref_regent.png ref_as_perm.png ref_crop.png

    # 값 지정 / 지표 생략
    python param_sweep.py --face 9.jpeg --shape ref_regent.png --keep-color \
        --mixing 0.85 --smooth 1 2 3
    python param_sweep.py --face 9.jpeg --shape ref_regent.png --no-metrics

산출물:
    sweep_out/<face>__<shape>/m{mixing}_s{smooth}.png
    sweep_grid_<face>__<shape>.png     스타일마다 1장. 행=smooth, 열=mixing
    sweep_log.csv                      전체 스타일 합쳐 1개 (shape 열로 구분)
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
except ImportError:
    raise SystemExit("opencv 미설치: pip install opencv-python")

import torchvision.transforms as T
from torchvision.utils import save_image

from hair_swap import HairFast, get_parser

TO_BISENET = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HAIR_LABEL = 13  # CelebAMask-HQ 기준 머리카락 라벨 (models/Net.py, Embedding.py 와 동일)


# --------------------------------------------------------------------------
# 파라미터 주입
# --------------------------------------------------------------------------

def apply_params(hair_fast, mixing, smooth):
    """모델 재로드 없이 mixing / smooth 를 교체한다."""
    # mixing: Embedding 이 호출 시점에 opts.mixing 을 읽는다.
    # HairFast.__init__ 이 같은 Namespace 객체를 embed/align/blend 에 전달하므로
    # args 하나만 바꾸면 전부에 반영된다.
    hair_fast.args.mixing = float(mixing)

    # smooth: 기록용으로 args 도 맞춰두되, 실제 효력은 아래 두 줄이다.
    hair_fast.args.smooth = int(smooth)
    hair_fast.align.dilate_erosion.dilate_erosion = int(smooth)
    hair_fast.blend.dilate_erosion.dilate_erosion = int(smooth)


def verify_params(hair_fast, mixing, smooth):
    """주입이 실제로 먹었는지 확인 (조용히 실패하는 것을 막기 위함)."""
    assert hair_fast.embed.opts.mixing == float(mixing), 'mixing 주입 실패'
    assert hair_fast.align.dilate_erosion.dilate_erosion == int(smooth), 'align smooth 주입 실패'
    assert hair_fast.blend.dilate_erosion.dilate_erosion == int(smooth), 'blend smooth 주입 실패'


# --------------------------------------------------------------------------
# 정량 지표 (머리 길이 / 레퍼런스 일치도)
# --------------------------------------------------------------------------

@torch.inference_mode()
def hair_mask(img_chw, device):
    """[3,H,W] 0..1 텐서 -> 머리 마스크 [1,1,256,256] (0/1 float)"""
    from models.Net import get_segmentation  # hair_swap import 시 이미 로드됨

    x = img_chw.unsqueeze(0).to(device).float().clamp(0, 1)
    x = TF.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
    x = TO_BISENET(x[0]).unsqueeze(0)
    mask = get_segmentation(x)  # resize=True -> [1,1,256,256]
    return (mask == HAIR_LABEL).float()


def mask_stats(mask):
    """면적비 / 최상단·최하단 y (0~1 정규화). 최하단 y 가 머리 길이 대용 지표."""
    m = mask[0, 0].cpu().numpy()
    h = m.shape[0]
    ys = np.where(m.sum(axis=1) > 0)[0]
    if len(ys) == 0:
        return dict(hair_ratio=0.0, hair_top=float('nan'), hair_bottom=float('nan'))
    return dict(hair_ratio=float(m.mean()),
                hair_top=float(ys.min() / h),
                hair_bottom=float(ys.max() / h))


def iou(a, b):
    inter = (a * b).sum().item()
    union = ((a + b).clamp(0, 1)).sum().item()
    return inter / union if union > 0 else float('nan')


# --------------------------------------------------------------------------

def label_tile(im, text, bg=(20, 20, 20), fg=(0, 255, 255)):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 22), bg, -1)
    cv2.putText(im, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, fg, 1)
    return im


def sweep_one(hair_fast, args, device, p_face, shape, color, combos, prog):
    """shape 하나에 대해 mixing x smooth 격자를 돌리고 (rows, grid_path) 반환."""
    p_shape = os.path.join(args.input_dir, shape)
    p_color = os.path.join(args.input_dir, color)

    pair = f'{os.path.splitext(args.face)[0]}__{os.path.splitext(shape)[0]}'
    out_dir = os.path.join(args.out, pair)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n===== SHAPE: {shape}  (COLOR: {color}) =====')

    # 레퍼런스 머리 마스크 (지표 기준선)
    ref_mask = ref_stats = None
    if not args.no_metrics:
        try:
            from torchvision.io import read_image, ImageReadMode
            ref_img = read_image(p_shape, mode=ImageReadMode.RGB).float() / 255
            ref_mask = hair_mask(ref_img, device)
            ref_stats = mask_stats(ref_mask)
            print(f'레퍼런스 머리: 면적 {ref_stats["hair_ratio"]:.3f} | '
                  f'하단 y {ref_stats["hair_bottom"]:.3f}')
        except Exception as e:
            print(f'[경고] 레퍼런스 지표 계산 실패, 지표 없이 진행: {type(e).__name__}: {e}')
            ref_mask = None

    rows, results = [], {}
    for m, s in combos:
        prog['i'] += 1
        apply_params(hair_fast, m, s)
        verify_params(hair_fast, m, s)

        out_path = os.path.join(out_dir, f'm{m:g}_s{s}.png')
        rec = {'face': args.face, 'shape': shape, 'color': color,
               'mixing': m, 'smooth': s, 'status': 'fail', 'sec': 0.0,
               'vram_mb': 0, 'hair_ratio': '', 'hair_top': '', 'hair_bottom': '',
               'iou_vs_shape': '', 'error': ''}

        t = time.time()
        try:
            img = hair_fast.swap(p_face, p_shape, p_color, seed=args.seed)
            rec['sec'] = round(time.time() - t, 2)
            save_image(img, out_path)
            results[(m, s)] = out_path
            rec['status'] = 'ok'

            if ref_mask is not None:
                try:
                    mk = hair_mask(img, device)
                    rec.update({k: round(v, 4) for k, v in mask_stats(mk).items()})
                    rec['iou_vs_shape'] = round(iou(mk, ref_mask), 4)
                except Exception as e:
                    rec['error'] = f'metric: {type(e).__name__}: {e}'
        except Exception as e:
            rec['sec'] = round(time.time() - t, 2)
            rec['error'] = f'{type(e).__name__}: {e}'
            print(f'  [실패] mixing={m} smooth={s}: {rec["error"]}')

        if device == 'cuda':
            rec['vram_mb'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2)

        rows.append(rec)
        extra = ''
        if rec['status'] == 'ok' and rec['hair_bottom'] != '' and ref_stats:
            extra = (f" | 하단y {rec['hair_bottom']:.3f}"
                     f" (ref {ref_stats['hair_bottom']:.3f})"
                     f" | IoU {rec['iou_vs_shape']}")
        print(f'[{prog["i"]}/{prog["total"]}] {shape} mixing={m:<5g} smooth={s} -> '
              f'{rec["status"]} ({rec["sec"]:.1f}s){extra}')

    # ---- 격자: 맨 위 줄은 원본/레퍼런스, 이후 행=smooth, 열=mixing ----
    tile = args.tile
    ncol = len(args.mixing) + 1
    grid = []

    head = [label_tile(cv2.resize(cv2.imread(p_face), (tile, tile)), f'FACE {args.face}'),
            label_tile(cv2.resize(cv2.imread(p_shape), (tile, tile)), f'SHAPE {shape}')]
    while len(head) < ncol:
        head.append(np.full((tile, tile, 3), 40, np.uint8))
    grid.append(np.hstack(head[:ncol]))

    for s in args.smooth:
        tiles = [label_tile(np.full((tile, tile, 3), 40, np.uint8), f'smooth = {s}')]
        for m in args.mixing:
            p = results.get((m, s))
            if p and os.path.exists(p):
                t_im = cv2.resize(cv2.imread(p), (tile, tile))
            else:
                t_im = np.full((tile, tile, 3), 40, np.uint8)
                cv2.putText(t_im, 'FAILED', (tile // 4, tile // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
            tiles.append(label_tile(t_im, f'mix {m:g} / sm {s}'))
        grid.append(np.hstack(tiles))

    w_min = min(r.shape[1] for r in grid)
    grid_path = f'sweep_grid_{pair}.png'
    cv2.imwrite(grid_path, np.vstack([r[:, :w_min] for r in grid]))

    # shape 별 베스트
    cand = [r for r in rows
            if r['status'] == 'ok' and r['iou_vs_shape'] != '' and r['hair_bottom'] != ''
            and not np.isnan(r['iou_vs_shape']) and not np.isnan(r['hair_bottom'])]
    if cand and ref_stats:
        bi = max(cand, key=lambda r: r['iou_vs_shape'])
        bl = min(cand, key=lambda r: abs(r['hair_bottom'] - ref_stats['hair_bottom']))
        print(f'  -> IoU 최대   : mixing={bi["mixing"]:g} smooth={bi["smooth"]} '
              f'(IoU {bi["iou_vs_shape"]})')
        print(f'  -> 길이 최근접: mixing={bl["mixing"]:g} smooth={bl["smooth"]} '
              f'(하단y {bl["hair_bottom"]:.3f} vs ref {ref_stats["hair_bottom"]:.3f})')

    return rows, grid_path, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='input')
    ap.add_argument('--face', required=True, help='얼굴 이미지 파일명 (1024x1024 정렬본)')
    ap.add_argument('--shape', nargs='+', required=True,
                    help='헤어스타일 레퍼런스 파일명. 여러 개 주면 스타일별로 격자를 따로 만든다')
    ap.add_argument('--color', default=None,
                    help='색상 레퍼런스 (모든 shape 에 공통 적용). 미지정 시 각 shape 자신')
    ap.add_argument('--keep-color', action='store_true', help='color=face (원본 머리색 유지)')
    ap.add_argument('--mixing', nargs='+', type=float, default=[0.5, 0.7, 0.85, 0.95])
    ap.add_argument('--smooth', nargs='+', type=int, default=[3, 5, 7])
    ap.add_argument('--seed', type=int, default=3407, help='hair_swap 기본값과 동일')
    ap.add_argument('--out', default='sweep_out')
    ap.add_argument('--tile', type=int, default=280)
    ap.add_argument('--no-metrics', action='store_true',
                    help='머리 마스크 기반 지표 계산 생략 (조합당 약간 빨라짐)')
    args = ap.parse_args()

    if args.keep_color and args.color:
        raise SystemExit('--keep-color 와 --color 는 함께 쓸 수 없습니다.')

    shapes = list(dict.fromkeys(args.shape))  # 중복 제거, 순서 유지
    if args.keep_color:
        colors = {sh: args.face for sh in shapes}
    elif args.color:
        colors = {sh: args.color for sh in shapes}
    else:
        colors = {sh: sh for sh in shapes}

    p_face = os.path.join(args.input_dir, args.face)
    need = [p_face] + [os.path.join(args.input_dir, x)
                       for x in list(shapes) + list(colors.values())]
    missing = [p for p in dict.fromkeys(need) if not os.path.exists(p)]
    if missing:
        raise SystemExit('파일 없음:\n  ' + '\n  '.join(missing) +
                         f'\n  -> {args.input_dir}\\ 안의 실제 파일명인지 확인하세요.')

    # 해상도가 섞이면 파이프라인 깊은 곳에서 tensor size mismatch 로 죽는다.
    # align_simple.py 를 거친 1024x1024 정렬본만 받는다.
    bad = []
    for p in dict.fromkeys(need):
        im = cv2.imread(p)
        if im is None:
            bad.append(f'{p} (읽기 실패)')
        elif im.shape[0] != 1024 or im.shape[1] != 1024:
            bad.append(f'{p} ({im.shape[1]}x{im.shape[0]})')
    if bad:
        raise SystemExit('1024x1024 정렬본이 아닙니다:\n  ' + '\n  '.join(bad) +
                         '\n  -> unprocessed\\ 에 넣고 python align_simple.py 를 먼저 실행하세요.')

    same = [sh for sh in shapes
            if os.path.abspath(p_face) == os.path.abspath(os.path.join(args.input_dir, sh))
            == os.path.abspath(os.path.join(args.input_dir, colors[sh]))]
    if same:
        raise SystemExit(f'face/shape/color 가 모두 같은 이미지인 shape: {same}\n'
                         '  mixing 분기가 실행되지 않습니다 '
                         '(Embedding.py:85 len(images_to_name) > 1 조건).')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=== 환경 ===')
    print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())
    if device == 'cpu':
        print('★ GPU 미사용 — 매우 느리거나 실패할 수 있습니다.')

    combos = [(m, s) for s in args.smooth for m in args.mixing]
    total = len(combos) * len(shapes)
    print(f'\nFACE  : {args.face}')
    print(f'SHAPE : {shapes}')
    print(f'COLOR : {"face (원본색 유지)" if args.keep_color else (args.color or "각 shape 자신")}')
    print(f'mixing: {args.mixing}')
    print(f'smooth: {args.smooth}')
    print(f'조합  : {len(combos)}개 x 스타일 {len(shapes)}개 = {total}회')

    print('\n=== 모델 로드 (첫 실행은 CUDA 커널 컴파일로 수 분) ===')
    t0 = time.time()
    hair_fast = HairFast(get_parser().parse_args([]))
    print(f'로드 완료: {time.time()-t0:.1f}초')
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    all_rows, grid_paths, out_dirs = [], [], []
    prog = {'i': 0, 'total': total}
    for sh in shapes:
        rows, gp, od = sweep_one(hair_fast, args, device, p_face, sh, colors[sh], combos, prog)
        all_rows += rows
        grid_paths.append(gp)
        out_dirs.append(od)

    csv_path = 'sweep_log.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    ok = [r for r in all_rows if r['status'] == 'ok']
    print('\n=== 전체 요약 ===')
    print(f'성공 {len(ok)}/{len(all_rows)}')
    if ok:
        secs = [r['sec'] for r in ok]
        print(f'평균 {np.mean(secs):.2f}초 | 최소 {min(secs):.2f} | 최대 {max(secs):.2f}')
    if device == 'cuda':
        print(f'VRAM 최고: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB')
    print('  ※ 지표는 후보 좁히기용. 최종 판단은 격자 눈으로 볼 것.')

    print(f'\n저장: {csv_path}')
    for gp, od in zip(grid_paths, out_dirs):
        print(f'      {gp}, {od}/')


if __name__ == '__main__':
    main()
