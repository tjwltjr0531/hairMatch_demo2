"""
HairFastGAN 성능 확인 — 배치 합성 + 속도/VRAM 측정

main.py 를 조합마다 실행하면 매번 모델을 다시 로드해 느리다.
이 스크립트는 HairFast 인스턴스를 한 번만 만들고 반복 호출한다.

사용:
    python test_hairfast.py
    python test_hairfast.py --input-dir input --faces 6.png 1.png --shapes 7.png 8.png
    python test_hairfast.py --keep-color        # color=face 로 원본 머리색 유지

산출물:
    perf_out/          개별 결과 이미지
    perf_grid.png      얼굴 x 스타일 격자 비교
    perf_log.csv       조합별 소요 시간
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
except ImportError:
    raise SystemExit("opencv 미설치: pip install opencv-python")

from torchvision.utils import save_image

from hair_swap import HairFast, get_parser


def vram_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def valid_1024(path):
    im = cv2.imread(path)
    return im is not None and im.shape[0] == 1024 and im.shape[1] == 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='input')
    ap.add_argument('--faces', nargs='*', default=None,
                    help='지정하지 않으면 1024 이미지 중 앞 2장을 얼굴로 사용')
    ap.add_argument('--shapes', nargs='*', default=None,
                    help='지정하지 않으면 나머지를 스타일 레퍼런스로 사용')
    ap.add_argument('--keep-color', action='store_true',
                    help='color 를 face 로 지정해 원본 머리색 유지')
    ap.add_argument('--out', default='perf_out')
    args = ap.parse_args()

    print('=== 환경 ===')
    print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU  ', torch.cuda.get_device_name(0),
              f'({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)')
    else:
        print('★ GPU 미사용 — CPU 로는 매우 느리거나 실패할 수 있습니다.')

    # 1024 이미지만 선별 (해상도가 섞이면 tensor size mismatch 로 죽는다)
    all_imgs = sorted(f for f in os.listdir(args.input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    ok = [f for f in all_imgs if valid_1024(os.path.join(args.input_dir, f))]
    skipped = [f for f in all_imgs if f not in ok]
    print(f'\n1024x1024 이미지 {len(ok)}개: {ok}')
    if skipped:
        print(f'제외됨 (해상도 불일치): {skipped}')
    if len(ok) < 2:
        raise SystemExit('1024 이미지가 2장 이상 필요합니다.')

    faces = args.faces or ok[:2]
    shapes = args.shapes or [f for f in ok if f not in faces][:4] or ok[1:2]
    print(f'\nFACE  : {faces}')
    print(f'SHAPE : {shapes}')
    print(f'COLOR : {"face (원본색 유지)" if args.keep_color else "shape 와 동일"}')

    os.makedirs(args.out, exist_ok=True)

    print('\n=== 모델 로드 (첫 실행은 CUDA 커널 컴파일로 수 분) ===')
    t0 = time.time()
    hair_fast = HairFast(get_parser().parse_args([]))
    print(f'로드 완료: {time.time()-t0:.1f}초')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f'로드 후 VRAM: {vram_mb():.0f} MB')

    rows = []
    results = {}
    total = len(faces) * len(shapes)
    idx = 0

    for face in faces:
        for shape in shapes:
            idx += 1
            color = face if args.keep_color else shape
            name = f'{os.path.splitext(face)[0]}__{os.path.splitext(shape)[0]}.png'
            out_path = os.path.join(args.out, name)

            t = time.time()
            try:
                img = hair_fast.swap(
                    os.path.join(args.input_dir, face),
                    os.path.join(args.input_dir, shape),
                    os.path.join(args.input_dir, color),
                )
                save_image(img, out_path)
                elapsed = time.time() - t
                results[(face, shape)] = out_path
                status = 'ok'
                err = ''
            except Exception as e:
                elapsed = time.time() - t
                status, err = 'fail', f'{type(e).__name__}: {e}'
                print(f'  [실패] {face} x {shape}: {err}')

            rows.append({'face': face, 'shape': shape, 'color': color,
                         'status': status, 'sec': round(elapsed, 2),
                         'vram_mb': round(vram_mb() or 0), 'error': err})
            print(f'[{idx}/{total}] {face} x {shape} -> {status} ({elapsed:.1f}s)')

    with open('perf_log.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    times = [r['sec'] for r in rows if r['status'] == 'ok']
    print('\n=== 요약 ===')
    print(f'성공 {len(times)}/{len(rows)}')
    if times:
        print(f'평균 {np.mean(times):.2f}초 | 최소 {min(times):.2f} | 최대 {max(times):.2f}')
        if len(times) > 1:
            print(f'첫 실행 제외 평균: {np.mean(times[1:]):.2f}초')
    if torch.cuda.is_available():
        print(f'VRAM 최고: {vram_mb():.0f} MB')

    # 격자 이미지: 1열은 원본 얼굴, 나머지는 스타일별 결과
    print('\n격자 생성 중...')
    grid_rows = []
    for face in faces:
        tiles = []
        im = cv2.imread(os.path.join(args.input_dir, face))
        t = cv2.resize(im, (280, 280))
        cv2.rectangle(t, (0, 0), (280, 22), (20, 20, 20), -1)
        cv2.putText(t, f'ORIG {face}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (0, 255, 255), 1)
        tiles.append(t)

        for shape in shapes:
            p = results.get((face, shape))
            if p and os.path.exists(p):
                im = cv2.resize(cv2.imread(p), (280, 280))
            else:
                im = np.full((280, 280, 3), 40, np.uint8)
                cv2.putText(im, 'FAILED', (80, 145), cv2.FONT_HERSHEY_SIMPLEX, .8,
                            (0, 0, 255), 2)
            cv2.rectangle(im, (0, 0), (280, 22), (20, 20, 20), -1)
            cv2.putText(im, f'shape {shape}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (0, 255, 255), 1)
            tiles.append(im)
        grid_rows.append(np.hstack(tiles))

    w = min(r.shape[1] for r in grid_rows)
    cv2.imwrite('perf_grid.png', np.vstack([r[:, :w] for r in grid_rows]))
    print('저장: perf_grid.png, perf_log.csv, ' + args.out + '/')
    print('\n★ 격자에서 확인할 것')
    print('  1. 본인으로 인식되는가 (정체성 보존)')
    print('  2. 스타일이 서로 구별되는가')
    print('  3. 헤어 경계와 배경이 자연스러운가')


if __name__ == '__main__':
    main()
