"""
9단계: 학습된 하이브리드 분류기로 얼굴형 예측 (서빙용)

hair_demo/face_shape.py 를 대체합니다. 08 이 저장한 hybrid_model.joblib 하나만
있으면 학습 때와 완전히 같은 전처리·스케일링·보정을 재현합니다.

라이브러리로 쓰기
    from importlib import import_module
    predictor = import_module("09_predict").FaceShapePredictor()
    result = predictor.predict_file("input/photo.jpg")
    # {'shape': 'Oval', 'confidence': 0.62,
    #  'top2': [('Oval', 0.62), ('Heart', 0.21)],
    #  'probs': {'Heart': 0.21, 'Oblong': 0.08, ...}}

    # 파일명이 숫자로 시작해 import 가 불편하면 09_predict.py 를
    # face_shape_cnn.py 같은 이름으로 복사해서 쓰세요.

명령줄로 쓰기
    python 09_predict.py --image input/photo.jpg
    python 09_predict.py --dir input/

Top-2 를 쓰는 편이 정직합니다. 테스트 기준 Top-1 은 74.5%, Top-2 는 91.0% 라
"계란형 62% / 하트형 21%" 처럼 두 후보를 함께 보여주고 양쪽에 어울리는
스타일을 추천하면 단일 라벨을 단정하는 것보다 실제 만족도가 높을 가능성이 큽니다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FACE_SHAPE_KOREAN = {
    "Heart": "하트형", "Oblong": "긴 얼굴형", "Oval": "계란형",
    "Round": "둥근형", "Square": "각진형",
}

# 04_prep_square.py / 07_extract_features.py 와 동일해야 한다
CONTOUR_PAIRS = [
    (103, 332), (54, 284), (21, 251), (162, 389), (127, 356),
    (234, 454), (93, 323), (132, 361), (58, 288), (172, 397),
    (136, 365), (150, 379),
]


class FaceShapePredictor:
    def __init__(self, bundle="hybrid_model.joblib", crop_size=256,
                 margin=0.30, up_bias=0.12, task_model="face_landmarker.task"):
        import joblib
        import tensorflow as tf
        import mediapipe as mp

        self._tf, self._mp = tf, mp
        if not Path(bundle).exists():
            sys.exit(f"{bundle} 없음. 먼저 08_train_hybrid.py 를 실행하세요.")
        b = joblib.load(bundle)
        self.b = b
        self.classes = b["classes"]
        self.img_size = b["img_size"]
        self.crop_size, self.margin, self.up_bias = crop_size, margin, up_bias

        if b["uses_cnn"]:
            model = tf.keras.models.load_model(b["cnn_model"])
            gap = next(l for l in model.layers
                       if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
            self.emb = tf.keras.Model(model.inputs, gap.output)
        else:
            self.emb = None

        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=False, min_detection_confidence=0.3)
        else:
            from mediapipe.tasks.python import BaseOptions, vision
            if not Path(task_model).exists():
                sys.exit(f"{task_model} 없음. mediapipe 를 0.10.x 로 내리거나 모델을 받으세요.")
            self.mesh = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=task_model),
                    running_mode=vision.RunningMode.IMAGE, num_faces=1))

        print(f"구성 {b['name']} | 테스트 Top-1 {b['test_top1']*100:.1f}% "
              f"Top-2 {b['test_top2']*100:.1f}%")

    # ---- 랜드마크 --------------------------------------------------------
    def _points(self, img_bgr):
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.legacy:
            r = self.mesh.process(rgb)
            lms = r.multi_face_landmarks[0].landmark if r.multi_face_landmarks else None
        else:
            mp = self._mp
            r = self.mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            lms = r.face_landmarks[0] if r.face_landmarks else None
        if lms is None:
            return None
        return np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64)

    def _crop_square(self, img, pts):
        h, w = img.shape[:2]
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(x1 - x0, y1 - y0) * (1.0 + self.margin)
        cy -= (y1 - y0) * self.up_bias
        half = side / 2.0
        left, top = int(round(cx - half)), int(round(cy - half))
        right, bottom = int(round(cx + half)), int(round(cy + half))
        pl, pt_ = max(0, -left), max(0, -top)
        pr, pb = max(0, right - w), max(0, bottom - h)
        if pl or pt_ or pr or pb:
            img = cv2.copyMakeBorder(img, pt_, pb, pl, pr, cv2.BORDER_REPLICATE)
            left, right = left + pl, right + pl
            top, bottom = top + pt_, bottom + pt_
        crop = img[top:bottom, left:right]
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            return None
        return cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _geo(pts):
        top, chin = pts[10], pts[152]
        face_length = float(np.linalg.norm(top - chin))
        widths = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in CONTOUR_PAIRS]
        max_w = max(widths)
        if max_w <= 0:
            return None
        f = [w / max_w for w in widths]
        f.append(face_length / max_w)
        v1, v2 = pts[148] - chin, pts[377] - chin
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        f.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
        return np.array(f, dtype=np.float32)

    # ---- 예측 ------------------------------------------------------------
    def predict_bgr(self, img_bgr):
        """BGR 이미지 하나를 받아 예측. 얼굴을 못 찾으면 None."""
        tf, b = self._tf, self.b
        pts = self._points(img_bgr)
        if pts is None:
            return None
        crop = self._crop_square(img_bgr, pts)
        if crop is None:
            return None

        parts = []
        if b["uses_cnn"]:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            x = tf.image.resize(rgb[None], [self.img_size, self.img_size], method="area")
            # 학습 때와 동일하게 좌우반전 평균
            e = (self.emb.predict(x, verbose=0)
                 + self.emb.predict(tf.image.flip_left_right(x), verbose=0)) / 2
            parts.append(b["scaler_cnn"].transform(e))

        if b["uses_geo"]:
            # 기하 특징은 크롭된 정사각 이미지에서 다시 잰다 (학습 때와 동일)
            cpts = self._points(crop)
            g = self._geo(cpts) if cpts is not None else None
            ok = g is not None
            if not ok:
                g = b["geo_fill"]
            g = np.hstack([g, [1.0 if ok else 0.0]])[None].astype(np.float32)
            parts.append(b["scaler_geo"].transform(g) * b["geo_weight"])

        X = np.hstack(parts)
        p = b["clf"].predict_proba(X)[0]

        # 학습 때 val 에서 고른 사전확률 보정을 그대로 적용
        p = p / np.power(np.maximum(b["prior_q"], 1e-9), b["prior_alpha"])
        p = p / p.sum()

        order = np.argsort(-p)
        return {
            "shape": self.classes[order[0]],
            "shape_ko": FACE_SHAPE_KOREAN.get(self.classes[order[0]], self.classes[order[0]]),
            "confidence": float(p[order[0]]),
            "top2": [(self.classes[i], float(p[i])) for i in order[:2]],
            "probs": {c: float(v) for c, v in zip(self.classes, p)},
        }

    def predict_file(self, path):
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return self.predict_bgr(img)


def show(path, r):
    if r is None:
        print(f"{Path(path).name:<30} 얼굴 검출 실패")
        return
    a, b = r["top2"]
    line = " / ".join(f"{FACE_SHAPE_KOREAN.get(c, c)} {v*100:.0f}%" for c, v in (a, b))
    print(f"{Path(path).name:<30} {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--dir")
    ap.add_argument("--bundle", default="hybrid_model.joblib")
    args = ap.parse_args()
    if not (args.image or args.dir):
        sys.exit("--image 또는 --dir 중 하나가 필요합니다")

    pr = FaceShapePredictor(args.bundle)
    if args.image:
        r = pr.predict_file(args.image)
        show(args.image, r)
        if r:
            print("\n전체 확률")
            for c, v in sorted(r["probs"].items(), key=lambda kv: -kv[1]):
                print(f"  {FACE_SHAPE_KOREAN.get(c, c):<8} {v*100:5.1f}%")
    else:
        files = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        for p in files:
            show(p, pr.predict_file(p))


if __name__ == "__main__":
    main()
