"""학습된 분할 모델을 실제 사광 사진에 적용해 입자/그림자 마스크와 확인 그림을 저장한다."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def imread_any(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"이미지를 읽지 못했습니다: {path}")
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def imwrite(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/particle_shadow_seg/weights/best.pt")
    ap.add_argument("--image", default="data/real/seq0064.tif", help="실제 사광 사진")
    ap.add_argument("--out_dir", default="results/seg_real")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = a.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(a.weights)
    img = imread_any(a.image)
    H, W = img.shape[:2]
    res = model.predict(img, imgsz=max(H, W), conf=a.conf, max_det=3000,
                        retina_masks=True, device=device, verbose=False)[0]

    particle = np.zeros((H, W), np.uint8)
    shadow = np.zeros((H, W), np.uint8)
    if res.masks is not None:
        for m, c in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
            m = cv2.resize(m, (W, H)) > 0.5
            (particle if c == 0 else shadow)[m] = 255

    vis = img.copy()
    vis[particle > 0] = (0.5 * vis[particle > 0] + [0, 127, 0]).astype(np.uint8)   # 입자 = 초록
    vis[shadow > 0] = (0.5 * vis[shadow > 0] + [0, 0, 127]).astype(np.uint8)       # 그림자 = 빨강

    out_dir, stem = Path(a.out_dir), Path(a.image).stem
    imwrite(out_dir / f"{stem}_particle_mask.png", particle)
    imwrite(out_dir / f"{stem}_shadow_mask.png", shadow)
    imwrite(out_dir / f"{stem}_overlay.jpg", vis)
    y, x = H // 3, W // 3
    imwrite(out_dir / f"{stem}_compare.jpg", np.hstack([img[y:y + 600, x:x + 600], vis[y:y + 600, x:x + 600]]))

    n_p = len(res.boxes) if res.boxes is not None else 0
    n_cls = np.bincount(res.boxes.cls.cpu().numpy().astype(int), minlength=2) if n_p else [0, 0]
    print(f"{Path(a.image).name}: 입자 {n_cls[0]}개, 그림자 {n_cls[1]}개")
    print(f"저장: {out_dir.resolve()}  (마스크 2장, 전체 겹침, 확대 비교)")


if __name__ == "__main__":
    main()
