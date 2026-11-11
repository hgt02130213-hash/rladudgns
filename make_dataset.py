"""합성 이미지(입자 + 그림자)를 YOLO 분할 학습용 640x640 타일로 변환한다.

입력: <syn_dir>/synthetic_XXXX_image.png, synthetic_XXXX_shadow.png
출력: <out>/images/{train,val}, <out>/labels/{train,val}, <out>/seg.yaml
"""
import argparse
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def imread_gray(path):
    """한글 경로에서도 읽히도록 numpy로 읽어 디코드"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def imwrite(path, img, params=None):
    ok, buf = cv2.imencode(Path(path).suffix, img, params or [])
    if ok:
        buf.tofile(str(path))


def bg(g):
    med = float(np.median(g))
    mad = 1.4826 * float(np.median(np.abs(g - med)))
    return med, max(mad, 2.0)


def clean(m):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def particle_mask(img):
    med, s = bg(img)
    return clean(img > med + 4 * s + 10)                 # 배경보다 밝은 곳 = 입자


def shadow_mask(sh):
    med, s = bg(sh)
    if med > 127:                                        # 밝은 배경 + 어두운 그림자
        return clean(sh < med - 4 * s - 10)
    return clean(sh > med + 4 * s + 10)                  # 어두운 배경 + 밝은 그림자


def realistic(img, pm):
    """합성 이미지를 실제 사진처럼: 배경 얼룩, 입자 밝기 변화, 흐림, 잡음, 대비, 감마"""
    out = img.astype(np.float32).copy()
    H, W = out.shape
    small = cv2.GaussianBlur(np.random.randn(H // 8 + 1, W // 8 + 1).astype(np.float32),
                             (0, 0), random.uniform(2, 8))
    tex = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    out += tex / (tex.std() + 1e-6) * random.uniform(3, 15)
    if random.random() < 0.4:
        k = random.uniform(0.3, 0.9)
        out[pm > 0] = out[pm > 0] * k + np.median(out) * (1 - k) * random.uniform(0.3, 1.0)
    ks = random.choice([1, 3, 5])
    if ks > 1:
        out = cv2.GaussianBlur(out, (ks, ks), 0)
    out += np.random.randn(H, W).astype(np.float32) * random.uniform(1, 8)
    a, b = random.uniform(0.7, 1.3), random.uniform(-25, 25)
    out = (out - out.mean()) * a + out.mean() + b
    out = 255 * (np.clip(out, 0, 255) / 255) ** random.uniform(0.8, 1.25)
    return np.clip(out, 0, 255).astype(np.uint8)


def polygons(mask, cls, W, H, min_area):
    """마스크의 각 덩어리를 YOLO 분할 라벨(정규화 다각형)로"""
    lines = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            lines.append(f"{cls} " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in c))
    return lines


def starts(L, tile):
    if L <= tile:
        return [0]
    s = list(range(0, L - tile, tile))
    s.append(L - tile)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn_dir", default="data/synthetic", help="합성 이미지 폴더")
    ap.add_argument("--out", default="data/seg_data", help="학습용 타일을 만들 폴더")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--min_area", type=int, default=15)
    ap.add_argument("--aug", type=int, default=2, help="학습 이미지당 '실제처럼' 변형본 수")
    ap.add_argument("--n_val", type=int, default=6, help="검증용 이미지 수 (마지막 n장)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    syn_dir, out = Path(a.syn_dir), Path(a.out)
    png_fast = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    if out.exists():
        shutil.rmtree(out)
    for sp in ["train", "val"]:
        (out / "images" / sp).mkdir(parents=True)
        (out / "labels" / sp).mkdir(parents=True)

    image_files = sorted(syn_dir.glob("synthetic_*_image.png"))
    if not image_files:
        raise SystemExit(f"합성 이미지를 찾지 못했습니다: {syn_dir.resolve()}")
    val_set = set(image_files[-a.n_val:])
    n_tiles = {"train": 0, "val": 0}

    for idx, ip in enumerate(image_files, 1):
        t1 = time.time()
        sp_path = ip.with_name(ip.name.replace("_image.png", "_shadow.png"))
        img = imread_gray(ip)
        pm = particle_mask(img.astype(np.float32))
        sm = shadow_mask(imread_gray(sp_path).astype(np.float32))
        sm[pm > 0] = 0                                    # 입자와 겹친 부분은 그림자에서 제외
        split = "val" if ip in val_set else "train"
        n_aug = a.aug if split == "train" else 1
        variants = [img] + [realistic(img, pm) for _ in range(n_aug)]
        H, W = img.shape

        for vi, var in enumerate(variants):
            for ty in starts(H, a.tile):
                for tx in starts(W, a.tile):
                    t_img = var[ty:ty + a.tile, tx:tx + a.tile]
                    t_pm = pm[ty:ty + a.tile, tx:tx + a.tile]
                    t_sm = sm[ty:ty + a.tile, tx:tx + a.tile]
                    h, w = t_img.shape
                    lines = polygons(t_pm, 0, w, h, a.min_area) + polygons(t_sm, 1, w, h, a.min_area)
                    name = f"{ip.stem}_v{vi}_y{ty}_x{tx}"
                    imwrite(out / "images" / split / f"{name}.png",
                            cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR), png_fast)
                    (out / "labels" / split / f"{name}.txt").write_text(
                        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    n_tiles[split] += 1
        print(f"[{idx:2d}/{len(image_files)}] {ip.name} ({split}) 완료, {time.time() - t1:.1f}초")

    yaml_path = out / "seg.yaml"
    yaml_path.write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "names:\n  0: particle\n  1: shadow\n", encoding="utf-8")

    examples = sorted((out / "images" / "train").glob("*_v1_*.png"))
    if examples:
        shutil.copy2(examples[min(3, len(examples) - 1)], out / "example_realistic.png")
    print(f"\n조각 수: {n_tiles}")
    print(f"설정 파일: {yaml_path}")
    print(f"'실제처럼' 변형 예시: {out / 'example_realistic.png'}")


if __name__ == "__main__":
    main()
