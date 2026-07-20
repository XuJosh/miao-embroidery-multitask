"""
Create an MVTec-style folder structure for Guizhou embroidery defect detection
so that anomaly-detection libraries (e.g. anomalib) can be used directly.
Only the defect-detection task is represented here (normal vs. defective).
"""
import os
import json
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/guizhou_embroidery_correct')
    parser.add_argument('--dst', type=str, default='data/guizhou_embroidery_dinomaly')
    args = parser.parse_args()

    src_img = Path(args.src) / 'images'
    src_mask = Path(args.src) / 'masks'
    dst = Path(args.dst)

    with open(Path(args.src) / 'labels.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)

    splits = ['train', 'test']
    counts = {'train': {'good': 0}, 'test': {'good': 0, 'defect': 0}}

    for split in splits:
        for name, meta in labels[split].items():
            defect = int(meta['defect'])  # 0 = normal, 1 = defective
            if split == 'train' and defect == 1:
                # anomaly detectors are trained only on normal samples
                continue
            cls = 'good' if defect == 0 else 'defect'
            counts[split][cls] += 1

            img_src = src_img / split / f"{name}.jpg"
            if not img_src.exists():
                for ext in ('.jpeg', '.png', '.bmp'):
                    img_src = src_img / split / f"{name}{ext}"
                    if img_src.exists():
                        break
            if not img_src.exists():
                raise FileNotFoundError(f"Image not found for {name}")

            img_dst_dir = dst / split / cls
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_src, img_dst_dir / img_src.name)

            if split == 'test' and defect == 1:
                mask_src = src_mask / split / f"{name}.png"
                mask_dst_dir = dst / 'ground_truth' / 'defect'
                mask_dst_dir.mkdir(parents=True, exist_ok=True)
                if mask_src.exists():
                    shutil.copy2(mask_src, mask_dst_dir / f"{name}_mask.png")

    print('Prepared Dinomaly data at', dst)
    print(counts)


if __name__ == '__main__':
    main()
