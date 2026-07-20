"""Generate a dataset example figure for the Guizhou embroidery paper."""
import json
import os
import re

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

DATA_ROOT = 'data/guizhou_embroidery_correct'
LABELS_PATH = os.path.join(DATA_ROOT, 'labels.json')
IMG_ROOT = os.path.join(DATA_ROOT, 'images', 'train')
MASK_ROOT = os.path.join(DATA_ROOT, 'masks', 'train')

PATTERN_NAMES = ['辫绣', '堆绣', '马尾绣', '其他', '数纱马尾绣']


def _find_image_path(img_dir, name):
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = os.path.join(img_dir, f"{name}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_image(name):
    path = _find_image_path(IMG_ROOT, name)
    if path is None:
        raise FileNotFoundError(f"No image found for {name}")
    return np.array(Image.open(path).convert('RGB'))


def load_mask(name):
    path = _find_image_path(MASK_ROOT, name)
    if path is None:
        return None
    return np.array(Image.open(path).convert('L'))


def find_sample(split, used_bases=None, **conds):
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    for name, meta in labels[split].items():
        if all(meta.get(k) == v for k, v in conds.items()):
            base = re.sub(r'_a\d+$', '', name)
            if used_bases is not None and base in used_bases:
                continue
            return name, base
    return None, None


def main():
    # 1) Five pattern categories: real, no defect
    pattern_names = []
    pattern_bases = set()
    for cls in range(5):
        name, base = find_sample('train', used_bases=pattern_bases,
                                 pattern=cls, auth=1, defect=0)
        if name is None:
            name, base = find_sample('train', used_bases=pattern_bases, pattern=cls)
        pattern_names.append(name)
        pattern_bases.add(base)

    # 2) Authenticity comparison: same pattern class, real vs fake
    real_name, _ = find_sample('train', pattern=0, auth=1, defect=0)
    fake_name, _ = find_sample('train', pattern=0, auth=0, defect=0)

    # 3) Defect comparison: same pattern class, real, no defect vs with defect
    no_defect_name, _ = find_sample('train', pattern=0, auth=1, defect=0)
    yes_defect_name, _ = find_sample('train', pattern=0, auth=1, defect=1)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 5, figure=fig, wspace=0.25, hspace=0.35)

    # Row 1: pattern categories
    for i, name in enumerate(pattern_names):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(load_image(name))
        ax.set_title(PATTERN_NAMES[i], fontsize=14, fontweight='bold')
        ax.axis('off')

    # Row 2: real vs fake
    auth_pairs = [(real_name, '真品（手工绣）'), (fake_name, '伪作（机绣）')]
    for i, (name, title) in enumerate(auth_pairs):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(load_image(name))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    for i in range(2, 5):
        ax = fig.add_subplot(gs[1, i])
        ax.axis('off')

    # Row 3: no defect vs with defect
    defect_pairs = [
        (no_defect_name, '无疵点', False),
        (yes_defect_name, '有疵点（红色掩膜）', True),
    ]
    for i, (name, title, overlay) in enumerate(defect_pairs):
        ax = fig.add_subplot(gs[2, i])
        img = load_image(name)
        ax.imshow(img)
        if overlay:
            mask = load_mask(name)
            if mask is not None:
                h, w = img.shape[:2]
                mask_r = Image.fromarray(mask).resize((w, h), Image.NEAREST)
                mask_r = np.array(mask_r)
                # red overlay
                overlay_rgba = np.zeros((*img.shape[:2], 4))
                overlay_rgba[mask_r > 127] = [1.0, 0.0, 0.0, 0.35]
                ax.imshow(overlay_rgba)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    for i in range(2, 5):
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')

    # Row labels on the left
    fig.text(0.02, 0.82, '纹样类别', fontsize=15, fontweight='bold', va='center', rotation='vertical')
    fig.text(0.02, 0.52, '真伪对比', fontsize=15, fontweight='bold', va='center', rotation='vertical')
    fig.text(0.02, 0.22, '疵点对比', fontsize=15, fontweight='bold', va='center', rotation='vertical')

    fig.suptitle('贵州苗绣多任务数据集示例', fontsize=18, fontweight='bold', y=0.98)
    out_path = 'dataset_examples.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved example figure to {out_path}')


if __name__ == '__main__':
    main()
