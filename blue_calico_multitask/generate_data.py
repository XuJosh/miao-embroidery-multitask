"""
Generate a synthetic Blue Calico (南通蓝印花布) dataset for multi-task learning.

Structure:
    data/blue_calico/
        images/{train,val,test}/xxx.png
        masks/{train,val,test}/xxx.png
        labels.json

Labels per image:
    auth: 0=fake, 1=real
    pattern: 0..N-1 pattern class
    defect: 0=no defect, 1=has defect
"""

import os
import json
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw

PATTERN_NAMES = ['fish', 'flower', 'geometry', 'cloud', 'phoenix']

BLUE = (25, 60, 120)
WHITE = (220, 220, 220)


def _rand_color(base, var=15):
    return tuple(max(0, min(255, c + random.randint(-var, var))) for c in base)


def draw_pattern(draw, size, pattern_type):
    """Draw a simple white pattern on the blue cloth."""
    w, h = size
    if pattern_type == 0:       # fish
        for _ in range(random.randint(2, 5)):
            cx, cy = random.randint(40, w - 40), random.randint(40, h - 40)
            draw.ellipse([cx-30, cy-15, cx+30, cy+15], fill=WHITE)
            draw.polygon([(cx+25, cy), (cx+55, cy-10), (cx+55, cy+10)], fill=WHITE)
    elif pattern_type == 1:     # flower
        for _ in range(random.randint(2, 4)):
            cx, cy = random.randint(50, w - 50), random.randint(50, h - 50)
            for angle in range(0, 360, 60):
                import math
                rad = math.radians(angle)
                dx, dy = int(cx + 25 * math.cos(rad)), int(cy + 25 * math.sin(rad))
                draw.ellipse([dx-12, dy-12, dx+12, dy+12], fill=WHITE)
            draw.ellipse([cx-10, cy-10, cx+10, cy+10], fill=WHITE)
    elif pattern_type == 2:     # geometry
        step = 40
        for x in range(step, w, step):
            for y in range(step, h, step):
                if random.random() > 0.4:
                    draw.regular_polygon((x, y, 14), n_sides=4, fill=WHITE)
    elif pattern_type == 3:     # cloud
        for _ in range(random.randint(3, 6)):
            cx, cy = random.randint(60, w - 60), random.randint(60, h - 60)
            draw.ellipse([cx-35, cy-18, cx-5, cy+18], fill=WHITE)
            draw.ellipse([cx-15, cy-25, cx+20, cy+15], fill=WHITE)
            draw.ellipse([cx+5, cy-18, cx+40, cy+18], fill=WHITE)
    else:                       # phoenix
        cx, cy = w // 2, h // 2
        draw.polygon([(cx, cy-70), (cx-25, cy+40), (cx+25, cy+40)], fill=WHITE)
        draw.ellipse([cx-50, cy-30, cx+50, cy+30], fill=WHITE)


def add_defects(img, mask):
    """Randomly add dye defects and update mask."""
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    w, h = img.size
    n = random.randint(1, 4)
    for _ in range(n):
        kind = random.choice(['blob', 'line', 'scratch'])
        if kind == 'blob':
            cx, cy = random.randint(30, w - 30), random.randint(30, h - 30)
            r = random.randint(15, 35)
            draw_img.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(160, 160, 160))
            draw_mask.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)
        elif kind == 'line':
            x1, y1 = random.randint(20, w - 20), random.randint(20, h - 20)
            x2, y2 = x1 + random.randint(-60, 60), y1 + random.randint(-60, 60)
            draw_img.line([(x1, y1), (x2, y2)], fill=(120, 120, 120), width=6)
            draw_mask.line([(x1, y1), (x2, y2)], fill=255, width=6)
        else:
            x, y = random.randint(30, w - 30), random.randint(30, h - 30)
            draw_img.line([(x, y), (x+random.randint(-50, 50), y+random.randint(-50, 50))],
                          fill=(90, 90, 90), width=4)
            draw_mask.line([(x, y), (x+random.randint(-50, 50), y+random.randint(-50, 50))],
                           fill=255, width=4)


def generate_image(size, pattern_type, real=True, with_defect=False):
    # base cloth with slight color variation
    arr = np.full((*size, 3), _rand_color(BLUE, var=10), dtype=np.uint8)
    # add subtle fabric texture
    noise = np.random.randint(-8, 8, (*size, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    draw = ImageDraw.Draw(img)
    draw_pattern(draw, size, pattern_type)

    mask = Image.new('L', size, 0)

    if not real:
        # fake signs: color shift, extra noise, wrong pattern rotation
        r, g, b = img.split()
        img = Image.merge('RGB', (
            r.point(lambda i: min(255, i + random.randint(10, 40))),
            g,
            b.point(lambda i: max(0, i - random.randint(10, 30))),
        ))
        # add heavy noise
        na = np.array(img)
        na = np.clip(na + np.random.randint(-30, 30, na.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(na)

    if with_defect:
        add_defects(img, mask)

    return img, mask


def make_dataset(root, split_counts, size=(224, 224), seed=42):
    random.seed(seed)
    np.random.seed(seed)

    images_dir = os.path.join(root, 'images')
    masks_dir = os.path.join(root, 'masks')
    labels = {}

    for split, n in split_counts.items():
        labels[split] = {}
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(masks_dir, split), exist_ok=True)

        for idx in range(n):
            pattern_type = idx % len(PATTERN_NAMES)
            real = random.random() > 0.3       # 70% real
            with_defect = real and (random.random() > 0.2)   # 80% real images have defects

            img, mask = generate_image(size, pattern_type, real=real, with_defect=with_defect)
            name = f"{split}_{idx:04d}"
            img.save(os.path.join(images_dir, split, f"{name}.png"))
            mask.save(os.path.join(masks_dir, split, f"{name}.png"))

            labels[split][name] = {
                'auth': 1 if real else 0,
                'pattern': pattern_type,
                'defect': 1 if with_defect else 0,
            }

    with open(os.path.join(root, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to {root}")
    for split, n in split_counts.items():
        print(f"  {split}: {n} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/blue_calico')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--train', type=int, default=400)
    parser.add_argument('--val', type=int, default=50)
    parser.add_argument('--test', type=int, default=50)
    args = parser.parse_args()

    make_dataset(args.root, {
        'train': args.train,
        'val': args.val,
        'test': args.test,
    }, size=(args.size, args.size))


if __name__ == '__main__':
    main()
