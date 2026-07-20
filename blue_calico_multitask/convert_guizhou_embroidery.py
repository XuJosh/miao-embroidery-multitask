"""
Convert the downloaded Guizhou Embroidery dataset (D:\song\newdata_en)
into the format required by blue_calico_multitask.

Preprocessing pipeline (user-requested order):
    1. Filter out images whose width or height is smaller than --min_size.
    2. Augment each kept original image --aug_factor times
       (horizontal/vertical flips, 90-degree rotations, color jitter).
    3. Split the expanded pool into train/val/test.
    4. Synthesize authentication labels (real / fake) and defect labels.
    5. Save images, masks, and labels.json.

Normalization is performed online in dataset.py using ImageNet statistics.

Note: augmenting before splitting means augmented variants of the same source
image may appear in different splits. This can lead to data leakage and
inflated validation/test metrics. Use with caution for research reporting.
"""

import os
import json
import argparse
import random
import math
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split

PATTERN_NAMES = ['byx', 'dx', 'mx', 'qt', 'szmwx']
PATTERN_CHINESE = {
    'byx': '辫绣',
    'dx': '堆绣',
    'mx': '马尾绣',
    'qt': '其他',
    'szmwx': '数纱马尾绣',
}


def _random_seed_from_name(name):
    """Deterministic seed derived from file name for reproducible augmentations."""
    return sum(ord(c) for c in name) % (2 ** 31)


# ---------------------------------------------------------------------------
# Fake / defect synthesis
# ---------------------------------------------------------------------------

def make_machine_embroidery(img, rng):
    """Simulate machine-embroidered fake: too regular, slightly blurred, less saturated."""
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(rng.uniform(0.6, 0.9))
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    grid_spacing = rng.randint(4, 8)
    strength = rng.randint(3, 7)
    for y in range(0, h, grid_spacing):
        arr[y:y+1, :, :] = np.clip(arr[y:y+1, :, :] + strength, 0, 255)
    for x in range(0, w, grid_spacing):
        arr[:, x:x+1, :] = np.clip(arr[:, x:x+1, :] + strength, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def make_low_quality_copy(img, rng):
    """Simulate poor-quality digital reproduction: color shift, noise, compression artifacts."""
    r, g, b = img.split()
    r = r.point(lambda i: int(np.clip(i + rng.randint(-25, 35), 0, 255)))
    g = g.point(lambda i: int(np.clip(i + rng.randint(-20, 20), 0, 255)))
    b = b.point(lambda i: int(np.clip(i + rng.randint(-35, 25), 0, 255)))
    img = Image.merge('RGB', (r, g, b))
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, rng.uniform(8, 18), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    quality = rng.randint(40, 70)
    buf = io_bytes(img, quality)
    img = Image.open(buf)
    return img


def make_digital_tamper(img, rng):
    """Simulate digital forgery: copy-paste a patch to another location."""
    w, h = img.size
    arr = np.array(img)
    rw, rh = int(w * rng.uniform(0.15, 0.3)), int(h * rng.uniform(0.15, 0.3))
    x1 = rng.randint(0, w - rw)
    y1 = rng.randint(0, h - rh)
    x2 = rng.randint(0, w - rw)
    y2 = rng.randint(0, h - rh)
    patch = arr[y1:y1+rh, x1:x1+rw].copy()
    patch = np.clip(patch * rng.uniform(0.9, 1.1) + rng.randint(-15, 15), 0, 255).astype(np.uint8)
    arr[y2:y2+rh, x2:x2+rw] = patch
    return Image.fromarray(arr)


def apply_fake(img, name):
    """Apply one of the embroidery-specific counterfeit strategies."""
    rng = np.random.RandomState(_random_seed_from_name(name))
    strategy = rng.choice(['machine', 'low_quality', 'tamper'])
    if strategy == 'machine':
        return make_machine_embroidery(img, rng)
    elif strategy == 'low_quality':
        return make_low_quality_copy(img, rng)
    else:
        return make_digital_tamper(img, rng)


def io_bytes(img, quality=60):
    """Save image to a BytesIO buffer with JPEG quality to simulate compression."""
    import io
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return buf


def add_embroidery_defect(img, mask, name):
    """Synthesize embroidery-specific defects and update mask."""
    rng = np.random.RandomState(_random_seed_from_name(name) + 1)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    w, h = img.size
    n_defects = rng.randint(1, 4)

    for _ in range(n_defects):
        kind = rng.choice(['missing_stitch', 'thread_break', 'stain', 'color_bleed', 'loose_thread'])

        if kind == 'missing_stitch':
            x1, y1 = rng.randint(20, w - 20), rng.randint(20, h - 20)
            angle = rng.uniform(0, math.pi)
            length = rng.randint(20, 60)
            x2 = int(x1 + length * math.cos(angle))
            y2 = int(y1 + length * math.sin(angle))
            draw_img.line([(x1, y1), (x2, y2)], fill=(40, 40, 40), width=rng.randint(1, 3))
            draw_mask.line([(x1, y1), (x2, y2)], fill=255, width=rng.randint(2, 4))

        elif kind == 'thread_break':
            x, y = rng.randint(30, w - 30), rng.randint(30, h - 30)
            length = rng.randint(10, 30)
            draw_img.line([(x, y), (x + length, y)], fill=(200, 200, 200), width=2)
            draw_mask.line([(x, y), (x + length, y)], fill=255, width=4)

        elif kind == 'stain':
            cx, cy = rng.randint(40, w - 40), rng.randint(40, h - 40)
            r = rng.randint(10, 25)
            color = (rng.randint(30, 80), rng.randint(30, 80), rng.randint(30, 80))
            draw_img.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
            draw_mask.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)

        elif kind == 'color_bleed':
            cx, cy = rng.randint(40, w - 40), rng.randint(40, h - 40)
            r = rng.randint(15, 35)
            arr = np.array(img)
            y1, y2 = max(0, cy-r), min(h, cy+r)
            x1, x2 = max(0, cx-r), min(w, cx+r)
            patch = arr[y1:y2, x1:x2].astype(np.float32)
            shift = rng.choice(['red', 'blue', 'green'])
            if shift == 'red':
                patch[:, :, 0] = np.clip(patch[:, :, 0] + rng.randint(30, 60), 0, 255)
            elif shift == 'blue':
                patch[:, :, 2] = np.clip(patch[:, :, 2] + rng.randint(30, 60), 0, 255)
            else:
                patch[:, :, 1] = np.clip(patch[:, :, 1] + rng.randint(30, 60), 0, 255)
            arr[y1:y2, x1:x2] = patch.astype(np.uint8)
            img = Image.fromarray(arr)
            draw_mask = ImageDraw.Draw(mask)
            draw_mask.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)
            draw_img = ImageDraw.Draw(img)

        else:  # loose_thread
            x, y = rng.randint(30, w - 30), rng.randint(30, h - 30)
            points = [(x, y)]
            for _ in range(rng.randint(2, 4)):
                x += rng.randint(-30, 30)
                y += rng.randint(-30, 30)
                points.append((x, y))
            color = (rng.randint(150, 220), rng.randint(150, 220), rng.randint(150, 220))
            draw_img.line(points, fill=color, width=rng.randint(1, 2))
            draw_mask.line(points, fill=255, width=rng.randint(2, 3))

    return img, mask


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _aug_seed(name, aug_idx):
    return (_random_seed_from_name(name) + aug_idx * 1009) % (2 ** 31)


def augment_image(img, name, aug_idx):
    """Apply random spatial + color augmentation to a single image."""
    rng = np.random.RandomState(_aug_seed(name, aug_idx))

    # Spatial transforms
    if rng.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.rand() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    rotation = rng.choice([0, 90, 180, 270])
    if rotation == 90:
        img = img.transpose(Image.ROTATE_90)
    elif rotation == 180:
        img = img.transpose(Image.ROTATE_180)
    elif rotation == 270:
        img = img.transpose(Image.ROTATE_270)

    # Color transforms
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    if rng.rand() < 0.5:
        img = _shift_hue(img, rng.uniform(-10, 10))

    return img


def _shift_hue(img, deg):
    """Small hue shift by rotating RGB channels (approximation)."""
    arr = np.array(img).astype(np.float32)
    shift = int(deg)
    if shift > 0:
        arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = arr[:, :, 2], arr[:, :, 0], arr[:, :, 1]
    elif shift < 0:
        arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = arr[:, :, 1], arr[:, :, 2], arr[:, :, 0]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------

def _save_sample(img, mask, images_dir, masks_dir, name):
    img.save(os.path.join(images_dir, f"{name}.jpg"), quality=95)
    mask.save(os.path.join(masks_dir, f"{name}.png"))


def convert_dataset(src_root, dst_root, input_size=224, seed=42,
                    min_size=128, aug_factor=10):
    random.seed(seed)
    np.random.seed(seed)

    images_dir = os.path.join(dst_root, 'images')
    masks_dir = os.path.join(dst_root, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # 1. Collect original images, filtering low-resolution ones
    original_samples = []  # list of (src_path, pattern_idx)
    excluded = []
    for pattern_idx, folder in enumerate(PATTERN_NAMES):
        folder_path = os.path.join(src_root, folder)
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        files = sorted([f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        kept = 0
        for f in files:
            fp = os.path.join(folder_path, f)
            try:
                with Image.open(fp) as im:
                    w, h = im.size
            except Exception as e:
                excluded.append((fp, f'open_failed: {e}'))
                continue
            if min(w, h) < min_size:
                excluded.append((fp, f'resolution_{w}x{h}'))
                continue
            original_samples.append((fp, pattern_idx))
            kept += 1
        print(f"{folder} ({PATTERN_CHINESE[folder]}): {kept} kept, "
              f"{len(files) - kept} excluded")

    if excluded:
        excluded_path = os.path.join(dst_root, 'excluded.json')
        with open(excluded_path, 'w', encoding='utf-8') as f:
            json.dump([{'file': fp, 'reason': reason} for fp, reason in excluded], f, indent=2, ensure_ascii=False)
        print(f"\nExcluded {len(excluded)} low-resolution images. List saved to {excluded_path}")

    # 2. Augment the filtered original images BEFORE splitting
    tmp_dir = os.path.join(dst_root, '.tmp_aug')
    tmp_img_dir = os.path.join(tmp_dir, 'images')
    os.makedirs(tmp_img_dir, exist_ok=True)

    print(f"\nAugmenting {len(original_samples)} original images {aug_factor}x ...")
    expanded_samples = []  # list of (tmp_img_path, pattern_idx, source_name)
    for orig_idx, (src_path, pattern_idx) in enumerate(original_samples):
        base_name = f"orig_{orig_idx:05d}"
        img = Image.open(src_path).convert('RGB')
        img = img.resize((input_size, input_size), Image.BILINEAR)

        for aug_idx in range(aug_factor):
            if aug_idx == 0:
                aug_img = img
            else:
                aug_img = augment_image(img, base_name, aug_idx)
            aug_name = f"{base_name}_a{aug_idx}"
            tmp_path = os.path.join(tmp_img_dir, f"{aug_name}.jpg")
            aug_img.save(tmp_path, quality=95)
            expanded_samples.append((tmp_path, pattern_idx, aug_name))

        if (orig_idx + 1) % 100 == 0 or orig_idx == len(original_samples) - 1:
            print(f"  {orig_idx + 1}/{len(original_samples)} originals augmented")

    # 3. Split the expanded pool into train/val/test (stratified by pattern)
    paths = [s[0] for s in expanded_samples]
    pattern_labels = [s[1] for s in expanded_samples]
    names = [s[2] for s in expanded_samples]

    train_paths, temp_paths, train_lbl, temp_lbl, train_names, temp_names = train_test_split(
        paths, pattern_labels, names, test_size=0.3, random_state=seed, shuffle=True, stratify=pattern_labels)
    val_paths, test_paths, val_lbl, test_lbl, val_names, test_names = train_test_split(
        temp_paths, temp_lbl, temp_names, test_size=0.5, random_state=seed, shuffle=True, stratify=temp_lbl)

    split_samples = {
        'train': list(zip(train_paths, train_lbl, train_names)),
        'val': list(zip(val_paths, val_lbl, val_names)),
        'test': list(zip(test_paths, test_lbl, test_names)),
    }

    labels = {split: {} for split in split_samples}

    print("\nSplit sizes:")
    for split, samples in split_samples.items():
        print(f"  {split}: {len(samples)}")

    # 4. Synthesize auth/defect labels and save final images/masks
    for split, samples in split_samples.items():
        split_img_dir = os.path.join(images_dir, split)
        split_mask_dir = os.path.join(masks_dir, split)
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        print(f"\nProcessing {split}: {len(samples)} samples ...")
        for idx, (tmp_path, pattern_idx, base_name) in enumerate(samples):
            name = f"{split}_{idx:05d}"
            img = Image.open(tmp_path).convert('RGB')

            # Decide auth and defect labels
            is_real = random.random() < 0.6
            has_defect = is_real and (random.random() < 0.4)
            mask = Image.new('L', (input_size, input_size), 0)

            if not is_real:
                img = apply_fake(img, name)
            if has_defect:
                img, mask = add_embroidery_defect(img, mask, name)

            _save_sample(img, mask, split_img_dir, split_mask_dir, name)
            labels[split][name] = {
                'auth': 1 if is_real else 0,
                'pattern': pattern_idx,
                'defect': 1 if has_defect else 0,
            }

            if (idx + 1) % 500 == 0 or idx == len(samples) - 1:
                print(f"  {idx + 1}/{len(samples)} done")

    # Clean up temporary augmented images
    shutil.rmtree(tmp_dir, ignore_errors=True)

    with open(os.path.join(dst_root, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to {dst_root}")
    for split in split_samples:
        print(f"  {split}: {len(labels[split])} images")
    for split in split_samples:
        real = sum(1 for m in labels[split].values() if m['auth'] == 1)
        fake = sum(1 for m in labels[split].values() if m['auth'] == 0)
        defect = sum(1 for m in labels[split].values() if m['defect'] == 1)
        print(f"  {split} labels: real={real}, fake={fake}, with_defect={defect}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=r'D:\song\newdata_en')
    parser.add_argument('--dst', type=str, default='data/guizhou_embroidery')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_size', type=int, default=128,
                        help='Minimum width/height in pixels; smaller images are excluded.')
    parser.add_argument('--aug_factor', type=int, default=10,
                        help='Number of augmented variants per original image before splitting.')
    args = parser.parse_args()

    # Clean old destination to avoid mixed data
    if os.path.exists(args.dst):
        print(f"Removing old destination directory: {args.dst}")
        shutil.rmtree(args.dst)

    convert_dataset(args.src, args.dst, input_size=args.size, seed=args.seed,
                    min_size=args.min_size, aug_factor=args.aug_factor)


if __name__ == '__main__':
    main()
