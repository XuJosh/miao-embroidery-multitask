"""
Generate a synthetic dataset for pipeline testing.

The real Shen Embroidery dataset is not publicly redistributable, so this script
creates random placeholder images in the same folder structure:

    data/synthetic/
        train/
            shenxiu/  *.jpg
            fei/      *.jpg
        val/
            shenxiu/  *.jpg
            fei/      *.jpg
        test/
            shenxiu/  *.jpg
            fei/      *.jpg

Replace these folders with your real dataset when ready.
"""

import os
import argparse
import numpy as np
from PIL import Image


def generate_image(size=(224, 224), mode='shenxiu', seed=None):
    """Create a random RGB image. Two modes produce slightly different textures."""
    if seed is not None:
        np.random.seed(seed)

    h, w = size
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    if mode == 'shenxiu':
        # add some structured diagonal stripes (placeholder for embroidery texture)
        for i in range(h):
            arr[i, ::8] = (arr[i, ::8] + np.array([80, 40, 40])).clip(0, 255).astype(np.uint8)
    else:
        # smoother blobs (placeholder for non-embroidery images)
        blur = np.random.randint(0, 255, (h // 4, w // 4, 3), dtype=np.uint8)
        arr = np.array(Image.fromarray(blur).resize((w, h), Image.BILINEAR))

    return Image.fromarray(arr)


def make_dataset(root, split_counts, size=(224, 224)):
    for split, counts in split_counts.items():
        for cls, n in counts.items():
            folder = os.path.join(root, split, cls)
            os.makedirs(folder, exist_ok=True)
            # clean old files
            for f in os.listdir(folder):
                if f.endswith('.jpg'):
                    os.remove(os.path.join(folder, f))
            for idx in range(n):
                img = generate_image(size, mode=cls, seed=idx + n * hash(cls) % 10000)
                img.save(os.path.join(folder, f"{cls}_{idx:04d}.jpg"), quality=90)
            print(f"Created {n} images in {folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/synthetic')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--train', type=int, default=200,
                        help='Images per class in train set')
    parser.add_argument('--val', type=int, default=20,
                        help='Images per class in val set')
    parser.add_argument('--test', type=int, default=20,
                        help='Images per class in test set')
    args = parser.parse_args()

    split_counts = {
        'train': {'shenxiu': args.train, 'fei': args.train},
        'val':   {'shenxiu': args.val,   'fei': args.val},
        'test':  {'shenxiu': args.test,  'fei': args.test},
    }
    make_dataset(args.root, split_counts, size=(args.size, args.size))
    print("\nSynthetic dataset ready. Replace with real images to reproduce paper results.")


if __name__ == '__main__':
    main()
