"""
PyTorch Dataset for the synthetic Blue Calico multi-task dataset.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_training=True, input_size=224):
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _find_image_path(img_dir, name):
    """Look for image in .jpg, .jpeg, .png, .bmp order."""
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = os.path.join(img_dir, f"{name}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No image found for {name} in {img_dir}")


class BlueCalicoDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform if transform is not None else get_transforms(split == 'train')
        self.img_dir = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'masks', split)

        with open(os.path.join(root, 'labels.json'), 'r', encoding='utf-8') as f:
            labels = json.load(f)
        self.samples = [(name, meta) for name, meta in labels[split].items()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, meta = self.samples[idx]
        img_path = _find_image_path(self.img_dir, name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, f"{name}.png")).convert('L')

        if self.transform:
            img = self.transform(img)
        # transform mask to tensor without normalization
        mask = transforms.ToTensor()(mask)

        return {
            'image': img,
            'auth': torch.tensor(meta['auth'], dtype=torch.long),
            'pattern': torch.tensor(meta['pattern'], dtype=torch.long),
            'defect': mask,                  # (1, H, W) segmentation target
            'defect_label': torch.tensor(meta['defect'], dtype=torch.long),  # 0/1 image-level
            'name': name,
        }
