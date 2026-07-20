"""
Explainability visualization for trained embroidery models.

Produces:
    - DINOv2 CLS-to-patch cosine attention map
    - Frequency magnitude map (enhanced model only)
    - Predicted labels and confidence
"""

import os
import argparse
import json
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import BlueCalicoDataset, get_transforms
from model_dinov2_mamba import EmbroideryNet
from model_enhanced import EmbroideryNetEnhanced


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(img):
    """img: torch tensor (3, H, W)"""
    img = img.cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def get_cls_attention_map(cls_token, patch_tokens, input_size=224):
    """
    cls_token:    (D,)
    patch_tokens: (N, D)
    Returns attention map resized to (input_size, input_size).
    """
    cls_token = F.normalize(cls_token, dim=-1)
    patch_tokens = F.normalize(patch_tokens, dim=-1)
    sim = torch.matmul(cls_token, patch_tokens.t())  # (N,)
    patch_size = int(math.sqrt(sim.numel()))
    sim = sim.view(patch_size, patch_size).cpu().numpy()
    # Normalize to [0, 1]
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    # Resize to image size
    sim = torch.from_numpy(sim).unsqueeze(0).unsqueeze(0).float()
    sim = F.interpolate(sim, size=(input_size, input_size), mode='bilinear', align_corners=False)
    sim = sim.squeeze().numpy()
    return sim


def get_frequency_map(img, target_size=224):
    """img: torch tensor (3, H, W)"""
    gray = img.mean(dim=0)  # (H, W)
    fft = torch.fft.rfft2(gray, norm='ortho')
    mag = torch.log1p(torch.abs(fft)).cpu().numpy()
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    # Resize to target_size x target_size for display
    mag = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float()
    mag = F.interpolate(mag, size=(target_size, target_size), mode='bilinear', align_corners=False)
    mag = mag.squeeze().numpy()
    return mag


def visualize_one(model, image_tensor, name, model_type, device, save_dir, idx):
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x, return_features=True)

    # Retrieve DINOv2 features by a forward hook-free approach: re-run only DINOv2
    # We use the model's dinov2 module directly.
    with torch.no_grad():
        dinov2_cls, patch_tokens = model.dinov2(x)

    img_np = denormalize(image_tensor)
    attn_map = get_cls_attention_map(dinov2_cls[0], patch_tokens[0], input_size=img_np.shape[0])

    has_freq = model_type == 'enhanced' and getattr(model, 'use_frequency', False)
    if has_freq:
        freq_map = get_frequency_map(image_tensor, target_size=img_np.shape[0])

    # Predictions
    auth_pred = out['auth'].argmax(1).item()
    auth_conf = F.softmax(out['auth'], dim=1)[0, auth_pred].item()
    pat_pred = out['pattern'].argmax(1).item()
    pat_conf = F.softmax(out['pattern'], dim=1)[0, pat_pred].item()

    fig, axes = plt.subplots(1, 3 if has_freq else 2, figsize=(12 if has_freq else 8, 4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(img_np)
    axes[0].set_title(f'Image: {name}\nAuth={auth_pred} ({auth_conf:.2f}), Pat={pat_pred} ({pat_conf:.2f})')
    axes[0].axis('off')

    axes[1].imshow(img_np, alpha=0.5)
    axes[1].imshow(attn_map, cmap='jet', alpha=0.5)
    axes[1].set_title('DINOv2 CLS attention')
    axes[1].axis('off')

    if has_freq:
        axes[2].imshow(freq_map, cmap='viridis')
        axes[2].set_title('Frequency magnitude (FFT)')
        axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{idx:03d}_{name}_explain.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='enhanced', choices=['base', 'enhanced'])
    parser.add_argument('--data_dir', type=str, default='data/guizhou_embroidery')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_images', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='visualizations')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else (args.device if args.device != 'auto' else 'cpu'))

    os.makedirs(args.save_dir, exist_ok=True)

    # Build model
    if args.model_type == 'base':
        model = EmbroideryNet(
            num_patterns=5, defect_mode='cls', seq_encoder='transformer',
            seq_layers=2, seq_nhead=8, seq_dim_feedforward=512,
            dropout=0.3, dinov2_checkpoint=False,
        ).to(device)
    else:
        model = EmbroideryNetEnhanced(
            num_patterns=5, defect_mode='cls', seq_encoder='transformer',
            seq_layers=2, seq_nhead=8, seq_dim_feedforward=512,
            dropout=0.3, dinov2_checkpoint=False,
            use_frequency=True, freq_dim=128,
            use_arcface=True, arcface_s=30.0, arcface_m=0.30,
        ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    print(f'Loaded {args.model_type} model from {args.checkpoint}')

    ds = BlueCalicoDataset(args.data_dir, args.split, get_transforms(False, args.input_size))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    count = 0
    for batch in loader:
        if count >= args.num_images:
            break
        image = batch['image'][0]
        name = batch['name'][0] if 'name' in batch else f'img{count}'
        visualize_one(model, image, name, args.model_type, device, args.save_dir, count)
        count += 1

    print(f'Saved {count} visualizations to {args.save_dir}')


if __name__ == '__main__':
    main()
