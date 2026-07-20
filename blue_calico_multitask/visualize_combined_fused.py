"""
Visualization / interpretability for combined_fused_correct model.
Generates:
    1. Grad-CAM heatmaps for auth / pattern / defect heads (ResNet50 layer4).
    2. DINOv2 last-block CLS-to-patch attention map.

Usage:
    python visualize_combined_fused.py \
        --checkpoint checkpoints_combined_fused_correct/checkpoint_last.pt \
        --data_dir data/guizhou_embroidery_correct \
        --save_dir vis_combined_fused \
        --num_samples 6
"""
import os
import argparse
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

from dataset import BlueCalicoDataset, get_transforms
from model_combined import EmbroideryNetCombined


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Denormalize a torch tensor (C,H,W) to numpy RGB (H,W,C) in [0,1]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def load_model(checkpoint_path, device):
    model = EmbroideryNetCombined(
        num_patterns=5,
        defect_mode='cls',
        lora_r=8, lora_alpha=16,
        seq_layers=2, seq_nhead=8, seq_dim_feedforward=512,
        dropout=0.3,
        use_frequency=False,
        use_arcface=True,
        auth_defect_use_fused=True,
        use_patch_transformer=True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


class GradCAM:
    """Grad-CAM for the ResNet50 layer4 feature map."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image, task, target_class):
        """
        Args:
            image: (1, 3, H, W) tensor on device, requires_grad not needed.
            task: one of 'auth', 'pattern', 'defect'.
            target_class: int class index.
        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.gradients = None
        self.activations = None

        image = image.requires_grad_(True)
        out = self.model(image)
        logits = out[task]  # (1, C)
        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError('Failed to capture activation/gradient.')

        # Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class DinoV2AttentionExtractor:
    """Extract CLS-to-patch attention from the last DINOv2 block."""

    def __init__(self, model):
        self.model = model
        # PEFT wrapped model path
        base = model.dinov2.dinov2.base_model.model
        self.attn_module = base.blocks[-1].attn
        self.attn_input = None
        self.hook = self.attn_module.register_forward_hook(self._save_input)

    def _save_input(self, module, input, output):
        # input is a tuple; DINOv2 attention receives (B, N, C)
        self.attn_input = input[0].detach()

    def __call__(self, image):
        self.attn_input = None
        with torch.no_grad():
            _ = self.model(image)
            if self.attn_input is None:
                raise RuntimeError('Failed to capture DINOv2 attention input.')

            x = self.attn_input
            B, N, C = x.shape
            # qkv with LoRA adaptation
            qkv = self.attn_module.qkv(x)  # (B, N, 3*C)
            qkv = qkv.reshape(B, N, 3, self.attn_module.num_heads, C // self.attn_module.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  # (B, heads, N, head_dim)
            scale = (C // self.attn_module.num_heads) ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, N, N)
            attn = F.softmax(attn, dim=-1)
            attn = attn.mean(dim=1)  # average over heads -> (B, N, N)
            cls_to_patch = attn[0, 0, 1:]  # CLS token (index 0) attends to patch tokens

            # reshape to spatial grid
            grid = int(np.sqrt(cls_to_patch.numel()))
            attn_map = cls_to_patch.reshape(grid, grid).cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map

    def remove_hook(self):
        self.hook.remove()


def overlay_heatmap(img, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """Overlay a heatmap on an RGB image."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    img_uint8 = np.uint8(255 * img)
    overlay = cv2.addWeighted(img_uint8, 1 - alpha, colored, alpha, 0)
    return overlay / 255.0


def visualize_sample(model, sample, gradcam, attn_extractor, save_dir, idx):
    image = sample['image'].unsqueeze(0).to(next(model.parameters()).device)
    img_np = denormalize(sample['image'])

    tasks = {
        'auth': int(sample['auth'].item()),
        'pattern': int(sample['pattern'].item()),
        'defect': int(sample['defect_label'].item()),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original\n{sample['name']}")
    axes[0].axis('off')

    # Grad-CAM for each task
    for col, (task, target_class) in enumerate(tasks.items(), start=1):
        cam = gradcam(image, task, target_class)
        overlay = overlay_heatmap(img_np, cam, alpha=0.5)
        axes[col].imshow(overlay)
        axes[col].set_title(f"Grad-CAM: {task}\nclass={target_class}")
        axes[col].axis('off')

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'sample_{idx:03d}_{sample["name"]}_gradcam.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    # DINOv2 attention
    attn_map = attn_extractor(image)
    attn_resized = cv2.resize(attn_map, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay_attn = overlay_heatmap(img_np, attn_resized, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(attn_resized, cmap='jet')
    axes[1].set_title('DINOv2 CLS attention')
    axes[1].axis('off')
    axes[2].imshow(overlay_attn)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    plt.tight_layout()
    out_path = os.path.join(save_dir, f'sample_{idx:03d}_{sample["name"]}_dinov2_attn.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_combined_fused_correct/checkpoint_last.pt')
    parser.add_argument('--data_dir', type=str, default='data/guizhou_embroidery_correct')
    parser.add_argument('--save_dir', type=str, default='vis_combined_fused')
    parser.add_argument('--num_samples', type=int, default=6,
                        help='Total samples if --balanced is not set; samples per group if --balanced is set.')
    parser.add_argument('--balanced', action='store_true',
                        help='Select a balanced subset: authentic-normal, authentic-defect, fake-normal, fake-defect.')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = load_model(args.checkpoint, device)
    gradcam = GradCAM(model, model.resnet.layer4)
    attn_extractor = DinoV2AttentionExtractor(model)

    test_ds = BlueCalicoDataset(args.data_dir, 'test', get_transforms(False, args.input_size))
    print(f'Test set size: {len(test_ds)}')

    if args.balanced:
        groups = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
        rng = np.random.default_rng(args.seed)
        for i in range(len(test_ds)):
            s = test_ds[i]
            key = (int(s['auth'].item()), int(s['defect_label'].item()))
            if key in groups:
                groups[key].append(i)
        indices = []
        for key, lst in groups.items():
            if lst:
                chosen = rng.choice(lst, size=min(args.num_samples, len(lst)), replace=False).tolist()
                indices.extend(chosen)
        indices = sorted(set(indices))
        print(f'Balanced selection (auth, defect) groups:')
        for key, lst in groups.items():
            print(f'  {key}: {len(lst)} available, selected {min(args.num_samples, len(lst))}')
    else:
        # Pick evenly spaced samples
        indices = np.linspace(0, len(test_ds) - 1, args.num_samples, dtype=int)

    for idx in indices:
        sample = test_ds[idx]
        print(f"Visualizing sample {idx}: {sample['name']} "
              f"auth={sample['auth']} pattern={sample['pattern']} defect={sample['defect_label']}")
        visualize_sample(model, sample, gradcam, attn_extractor, args.save_dir, idx)

    gradcam.remove_hooks()
    attn_extractor.remove_hook()
    print(f'\nSaved visualizations to {args.save_dir}')


if __name__ == '__main__':
    main()
