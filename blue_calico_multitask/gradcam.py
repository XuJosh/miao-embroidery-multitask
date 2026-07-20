"""
Grad-CAM for BlueCalicoNet.
Can target either the authenticity head or the defect classification head.
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_gradcam(model, image, head='auth', target_class=None):
    """
    model: BlueCalicoNet
    image: (1, 3, H, W) tensor
    head: 'auth' or 'defect'
    target_class: int; if None uses argmax of the chosen head
    Returns: heatmap (H, W) numpy array, predicted class
    """
    assert head in ('auth', 'defect')
    model.eval()
    image = image.requires_grad_(True)

    out = model(image, return_features=True)
    logits = out[head]
    cnn_feat = out['cnn_feat']

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    score = logits[0, target_class]
    grads = torch.autograd.grad(score, cnn_feat, retain_graph=False)[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
    cam = (weights * cnn_feat).sum(dim=1).clamp(min=0)    # (1, H, W)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # Resize to input resolution
    _, _, H, W = image.size()
    cam = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam = np.array(cam) / 255.0
    return cam, target_class


def save_cam_overlay(image_tensor, cam, save_path, alpha=0.5):
    """Overlay Grad-CAM heatmap on the original image."""
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    H, W = img.shape[:2]
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam_array = np.array(cam_resized) / 255.0

    cmap = plt.get_cmap('jet')
    colored = cmap(cam_array)[:, :, :3]

    overlay = (1 - alpha) * img + alpha * colored
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def cam_to_mask(cam, threshold=0.5):
    """Binarize a Grad-CAM heatmap to a localization mask."""
    return (cam >= threshold).astype(np.uint8)
