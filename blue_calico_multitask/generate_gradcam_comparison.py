import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import torchvision

sys.path.insert(0, 'D:/song/kimi_DEMO/blue_calico_multitask')
from dataset import BlueCalicoDataset, get_transforms
from model_combined import EmbroideryNetCombined
from model_external import MultiTaskBackbone


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/data/guizhou_embroidery_correct'
OUTPUT_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/gradcam_comparison_outputs'


def denormalize_image(img_tensor):
    """Convert normalized tensor to [0,1] numpy image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


def overlay_heatmap(img, heatmap, alpha=0.5, cmap='jet'):
    """Overlay a heatmap on an image."""
    cmap = plt.get_cmap(cmap)
    colored = cmap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img + alpha * colored
    return np.clip(overlay, 0, 1)


class GradCAMExtractor:
    """Generic Grad-CAM extractor using hooks."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature = None
        self.gradient = None
        self.hook_f = None
        self.hook_b = None

    def _save_feature(self, module, input, output):
        self.feature = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def register_hooks(self):
        self.hook_f = self.target_layer.register_forward_hook(self._save_feature)
        self.hook_b = self.target_layer.register_full_backward_hook(self._save_gradient)

    def remove_hooks(self):
        if self.hook_f:
            self.hook_f.remove()
        if self.hook_b:
            self.hook_b.remove()

    def generate(self, image, head='auth', target_class=None):
        self.model.eval()
        image = image.requires_grad_(True)

        out = self.model(image)
        logits = out[head]

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        feat = self.feature
        grads = self.gradient

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feat).sum(dim=1).clamp(min=0)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        _, _, H, W = image.size()
        cam_resized = F.interpolate(
            torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)
        return cam_resized, target_class


class ViTGradCAMExtractor:
    """Extract Grad-CAM from ViT encoder output tokens (reshaped as spatial features)."""

    def __init__(self, model):
        self.model = model
        self.feature = None
        self.gradient = None
        self.hook_f = None
        self.hook_b = None

    def _save_feature(self, module, input, output):
        self.feature = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def register_hooks(self):
        # Hook on the encoder output (before final layer norm)
        target = self.model.extractor.encoder.ln
        self.hook_f = target.register_forward_hook(self._save_feature)
        self.hook_b = target.register_full_backward_hook(self._save_gradient)

    def remove_hooks(self):
        if self.hook_f:
            self.hook_f.remove()
        if self.hook_b:
            self.hook_b.remove()

    def generate(self, image):
        self.model.eval()
        image = image.requires_grad_(True)

        out = self.model(image)
        logits = out['auth']
        target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # feature: (1, L, D) where L = 1 + num_patches
        feat = self.feature  # (1, 197, D)
        grads = self.gradient  # (1, 197, D)

        # Remove cls token, keep patch tokens
        patch_feat = feat[:, 1:, :]  # (1, num_patches, D)
        patch_grads = grads[:, 1:, :]  # (1, num_patches, D)

        # Reshape to spatial grid
        num_patches = patch_feat.shape[1]
        grid_size = int(np.sqrt(num_patches))
        D = patch_feat.shape[2]
        patch_feat = patch_feat.reshape(1, grid_size, grid_size, D).permute(0, 3, 1, 2)
        patch_grads = patch_grads.reshape(1, grid_size, grid_size, D).permute(0, 3, 1, 2)

        weights = patch_grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * patch_feat).sum(dim=1).clamp(min=0)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        _, _, H, W = image.size()
        cam_resized = F.interpolate(
            torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)
        return cam_resized, target_class


def load_ours_model():
    model = EmbroideryNetCombined(
        num_patterns=5, defect_mode='cls', use_patch_transformer=True,
        use_arcface=True, auth_defect_use_fused=True,
    ).to(DEVICE)
    ckpt = torch.load(
        'D:/song/kimi_DEMO/blue_calico_multitask/checkpoints_combined_fused_correct/embroidery_cls_best.pth',
        map_location=DEVICE, weights_only=False
    )
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    return model


def load_external_model(ckpt_dir, backbone):
    model = MultiTaskBackbone(backbone=backbone, num_patterns=5, defect_mode='cls').to(DEVICE)
    # Try multiple checkpoint names
    candidates = ['embroidery_cls_best.pth', 'embroidery_cls_epoch25.pth',
                  'checkpoint_best.pt', 'checkpoint_last.pt']
    ckpt_path = None
    for cand in candidates:
        p = os.path.join(ckpt_dir, cand)
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    return model


def get_target_layer(model, net_name):
    """Get the target layer for Grad-CAM based on network type."""
    if net_name == 'Ours':
        return model.resnet.layer4
    if net_name == 'ResNet50-CBAM':
        # extractor is ResNet50_CBAM, layer4 is a Sequential with CBAM at the end
        return model.extractor.layer4
    if net_name == 'EfficientNet-B3':
        # extractor is torchvision EfficientNet, last conv block in features
        return model.extractor.features[-1]
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models
    models_info = [
        ('Ours', load_ours_model(), 'gradcam'),
        ('ResNet50-CBAM', load_external_model('checkpoints_external_resnet50_cbam_correct', 'resnet50_cbam'), 'gradcam'),
        ('EfficientNet-B3', load_external_model('checkpoints_external_efficientnet_b3_correct', 'efficientnet_b3'), 'gradcam'),
        ('ViT-B/16', load_external_model('checkpoints_external_vit_b_16_correct', 'vit_b_16'), 'gradcam_vit'),
    ]

    # Extractors
    extractors = {}
    for net_name, model, method in models_info:
        if method == 'gradcam':
            target_layer = get_target_layer(model, net_name)
            extractor = GradCAMExtractor(model, target_layer)
            extractor.register_hooks()
            extractors[net_name] = (extractor, 'gradcam')
        else:
            extractor = ViTGradCAMExtractor(model)
            extractor.register_hooks()
            extractors[net_name] = (extractor, 'gradcam_vit')

    # Select 6 real samples
    test_ds = BlueCalicoDataset(DATA_DIR, 'test', get_transforms(False, 224))
    selected = []
    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        if sample['auth'].item() == 1:  # real samples only
            selected.append((idx, sample))
        if len(selected) >= 6:
            break

    # Generate CAMs
    samples = []
    for idx, sample in selected:
        img_tensor = sample['image'].unsqueeze(0).to(DEVICE)
        cams = {}
        for net_name, (extractor, method) in extractors.items():
            if method == 'gradcam_vit':
                cam, pred = extractor.generate(img_tensor)
            else:
                cam, pred = extractor.generate(img_tensor, head='auth')
            cams[net_name] = cam
        samples.append({
            'image': img_tensor.squeeze(0),
            'cams': cams,
            'name': f'sample_{idx}',
        })

    # Remove hooks
    for extractor, method in extractors.values():
        extractor.remove_hooks()

    # Plot grid: rows=samples, cols=Original + Ours + ResNet50-CBAM + EfficientNet-B3 + ViT-B/16
    n_samples = len(samples)
    n_cols = 5
    net_names = ['Ours', 'ResNet50-CBAM', 'EfficientNet-B3', 'ViT-B/16']

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(samples):
        img = denormalize_image(sample['image'])

        # Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original', fontsize=12)
        axes[i, 0].axis('off')

        # CAMs
        for j, net_name in enumerate(net_names):
            cam = sample['cams'][net_name]
            overlay = overlay_heatmap(img, cam, alpha=0.5)
            axes[i, j + 1].imshow(overlay)
            title = net_name if i == 0 else ''
            axes[i, j + 1].set_title(title, fontsize=12)
            axes[i, j + 1].axis('off')

    plt.suptitle('Grad-CAM Visualization Comparison across Networks (Real Samples)', fontsize=15, y=1.00)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'gradcam_comparison_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved comparison grid to: {save_path}')


if __name__ == '__main__':
    main()
