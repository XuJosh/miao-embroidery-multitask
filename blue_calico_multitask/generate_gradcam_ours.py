import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'D:/song/kimi_DEMO/blue_calico_multitask')
from dataset import BlueCalicoDataset, get_transforms
from model_combined import EmbroideryNetCombined


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = 'D:/song/kimi_DEMO/blue_calico_multitask/checkpoints_combined_fused_correct/embroidery_cls_best.pth'
DATA_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/data/guizhou_embroidery_correct'
OUTPUT_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/gradcam_outputs'


class GradCAMExtractor:
    """Extract Grad-CAM for auth head using ResNet50 feature maps."""

    def __init__(self, model):
        self.model = model
        self.feature = None
        self.gradient = None
        self.handle_feat = None
        self.handle_grad = None

    def _save_feature(self, module, input, output):
        self.feature = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def register_hooks(self):
        # Hook on the last layer of ResNet50 (layer4)
        target_layer = self.model.resnet.layer4
        self.handle_feat = target_layer.register_forward_hook(self._save_feature)
        self.handle_grad = target_layer.register_full_backward_hook(self._save_gradient)

    def remove_hooks(self):
        if self.handle_feat:
            self.handle_feat.remove()
        if self.handle_grad:
            self.handle_grad.remove()

    def generate(self, image, head='auth', target_class=None):
        """
        image: (1, 3, H, W) tensor on DEVICE
        Returns: cam (H, W) numpy array, pred_class
        """
        self.model.eval()
        image = image.requires_grad_(True)

        out = self.model(image, labels=None, return_features=True)
        logits = out[head]

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        # feature: (1, C, Hf, Wf), gradient: (1, C, Hf, Wf)
        feat = self.feature
        grads = self.gradient

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * feat).sum(dim=1).clamp(min=0)  # (1, Hf, Wf)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        # Resize to input image size
        _, _, H, W = image.size()
        cam_resized = F.interpolate(
            torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)

        return cam_resized, target_class


def overlay_cam(image_tensor, cam, alpha=0.5):
    """Overlay Grad-CAM on normalized image tensor."""
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    # Denormalize from ImageNet stats for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)

    cmap = plt.get_cmap('jet')
    colored = cmap(cam)[:, :, :3]

    overlay = (1 - alpha) * img + alpha * colored
    overlay = np.clip(overlay, 0, 1)
    return overlay


def visualize_grid(images, cams, titles, save_path, alpha=0.5):
    """Create a grid of original images + Grad-CAM overlays."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i in range(n):
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Original image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original: {titles[i]}', fontsize=12)
        axes[0, i].axis('off')

        # Overlay
        overlay = overlay_cam(images[i], cams[i], alpha)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title('Grad-CAM Overlay', fontsize=12)
        axes[1, i].axis('off')

    plt.suptitle('Grad-CAM Visualization of Ours (Auth Head)', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved Grad-CAM grid to: {save_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model = EmbroideryNetCombined(
        num_patterns=5,
        defect_mode='cls',
        use_patch_transformer=True,
        use_arcface=True,
        auth_defect_use_fused=True,
    ).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)

    extractor = GradCAMExtractor(model)
    extractor.register_hooks()

    # Load dataset and pick a few samples from different classes
    test_ds = BlueCalicoDataset(DATA_DIR, 'test', get_transforms(False, 224))

    # Pick 4 samples: 2 real, 2 fake (based on auth label)
    selected = []
    seen = {0: 0, 1: 0}
    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        auth = sample['auth'].item()
        if seen[auth] < 2:
            selected.append((idx, sample))
            seen[auth] += 1
        if len(selected) >= 4:
            break

    images = []
    cams = []
    titles = []

    for idx, sample in selected:
        img_tensor = sample['image'].unsqueeze(0).to(DEVICE)
        cam, pred = extractor.generate(img_tensor, head='auth')

        label = 'real' if sample['auth'].item() == 1 else 'fake'
        pred_label = 'real' if pred == 1 else 'fake'
        titles.append(f'{label}\npred:{pred_label}')

        images.append(img_tensor.squeeze(0))
        cams.append(cam)

    extractor.remove_hooks()

    save_path = os.path.join(OUTPUT_DIR, 'ours_gradcam_auth_grid.png')
    visualize_grid(images, cams, titles, save_path, alpha=0.5)

    # Also save individual overlays
    for i, (img, cam) in enumerate(zip(images, cams)):
        overlay = overlay_cam(img, cam, alpha=0.5)
        plt.figure(figsize=(5, 5))
        plt.imshow(overlay)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'ours_gradcam_auth_{i}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    main()
