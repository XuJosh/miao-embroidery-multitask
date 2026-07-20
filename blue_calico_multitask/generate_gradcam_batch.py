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
from model_external import MultiTaskBackbone


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/data/guizhou_embroidery_correct'
OUTPUT_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/gradcam_comparison_batch'
NUM_SAMPLES = 50  # number of real samples to process


def denormalize_image(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


def overlay_heatmap(img, heatmap, alpha=0.5, cmap='jet'):
    cmap = plt.get_cmap(cmap)
    colored = cmap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img + alpha * colored
    return np.clip(overlay, 0, 1)


def safe_dir_name(net_name):
    """Replace characters that are illegal/ambiguous in directory names."""
    return net_name.replace('/', '-').replace('\\', '-')


class GradCAMExtractor:
    def __init__(self, model, target_layer, feature_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.feature_transform = feature_transform
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
        if self.feature_transform:
            feat = self.feature_transform(feat)
            grads = self.feature_transform(grads)

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


class ViTAttentionExtractor:
    """Extract cls-to-patch attention from the last ViT encoder layer."""

    def __init__(self, model):
        self.model = model
        self.attention = None
        self.hook = None
        self.original_forward = None

    def _patched_forward(self, *args, **kwargs):
        # The TransformerEncoderLayer calls self_attention(x, x, x, need_weights=False)
        # Replace need_weights and set average_attn_weights=False
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False
        return self.original_forward(*args, **kwargs)

    def _save_attention(self, module, input, output):
        # output is (attn_output, attn_weights)
        self.attention = output[1]

    def register_hooks(self):
        # Hook on the last encoder layer's self-attention and patch forward
        target = self.model.extractor.encoder.layers[-1].self_attention
        self.original_forward = target.forward
        target.forward = self._patched_forward.__get__(target, type(target))
        self.hook = target.register_forward_hook(self._save_attention)

    def remove_hooks(self):
        if self.hook:
            self.hook.remove()
        if self.original_forward is not None:
            target = self.model.extractor.encoder.layers[-1].self_attention
            target.forward = self.original_forward

    def generate(self, image):
        self.model.eval()
        with torch.no_grad():
            out = self.model(image)

        # attention: (num_heads, L, L)
        attn = self.attention[0]
        # cls token to patches
        cls_attn = attn[:, 0, 1:].mean(dim=0).cpu().numpy()

        # Reshape to 2D grid (14x14 for 224x224 with patch_size=16)
        num_patches = cls_attn.shape[0]
        grid_size = int(np.sqrt(num_patches))
        attn_map = cls_attn.reshape(grid_size, grid_size)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-8)

        _, _, H, W = image.size()
        attn_resized = F.interpolate(
            torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() + 1e-8)
        return attn_resized


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


def load_external_model(ckpt_dir, backbone, pretrained=False):
    model = MultiTaskBackbone(backbone=backbone, num_patterns=5, defect_mode='cls', pretrained=pretrained).to(DEVICE)
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
    if net_name == 'Ours':
        return model.resnet.layer4
    if net_name == 'ResNet50-CBAM':
        return model.extractor.layer4
    if net_name in ('EfficientNet-B3', 'EfficientNet-B0'):
        return model.extractor.features[-1]
    if net_name == 'MobileNetV4-Conv-Small':
        # timm MobileNetV4 uses 'blocks' instead of 'features'
        return model.extractor.net.blocks[-1]
    if net_name == 'MambaOut':
        # Last stage output is (B, H, W, C)
        return model.extractor.net.stages[3]
    return None


def get_feature_transform(net_name):
    if net_name == 'MambaOut':
        # Convert (B, H, W, C) -> (B, C, H, W)
        return lambda x: x.permute(0, 3, 1, 2)
    return None


def save_overlay(img_tensor, cam, save_path, alpha=0.5):
    img = denormalize_image(img_tensor)
    overlay = overlay_heatmap(img, cam, alpha)
    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models
    models_info = [
        ('Ours', load_ours_model(), 'gradcam'),
        ('ResNet50-CBAM', load_external_model('checkpoints_external_resnet50_cbam_correct', 'resnet50_cbam'), 'gradcam'),
        ('EfficientNet-B0', load_external_model('checkpoints_external_efficientnet_b0_correct', 'efficientnet_b0'), 'gradcam'),
        ('EfficientNet-B3', load_external_model('checkpoints_external_efficientnet_b3_correct', 'efficientnet_b3'), 'gradcam'),
        ('MobileNetV4-Conv-Small', load_external_model('checkpoints_external_mobilenetv4_conv_small.e1200_r224_in1k_correct', 'mobilenetv4_conv_small.e1200_r224_in1k'), 'gradcam'),
        ('MambaOut', load_external_model('checkpoints_external_mambaout_small.in1k_correct', 'mambaout_small.in1k'), 'gradcam'),
        ('ViT-B/16', load_external_model('checkpoints_external_vit_b_16_correct', 'vit_b_16'), 'gradcam_vit'),
    ]

    extractors = {}
    for net_name, model, method in models_info:
        if method == 'gradcam':
            target_layer = get_target_layer(model, net_name)
            feature_transform = get_feature_transform(net_name)
            extractor = GradCAMExtractor(model, target_layer, feature_transform)
        else:
            extractor = ViTAttentionExtractor(model)
        extractor.register_hooks()
        extractors[net_name] = (extractor, method)
        os.makedirs(os.path.join(OUTPUT_DIR, safe_dir_name(net_name)), exist_ok=True)

    # Select first NUM_SAMPLES real samples
    test_ds = BlueCalicoDataset(DATA_DIR, 'test', get_transforms(False, 224))
    selected = []
    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        if sample['auth'].item() == 1:
            selected.append((idx, sample))
        if len(selected) >= NUM_SAMPLES:
            break

    print(f'Processing {len(selected)} real samples for {len(models_info)} networks...')

    for idx, sample in selected:
        img_tensor = sample['image'].unsqueeze(0).to(DEVICE)
        for net_name, (extractor, method) in extractors.items():
            if method == 'gradcam_vit':
                cam = extractor.generate(img_tensor)
                pred = 1
            else:
                cam, pred = extractor.generate(img_tensor, head='auth')
            save_path = os.path.join(OUTPUT_DIR, safe_dir_name(net_name), f'sample_{idx:03d}_pred_{pred}.png')
            save_overlay(img_tensor.squeeze(0), cam, save_path)
        print(f'  Done sample {idx}')

    for extractor, method in extractors.values():
        extractor.remove_hooks()

    print(f'All saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
