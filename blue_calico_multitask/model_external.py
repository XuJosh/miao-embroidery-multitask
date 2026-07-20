"""
Generic multi-task baseline models using popular ImageNet backbones.
Adds recent 2025/2026 backbones for traditional textile/embroidery comparisons:
    - ResNet50-CBAM (2026 ethnic embroidery pattern classification)
    - MobileNetV4-Conv-Small (2026 Jin-Cang embroidery stitch recognition)
    - MambaOut Small / Tiny (CVPR 2025 image classification backbones)
"""

import os
import torch
import torch.nn as nn
import torchvision

try:
    import timm
except ImportError:
    timm = None

from model_dinov2_mamba import DinoV2LoRA


# ---------------------------------------------------------------------------
# CBAM attention (Woo et al., ECCV 2018) used in recent 2026 embroidery work
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class ResNet50_CBAM(nn.Module):
    """ResNet50 with CBAM modules after each residual stage."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = torchvision.models.resnet50(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = nn.Sequential(base.layer1, CBAM(256))
        self.layer2 = nn.Sequential(base.layer2, CBAM(512))
        self.layer3 = nn.Sequential(base.layer3, CBAM(1024))
        self.layer4 = nn.Sequential(base.layer4, CBAM(2048))
        self.avgpool = base.avgpool
        self.feature_dim = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


# ---------------------------------------------------------------------------
# MobileNetV4 wrapper via timm (used in 2026 Lite-YOLOv11s embroidery work)
# ---------------------------------------------------------------------------

class MobileNetV4Extractor(nn.Module):
    def __init__(self, variant='mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True):
        super().__init__()
        if timm is None:
            raise ImportError('timm is required for MobileNetV4 backbones. Install with pip install timm')
        self.net = timm.create_model(variant, pretrained=pretrained, num_classes=0)
        # Determine feature dimension with a dummy forward pass
        self.net.eval()
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            feat = self.net(dummy)
            self.feature_dim = feat.shape[-1]

    def forward(self, x):
        return self.net(x)


class TimmFeatureExtractor(nn.Module):
    """Generic timm wrapper for any model that supports num_classes=0."""
    def __init__(self, variant, pretrained=True):
        super().__init__()
        if timm is None:
            raise ImportError('timm is required for this backbone. Install with pip install timm')
        self.net = timm.create_model(variant, pretrained=pretrained, num_classes=0)
        self.net.eval()
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            feat = self.net(dummy)
            self.feature_dim = feat.shape[-1]
            if feat.dim() > 2:
                feat = feat.mean(dim=tuple(range(2, feat.dim())))
                self.feature_dim = feat.shape[-1]

    def forward(self, x):
        out = self.net(x)
        if out.dim() > 2:
            out = out.mean(dim=tuple(range(2, out.dim())))
        return out


# Known timm backbones that can be loaded through TimmFeatureExtractor.
TIMM_BACKBONES = {
    'mobilenetv4_conv_small.e1200_r224_in1k',
    'ghostnetv3_100.in1k',
    'mambaout_small.in1k',
    'mambaout_tiny.in1k',
}


# ---------------------------------------------------------------------------
# Multi-task backbone dispatcher
# ---------------------------------------------------------------------------

class MultiTaskBackbone(nn.Module):
    def __init__(self, backbone='resnet50', num_patterns=5, defect_mode='cls', dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.defect_mode = defect_mode
        self.pretrained = pretrained
        self.feature_dim = None
        self.extractor = self._build_extractor(backbone)
        assert self.feature_dim is not None

        def head(out_dim):
            return nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, out_dim),
            )

        self.auth_head = head(2)
        self.pattern_head = head(num_patterns)
        if defect_mode == 'seg':
            self.defect_head = head(1)
        else:
            self.defect_head = head(2)

    def _build_extractor(self, backbone):
        if backbone == 'resnet50':
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if self.pretrained else None
            m = torchvision.models.resnet50(weights=weights)
            self.feature_dim = m.fc.in_features
            m.fc = nn.Identity()
            return m

        if backbone == 'resnet50_cbam':
            m = ResNet50_CBAM(pretrained=self.pretrained)
            self.feature_dim = m.feature_dim
            return m

        if backbone.startswith('mobilenetv4_'):
            m = MobileNetV4Extractor(variant=backbone, pretrained=self.pretrained)
            self.feature_dim = m.feature_dim
            return m

        if backbone.startswith('mambaout_') or backbone in TIMM_BACKBONES:
            m = TimmFeatureExtractor(variant=backbone, pretrained=self.pretrained)
            self.feature_dim = m.feature_dim
            return m

        # Legacy torchvision / custom backbones below
        if backbone == 'efficientnet_b0':
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if self.pretrained else None
            m = torchvision.models.efficientnet_b0(weights=weights)
            self.feature_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m

        if backbone == 'efficientnet_b3':
            weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1 if self.pretrained else None
            m = torchvision.models.efficientnet_b3(weights=weights)
            self.feature_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m

        if backbone == 'efficientnet_b4':
            weights = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1 if self.pretrained else None
            m = torchvision.models.efficientnet_b4(weights=weights)
            self.feature_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m

        if backbone == 'convnext_tiny':
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if self.pretrained else None
            m = torchvision.models.convnext_tiny(weights=weights)
            self.feature_dim = m.classifier[2].in_features
            # Keep Flatten + LayerNorm, replace final Linear with Identity
            m.classifier[2] = nn.Identity()
            return m

        if backbone == 'swin_t':
            weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1 if self.pretrained else None
            m = torchvision.models.swin_t(weights=weights)
            self.feature_dim = m.head.in_features
            m.head = nn.Identity()
            return m

        if backbone == 'vit_b_16':
            weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1 if self.pretrained else None
            m = torchvision.models.vit_b_16(weights=weights)
            self.feature_dim = m.heads.head.in_features
            m.heads = nn.Identity()
            return m

        if backbone == 'dinov2_vits14_frozen':
            local_repo = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
            if self.pretrained and os.path.isdir(local_repo):
                dino = torch.hub.load(local_repo, 'dinov2_vits14', pretrained=True, source='local')
            elif self.pretrained:
                dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
            else:
                raise ValueError('dinov2_vits14_frozen requires pretrained=True')
            for p in dino.parameters():
                p.requires_grad = False
            self.feature_dim = dino.embed_dim
            return dino

        if backbone == 'dinov2_vits14_lora':
            dino = DinoV2LoRA(
                model_name='dinov2_vits14',
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=['qkv'],
                use_checkpoint=False,
            )
            self.feature_dim = dino.embed_dim
            return dino

        raise ValueError(f'Unknown backbone: {backbone}')

    def forward(self, x):
        feat = self.extractor(x)
        if self.backbone_name == 'dinov2_vits14_lora':
            feat = feat[0]  # keep cls token, discard patch tokens
        return {
            'auth': self.auth_head(feat),
            'pattern': self.pattern_head(feat),
            'defect': self.defect_head(feat),
        }
