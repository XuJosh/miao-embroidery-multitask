"""
Combined ResNet50 + DINOv2/Transformer multi-task network.

Two modes:
    1. auth/defect directly from ResNet50 (default).
    2. auth/defect from the fused ResNet50 + DINOv2 + Transformer feature.

Pattern always uses the fused feature.
"""

import torch
import torch.nn as nn
import torchvision

from model_enhanced import (
    DinoV2LoRA,
    PatchTransformer,
    FrequencyEncoder,
    ArcMarginProduct,
    DynamicMultiTaskLoss,
)


class ResNet50Backbone(nn.Module):
    """Pretrained ResNet50 up to layer4, returning both feature map and global vector."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)           # (B, 2048, 7, 7)
        vec = self.avgpool(feat).view(feat.size(0), -1)  # (B, 2048)
        return feat, vec


class EmbroideryNetCombined(nn.Module):
    """ResNet50 + DINOv2/Transformer fusion for multi-task embroidery classification."""

    def __init__(self, num_patterns=5, defect_mode='cls',
                 dinov2_name='dinov2_vits14',
                 lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 seq_layers=2, seq_nhead=8, seq_dim_feedforward=512,
                 dropout=0.3,
                 use_frequency=False, freq_dim=128,
                 use_arcface=True, arcface_s=30.0, arcface_m=0.30,
                 auth_defect_use_fused=False,
                 use_patch_transformer=True):
        super().__init__()
        assert defect_mode == 'cls', 'Combined model only supports defect_mode="cls"'
        self.defect_mode = defect_mode
        self.use_frequency = use_frequency
        self.use_arcface = use_arcface
        self.auth_defect_use_fused = auth_defect_use_fused
        self.use_patch_transformer = use_patch_transformer

        # Branches
        self.resnet = ResNet50Backbone(pretrained=True)
        self.dinov2 = DinoV2LoRA(
            dinov2_name, lora_r, lora_alpha, lora_dropout,
            use_checkpoint=False,
        )
        d_model = self.dinov2.embed_dim
        if self.use_patch_transformer:
            self.seq_branch = PatchTransformer(
                d_model=d_model, n_layers=seq_layers, nhead=seq_nhead,
                dim_feedforward=seq_dim_feedforward, dropout=dropout,
            )
        else:
            self.seq_branch = None
        if use_frequency:
            self.freq_branch = FrequencyEncoder(out_dim=freq_dim)
        else:
            self.freq_branch = None

        # Shared fusion for pattern (and optionally auth/defect)
        res_dim = 2048
        fused_dim = (res_dim + d_model
                     + (d_model if self.use_patch_transformer else 0)
                     + (freq_dim if use_frequency else 0))
        # Kept as 'pattern_fusion' for backward compatibility with earlier checkpoints.
        self.pattern_fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Task heads
        feat_dim = 256
        auth_defect_in_dim = feat_dim if auth_defect_use_fused else res_dim

        if use_arcface:
            self.auth_head = ArcMarginProduct(auth_defect_in_dim, 2, s=arcface_s, m=arcface_m)
            self.pattern_head = ArcMarginProduct(feat_dim, num_patterns, s=arcface_s, m=arcface_m)
        else:
            self.auth_head = nn.Linear(auth_defect_in_dim, 2)
            self.pattern_head = nn.Linear(feat_dim, num_patterns)

        self.defect_head = nn.Sequential(
            nn.Linear(auth_defect_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x, labels=None, return_features=False):
        # ResNet50 branch
        res_feat, res_vec = self.resnet(x)   # res_feat (B,2048,7,7), res_vec (B,2048)

        # DINOv2 + LoRA branch
        dinov2_cls, patch_tokens = self.dinov2(x)

        # Shared fused feature
        feats = [res_vec, dinov2_cls]
        if self.use_patch_transformer:
            seq_vec = self.seq_branch(patch_tokens)
            feats.append(seq_vec)
        if self.use_frequency:
            freq_vec = self.freq_branch(x)
            feats.append(freq_vec)
        fused = self.pattern_fusion(torch.cat(feats, dim=1))

        # Auth / pattern / defect inputs
        auth_input = fused if self.auth_defect_use_fused else res_vec
        defect_input = fused if self.auth_defect_use_fused else res_vec

        if self.use_arcface:
            auth_logits = self.auth_head(auth_input, labels['auth'] if labels is not None else None)
            pattern_logits = self.pattern_head(fused, labels['pattern'] if labels is not None else None)
        else:
            auth_logits = self.auth_head(auth_input)
            pattern_logits = self.pattern_head(fused)

        defect_logits = self.defect_head(defect_input)

        out = {
            'auth': auth_logits,
            'pattern': pattern_logits,
            'defect': defect_logits,
        }
        if return_features:
            out['res_vec'] = res_vec
            out['fused'] = fused
        return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    labels = {'auth': torch.tensor([0, 1]), 'pattern': torch.tensor([0, 2])}
    for fused in (False, True):
        model = EmbroideryNetCombined(use_frequency=False, use_arcface=True,
                                      auth_defect_use_fused=fused)
        out = model(x, labels)
        print(f'auth_defect_use_fused={fused}:', out['auth'].shape, out['pattern'].shape, out['defect'].shape)
