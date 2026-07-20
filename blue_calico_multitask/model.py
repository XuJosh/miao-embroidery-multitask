"""
Multi-task network for Blue Calico (南通蓝印花布) intelligent authentication.

Tasks:
  1. Authenticity classification (real / fake)
  2. Pattern/style classification
  3. Defect segmentation (pixel-wise anomaly map)

Architecture:
  - CNN branch: extracts local texture features
  - ViT branch: extracts global pattern structure
  - Fusion + multi-task heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B, dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)      # (B, N_patches, dim)
        return x


class TinyViT(nn.Module):
    """Lightweight Vision Transformer for global pattern structure."""

    def __init__(self, img_size=224, patch_size=16, dim=128, depth=4, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]   # class token


def _conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNNEncoder(nn.Module):
    """Lightweight CNN encoder for local texture."""

    def __init__(self, in_ch=3):
        super().__init__()
        self.layers = nn.Sequential(
            _conv_block(in_ch, 32, 2),   # 112
            _conv_block(32, 64, 2),      # 56
            _conv_block(64, 128, 2),     # 28
            _conv_block(128, 256, 2),    # 14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.layers(x)            # (B, 256, 14, 14)
        pooled = self.gap(feat).view(x.size(0), -1)
        return feat, pooled


class DefectDecoder(nn.Module):
    """Upsample CNN feature map to a full-resolution defect mask."""

    def __init__(self, in_ch=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)           # (B, 1, 224, 224)


class BlueCalicoNet(nn.Module):
    def __init__(self, num_patterns, cnn_dim=256, vit_dim=128,
                 img_size=224, patch_size=16, vit_depth=4, vit_heads=4,
                 dropout=0.3, defect_mode='seg'):
        """
        defect_mode:
            'seg'  -> pixel-wise defect segmentation (needs masks)
            'cls'  -> image-level defect classification (only needs 0/1 labels)
        """
        super().__init__()
        assert defect_mode in ('seg', 'cls')
        self.defect_mode = defect_mode

        self.cnn = CNNEncoder(in_ch=3)
        self.vit = TinyViT(img_size=img_size, patch_size=patch_size,
                           dim=vit_dim, depth=vit_depth, heads=vit_heads,
                           dropout=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.auth_head = nn.Linear(256, 2)        # real / fake
        self.pattern_head = nn.Linear(256, num_patterns)

        if defect_mode == 'seg':
            self.defect_head = DefectDecoder(in_ch=cnn_dim)
        else:
            # image-level defect classifier: 0=no defect, 1=has defect
            self.defect_head = nn.Sequential(
                nn.Linear(cnn_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )

    def forward(self, x, return_features=False):
        cnn_feat, cnn_vec = self.cnn(x)
        vit_vec = self.vit(x)

        fused = self.fusion(torch.cat([cnn_vec, vit_vec], dim=1))

        auth_logits = self.auth_head(fused)
        pattern_logits = self.pattern_head(fused)

        if self.defect_mode == 'seg':
            defect_logits = self.defect_head(cnn_feat)
        else:
            defect_logits = self.defect_head(cnn_vec)

        out = {
            'auth': auth_logits,
            'pattern': pattern_logits,
            'defect': defect_logits,
        }
        if return_features:
            out['cnn_feat'] = cnn_feat
        return out


class MultiTaskLoss(nn.Module):
    """Weighted sum of three task losses."""

    def __init__(self, w_auth=1.0, w_pattern=0.5, w_defect=1.0, defect_mode='seg'):
        super().__init__()
        self.w_auth = w_auth
        self.w_pattern = w_pattern
        self.w_defect = w_defect
        self.defect_mode = defect_mode
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        auth_loss = self.ce(preds['auth'], targets['auth'])
        pattern_loss = self.ce(preds['pattern'], targets['pattern'])

        if self.defect_mode == 'seg':
            defect_loss = self.bce(preds['defect'], targets['defect'])
        else:
            defect_loss = self.ce(preds['defect'], targets['defect_label'])

        total = (self.w_auth * auth_loss
                 + self.w_pattern * pattern_loss
                 + self.w_defect * defect_loss)
        return {
            'total': total,
            'auth': auth_loss.item(),
            'pattern': pattern_loss.item(),
            'defect': defect_loss.item(),
        }


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    for mode in ('seg', 'cls'):
        model = BlueCalicoNet(num_patterns=5, defect_mode=mode)
        out = model(x)
        print(f'[{mode}] auth:', out['auth'].shape,
              'pattern:', out['pattern'].shape,
              'defect:', out['defect'].shape)
