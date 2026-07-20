"""
Enhanced DINOv2 + LoRA + Transformer for Guizhou embroidery authentication.

Additions over the baseline model:
    - Frequency-domain branch (FFT magnitude) to capture stitch periodicity
    - ArcFace margin loss for auth / pattern classification
    - Dynamic multi-task uncertainty weighting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_dinov2_mamba import (
    CNNEncoder, DefectDecoder, DinoV2LoRA,
    PatchMamba, PatchBiGRU, PatchTransformer,
)


# ---------------------------------------------------------------------------
# Frequency-domain branch
# ---------------------------------------------------------------------------

class FrequencyEncoder(nn.Module):
    """Extract log-magnitude FFT features from the image."""

    def __init__(self, out_dim=128, target_size=14):
        super().__init__()
        self.target_size = target_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        gray = x.mean(dim=1, keepdim=True)                 # (B, 1, H, W)
        # Move to frequency domain
        fft = torch.fft.rfft2(gray, norm='ortho')          # (B, 1, H, W//2+1)
        mag = torch.log1p(torch.abs(fft))                  # log magnitude
        # Resize magnitude map to a fixed spatial size
        mag = F.interpolate(mag, size=(self.target_size, self.target_size),
                            mode='bilinear', align_corners=False)
        feat = self.encoder(mag).view(x.size(0), -1)       # (B, 32)
        return self.proj(feat)                             # (B, out_dim)


# ---------------------------------------------------------------------------
# ArcFace margin head
# ---------------------------------------------------------------------------

class ArcMarginProduct(nn.Module):
    """Additive angular margin (ArcFace) classification head."""

    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        # x: (B, in_features)
        cos = F.linear(F.normalize(x), F.normalize(self.weight))  # (B, out_features) in [-1, 1]

        if label is None:
            return self.s * cos

        # One-hot target
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        sin = torch.sqrt(1.0 - cos.pow(2) + 1e-6)
        phi = cos * cos_m - sin * sin_m

        # Add margin only to the target class
        output = one_hot * phi + (1.0 - one_hot) * cos
        output *= self.s
        return output


# ---------------------------------------------------------------------------
# Dynamic multi-task loss (homoscedastic uncertainty weighting)
# ---------------------------------------------------------------------------

class DynamicMultiTaskLoss(nn.Module):
    """Kendall et al. multi-task uncertainty weighting."""

    def __init__(self, w_auth=1.0, w_pattern=0.5, w_defect=1.0,
                 defect_mode='seg', use_dynamic=True):
        super().__init__()
        self.defect_mode = defect_mode
        self.use_dynamic = use_dynamic
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

        # Learnable log variance terms for dynamic weighting
        if use_dynamic:
            self.log_vars = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer('fixed_weights',
                                 torch.tensor([w_auth, w_pattern, w_defect]))

    def forward(self, preds, targets):
        auth_loss = self.ce(preds['auth'], targets['auth'])
        pattern_loss = self.ce(preds['pattern'], targets['pattern'])

        if self.defect_mode == 'seg':
            defect_loss = self.bce(preds['defect'], targets['defect'])
        else:
            defect_loss = self.ce(preds['defect'], targets['defect_label'])

        if self.use_dynamic:
            # precision_i = exp(-log_var_i); total = sum(precision_i * loss_i + log_var_i)
            losses = torch.stack([auth_loss, pattern_loss, defect_loss])
            precisions = torch.exp(-self.log_vars)
            total = torch.sum(precisions * losses + self.log_vars)
        else:
            total = (self.fixed_weights[0] * auth_loss
                     + self.fixed_weights[1] * pattern_loss
                     + self.fixed_weights[2] * defect_loss)

        return {
            'total': total,
            'auth': auth_loss.item(),
            'pattern': pattern_loss.item(),
            'defect': defect_loss.item(),
        }


# ---------------------------------------------------------------------------
# Enhanced multi-task network
# ---------------------------------------------------------------------------

class EmbroideryNetEnhanced(nn.Module):
    def __init__(self, num_patterns=5, defect_mode='cls',
                 dinov2_name='dinov2_vits14',
                 lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 seq_encoder='transformer', seq_layers=2, seq_nhead=8,
                 seq_dim_feedforward=512, vim_d_state=16, vim_expand=2,
                 dropout=0.3, dinov2_checkpoint=False,
                 use_frequency=True, freq_dim=128,
                 use_arcface=True, arcface_s=30.0, arcface_m=0.30,
                 defect_use_fused=False):
        super().__init__()
        assert defect_mode in ('seg', 'cls')
        assert seq_encoder in ('vim', 'gru', 'transformer')
        assert not (defect_use_fused and defect_mode == 'seg'), \
            'defect_use_fused is only supported for defect_mode="cls"'
        self.defect_mode = defect_mode
        self.use_frequency = use_frequency
        self.use_arcface = use_arcface
        self.defect_use_fused = defect_use_fused

        # Branches
        self.cnn = CNNEncoder(in_ch=3)
        self.dinov2 = DinoV2LoRA(
            dinov2_name, lora_r, lora_alpha, lora_dropout,
            use_checkpoint=dinov2_checkpoint,
        )

        d_model = self.dinov2.embed_dim
        if seq_encoder == 'vim':
            self.seq_branch = PatchMamba(
                d_model=d_model, n_layers=seq_layers,
                d_state=vim_d_state, expand_factor=vim_expand,
            )
        elif seq_encoder == 'gru':
            self.seq_branch = PatchBiGRU(
                d_model=d_model, n_layers=seq_layers, dropout=dropout,
            )
        else:
            self.seq_branch = PatchTransformer(
                d_model=d_model, n_layers=seq_layers, nhead=seq_nhead,
                dim_feedforward=seq_dim_feedforward, dropout=dropout,
            )

        if use_frequency:
            self.freq_branch = FrequencyEncoder(out_dim=freq_dim)
        else:
            self.freq_branch = None

        # Feature fusion
        cnn_dim = 256
        dinov2_dim = d_model
        seq_dim = d_model
        fused_dim = cnn_dim + dinov2_dim + seq_dim + (freq_dim if use_frequency else 0)

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Task heads
        feat_dim = 256
        if use_arcface:
            self.auth_head = ArcMarginProduct(feat_dim, 2, s=arcface_s, m=arcface_m)
            self.pattern_head = ArcMarginProduct(feat_dim, num_patterns, s=arcface_s, m=arcface_m)
        else:
            self.auth_head = nn.Linear(feat_dim, 2)
            self.pattern_head = nn.Linear(feat_dim, num_patterns)

        # Defect head: classification accepts cnn_vec (256) or fused feature (256)
        defect_in_dim = 256 if defect_use_fused else cnn_dim
        if defect_mode == 'seg':
            self.defect_head = DefectDecoder(in_ch=cnn_dim)
        else:
            self.defect_head = nn.Sequential(
                nn.Linear(defect_in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )

    def forward(self, x, labels=None, return_features=False):
        # CNN branch
        cnn_feat, cnn_vec = self.cnn(x)

        # DINOv2 + LoRA branch
        dinov2_cls, patch_tokens = self.dinov2(x)

        # Sequence branch
        seq_vec = self.seq_branch(patch_tokens)

        # Frequency branch
        feats = [cnn_vec, dinov2_cls, seq_vec]
        if self.use_frequency:
            freq_vec = self.freq_branch(x)
            feats.append(freq_vec)

        # Fusion
        fused = self.fusion(torch.cat(feats, dim=1))

        # Auth / pattern heads (ArcFace needs labels during training)
        if self.use_arcface:
            auth_logits = self.auth_head(fused, labels['auth'] if labels is not None else None)
            pattern_logits = self.pattern_head(fused, labels['pattern'] if labels is not None else None)
        else:
            auth_logits = self.auth_head(fused)
            pattern_logits = self.pattern_head(fused)

        # Defect head
        if self.defect_mode == 'seg':
            defect_logits = self.defect_head(cnn_feat)
        else:
            defect_input = fused if self.defect_use_fused else cnn_vec
            defect_logits = self.defect_head(defect_input)

        out = {
            'auth': auth_logits,
            'pattern': pattern_logits,
            'defect': defect_logits,
        }
        if return_features:
            out['feat'] = fused
            out['cnn_feat'] = cnn_feat
        return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    labels = {'auth': torch.tensor([0, 1]), 'pattern': torch.tensor([0, 2])}
    for use_freq in (True, False):
        for use_arc in (True, False):
            model = EmbroideryNetEnhanced(
                num_patterns=5, defect_mode='cls',
                seq_encoder='transformer', use_frequency=use_freq,
                use_arcface=use_arc, lora_r=4,
            )
            out = model(x, labels=labels)
            print(f'freq={use_freq}, arc={use_arc}: '
                  f'auth={out["auth"].shape}, pattern={out["pattern"].shape}, defect={out["defect"].shape}')
