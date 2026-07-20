"""
DINOv2 + LoRA + Vision Mamba multi-task network for Guizhou embroidery authentication.

Architecture:
    - CNN branch: local texture features
    - DINOv2 ViT-S/14: pretrained self-supervised global features (frozen + LoRA)
    - Vision Mamba (mambapy VMamba): bidirectional SSM over DINOv2 patch tokens
    - Fusion + multi-task heads for auth / pattern / defect
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# CNN branch (reused from baseline)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DINOv2 + LoRA wrapper
# ---------------------------------------------------------------------------

class DinoV2LoRA(nn.Module):
    def __init__(self, model_name='dinov2_vits14', lora_r=8, lora_alpha=16,
                 lora_dropout=0.05, target_modules=None, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if target_modules is None:
            target_modules = ['qkv']

        local_repo = os.path.expanduser(r'~/.cache/torch/hub/facebookresearch_dinov2_main')
        if os.path.isdir(local_repo):
            self.dinov2 = torch.hub.load(local_repo, model_name, pretrained=True, source='local')
        else:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.embed_dim = self.dinov2.embed_dim

        # Freeze base parameters and inject LoRA adapters
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias='none',
        )
        self.dinov2 = get_peft_model(self.dinov2, config)
        # get_peft_model freezes base params; keep LoRA params trainable

    def _forward_features(self, x):
        """Replica of DINOv2 forward_features with optional gradient checkpointing."""
        x = self.dinov2.prepare_tokens_with_masks(x)

        if self.use_checkpoint and self.training:
            import torch.utils.checkpoint as cp
            for blk in self.dinov2.blocks:
                x = cp.checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.dinov2.blocks:
                x = blk(x)

        x_norm = self.dinov2.norm(x)
        return {
            'x_norm_clstoken': x_norm[:, 0],
            'x_norm_regtokens': x_norm[:, 1 : self.dinov2.num_register_tokens + 1],
            'x_norm_patchtokens': x_norm[:, self.dinov2.num_register_tokens + 1 :],
            'x_prenorm': x,
            'masks': None,
        }

    def forward(self, x):
        # DINOv2 forward_features returns a dict with normalized cls and patch tokens
        out = self._forward_features(x)
        cls_token = out['x_norm_clstoken']          # (B, embed_dim)
        patch_tokens = out['x_norm_patchtokens']    # (B, N_patches, embed_dim)
        return cls_token, patch_tokens


# ---------------------------------------------------------------------------
# Vision Mamba wrapper over patch tokens
# ---------------------------------------------------------------------------

from mambapy.vim import VMamba, MambaConfig


class PatchMamba(nn.Module):
    """Bidirectional Vision Mamba over a sequence of patch tokens."""

    def __init__(self, d_model=384, n_layers=2, d_state=16, expand_factor=2,
                 d_conv=4, dt_rank='auto', pscan=True, bidirectional=True):
        super().__init__()
        config = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            dt_rank=dt_rank,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            use_cuda=False,          # pure PyTorch; works on Windows
            pscan=pscan,
            bidirectional=bidirectional,
        )
        self.vmamba = VMamba(config)

    def forward(self, patch_tokens):
        # patch_tokens: (B, N, d_model)
        x = self.vmamba(patch_tokens)   # (B, N, d_model)
        # Global average pooling over the patch dimension
        return x.mean(dim=1)            # (B, d_model)


class PatchBiGRU(nn.Module):
    """Lightweight bidirectional GRU over patch tokens."""

    def __init__(self, d_model=384, n_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            d_model, d_model, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, patch_tokens):
        # patch_tokens: (B, N, d_model)
        out, _ = self.gru(patch_tokens)          # (B, N, 2*d_model)
        out = out.mean(dim=1)                    # (B, 2*d_model)
        return self.proj(out)                    # (B, d_model)


class PatchTransformer(nn.Module):
    """Lightweight Transformer encoder over patch tokens."""

    def __init__(self, d_model=384, n_layers=2, nhead=8, dim_feedforward=512,
                 dropout=0.1, max_len=512):
        super().__init__()
        assert d_model % nhead == 0, f'd_model={d_model} must be divisible by nhead={nhead}'
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, patch_tokens):
        # patch_tokens: (B, N, d_model)
        b, n, d = patch_tokens.shape
        x = patch_tokens + self.pos_embed[:, :n, :]
        out = self.encoder(x)                    # (B, N, d_model)
        return out.mean(dim=1)                   # (B, d_model)


# ---------------------------------------------------------------------------
# Full multi-task network
# ---------------------------------------------------------------------------

class EmbroideryNet(nn.Module):
    def __init__(self, num_patterns=5, defect_mode='cls',
                 dinov2_name='dinov2_vits14',
                 lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 seq_encoder='vim', seq_layers=2, seq_nhead=8,
                 seq_dim_feedforward=512, vim_d_state=16, vim_expand=2,
                 dropout=0.3, dinov2_checkpoint=True):
        super().__init__()
        assert defect_mode in ('seg', 'cls')
        assert seq_encoder in ('vim', 'gru', 'transformer')
        self.defect_mode = defect_mode
        self.seq_encoder_name = seq_encoder

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
        else:  # transformer
            self.seq_branch = PatchTransformer(
                d_model=d_model, n_layers=seq_layers, nhead=seq_nhead,
                dim_feedforward=seq_dim_feedforward, dropout=dropout,
            )

        # Feature fusion
        cnn_dim = 256
        dinov2_dim = d_model
        seq_dim = d_model
        fused_dim = cnn_dim + dinov2_dim + seq_dim

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Task heads
        self.auth_head = nn.Linear(256, 2)
        self.pattern_head = nn.Linear(256, num_patterns)

        if defect_mode == 'seg':
            self.defect_head = DefectDecoder(in_ch=cnn_dim)
        else:
            self.defect_head = nn.Sequential(
                nn.Linear(cnn_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )

    def forward(self, x, return_features=False):
        # CNN branch
        cnn_feat, cnn_vec = self.cnn(x)      # (B,256,14,14), (B,256)

        # DINOv2 + LoRA branch
        dinov2_cls, patch_tokens = self.dinov2(x)   # (B,384), (B,256,384)

        # Sequence modeling branch (Vim / BiGRU / Transformer)
        vim_vec = self.seq_branch(patch_tokens)     # (B,384)

        # Fusion
        fused = self.fusion(torch.cat([cnn_vec, dinov2_cls, vim_vec], dim=1))

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


# ---------------------------------------------------------------------------
# Loss (same structure as baseline, reusable)
# ---------------------------------------------------------------------------

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
    for enc in ('vim', 'gru', 'transformer'):
        for mode in ('seg', 'cls'):
            model = EmbroideryNet(
                num_patterns=5, defect_mode=mode, seq_encoder=enc,
                seq_layers=2, lora_r=4, dinov2_checkpoint=False,
            )
            out = model(x)
            print(f'[{enc}/{mode}] auth:', out['auth'].shape,
                  'pattern:', out['pattern'].shape,
                  'defect:', out['defect'].shape)
