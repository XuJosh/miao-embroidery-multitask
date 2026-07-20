"""
加载并推理 combined_fused 原网络（未剪枝版）。
"""

import gzip
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_combined import EmbroideryNetCombined


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PATTERN_CHINESE = {
    0: '辫绣',
    1: '堆绣',
    2: '马尾绣',
    3: '其他',
    4: '数纱马尾绣',
}

AUTH_LABEL = {0: '伪作 / 机绣', 1: '真品 / 手工'}
DEFECT_LABEL = {0: '无疵点', 1: '有疵点'}

# 与训练 combined_fused_correct 时完全一致的超参（原网络）
MODEL_KWARGS = {
    'num_patterns': 5,
    'defect_mode': 'cls',
    'dinov2_name': 'dinov2_vits14',
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'seq_layers': 2,
    'seq_nhead': 8,
    'seq_dim_feedforward': 512,
    'dropout': 0.3,
    'use_frequency': False,
    'use_arcface': True,
    'arcface_s': 30.0,
    'arcface_m': 0.30,
    'auth_defect_use_fused': True,
    'use_patch_transformer': True,
}


def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_model(device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EmbroideryNetCombined(**MODEL_KWARGS)
    model.to(device)
    model.eval()
    return model, torch.device(device)


def load_checkpoint(model, checkpoint_path, device='cpu'):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'找不到 checkpoint: {checkpoint_path}')

    if str(checkpoint_path).endswith('.gz'):
        opener = gzip.open(checkpoint_path, 'rb')
    else:
        opener = open(checkpoint_path, 'rb')

    with opener as f:
        ckpt = torch.load(f, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        epoch = ckpt.get('epoch', None)
    else:
        state_dict = ckpt
        epoch = None

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    return epoch


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


@torch.no_grad()
def predict(model, image_path, device, input_size=224):
    img = Image.open(image_path).convert('RGB')
    tensor = get_transform(input_size)(img).unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        out = model(tensor)

    auth_prob = F.softmax(out['auth'].float(), dim=1)[0]
    pat_prob = F.softmax(out['pattern'].float(), dim=1)[0]
    def_prob = F.softmax(out['defect'].float(), dim=1)[0]

    auth_pred = int(auth_prob.argmax())
    pat_pred = int(pat_prob.argmax())
    def_pred = int(def_prob.argmax())

    return {
        'auth': {
            'label': auth_pred,
            'name': AUTH_LABEL[auth_pred],
            'prob': float(auth_prob[auth_pred]),
            'probs': {AUTH_LABEL[i]: float(auth_prob[i]) for i in range(len(auth_prob))},
        },
        'pattern': {
            'label': pat_pred,
            'name': PATTERN_CHINESE[pat_pred],
            'prob': float(pat_prob[pat_pred]),
            'probs': {PATTERN_CHINESE[i]: float(pat_prob[i]) for i in range(len(pat_prob))},
        },
        'defect': {
            'label': def_pred,
            'name': DEFECT_LABEL[def_pred],
            'prob': float(def_prob[def_pred]),
            'probs': {DEFECT_LABEL[i]: float(def_prob[i]) for i in range(len(def_prob))},
        },
    }


def predict_from_path(checkpoint_path, image_path, device='auto', input_size=224):
    model, device = build_model(device)
    load_checkpoint(model, checkpoint_path, device)
    return predict(model, image_path, device, input_size)
