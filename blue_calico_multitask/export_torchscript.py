"""
将训练好的 combined_fused_correct 原网络导出为 TorchScript（供 PyTorch Mobile / Android 使用）。
推理路径：labels=None，return_features=False。
"""

import os
import argparse
import torch

from model_combined import EmbroideryNetCombined
from evaluate_combined import load_checkpoint


class InferenceWrapper(torch.nn.Module):
    """
    TorchScript 不允许输出 dict，因此包装成返回 (auth, pattern, defect) 元组。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out['auth'], out['pattern'], out['defect']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_combined_fused_correct/embroidery_cls_best.pth')
    parser.add_argument('--output', type=str, default='checkpoints_combined_fused_correct_android/model.ptl')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    model_kwargs = {
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

    device = torch.device(args.device)
    print('Building model...')
    model = EmbroideryNetCombined(**model_kwargs).to(device).eval()

    print(f'Loading checkpoint from {args.checkpoint}...')
    load_checkpoint(model, args.checkpoint, device)

    # 构造示例输入；labels 默认为 None，return_features 默认为 False
    example = torch.randn(1, 3, 224, 224).to(device)

    print('Tracing model...')
    wrapped = InferenceWrapper(model).eval()
    try:
        traced = torch.jit.trace(wrapped, example)
    except Exception as e:
        print(f'Trace failed: {e}')
        raise

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 保存为 Lite Interpreter 格式，Android 推荐
    print(f'Saving TorchScript Lite model to {args.output}...')
    try:
        torch.jit._save_for_lite_interpreter(traced, args.output)
    except AttributeError:
        # 旧版 PyTorch 可能没有 _save_for_lite_interpreter
        traced._save_for_lite_interpreter(args.output)

    print('Done.')
    print(f'File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
