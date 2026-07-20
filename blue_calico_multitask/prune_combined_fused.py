"""
对 EmbroideryNetCombined（combined_fused_correct）进行全局 L1 非结构化剪枝。
- 保留原始网络和原始 checkpoint 不动。
- 生成新的剪枝后 checkpoint 与评估报告。
- 支持通过 --amount 控制剪枝比例（默认 0.3，即 30% 权重置零）。
"""

import os
import argparse
import json
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from model_combined import EmbroideryNetCombined
from dataset import BlueCalicoDataset, get_transforms
from evaluate_combined import evaluate, load_checkpoint


def count_parameters(model, count_nonzero=True):
    """统计总参数量 / 非零参数量。"""
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        if count_nonzero:
            nonzero += (p != 0).sum().item()
    return total, nonzero


def sparsity(model):
    total, nonzero = count_parameters(model, count_nonzero=True)
    return 1.0 - nonzero / total if total else 0.0


def _prune_parameter_tensor(param, amount):
    """直接对某个 Parameter Tensor 做 L1 幅度剪枝（用于 MultiheadAttention.in_proj_weight 等）。"""
    if amount <= 0:
        return 0
    flat = param.view(-1)
    n = flat.numel()
    k = int(math.ceil(amount * n))
    if k == 0:
        return 0
    # 保留绝对值最大的 n-k 个
    threshold = flat.abs().kthvalue(k).values.item()
    mask = (flat.abs() > threshold).view_as(param)
    # 处理并列情况：刚好等于 threshold 的也保留，直到只保留 n-k 个
    kept = mask.sum().item()
    if kept > n - k:
        flat_abs = flat.abs().clone()
        flat_abs[~mask] = float('inf')  # 已经被 mask 掉的不再处理
        # 在 mask==1 的位置里，把最小的 (kept - (n-k)) 个也置零
        need_remove = kept - (n - k)
        if need_remove > 0:
            _, idx = flat_abs.view(-1).topk(int(need_remove), largest=False)
            tmp = mask.view(-1).clone()
            tmp[idx] = 0
            mask = tmp.view_as(param)
    param.data *= mask.to(param.device, param.dtype)
    return (1.0 - mask.float().mean().item())


def prune_model(model, amount=0.3, skip_modules=None):
    """
    对模型中所有 Conv2d / Linear / ArcFace / MultiheadAttention 做非结构化 L1 剪枝。
    skip_modules: 可指定不剪枝的模块名前缀列表（如 ['dinov2.dinov2'] 则保留 DINOv2 backbone）。
    """
    skip_modules = skip_modules or []
    pruned_info = []

    # 1) Conv2d / Linear / ArcFace / LoRA base 等所有含 2D/4D weight 的模块
    for name, m in model.named_modules():
        if any(name.startswith(s) for s in skip_modules):
            continue
        if not hasattr(m, 'weight') or m.weight is None:
            continue
        # PyTorch prune 需要 weight 是 module._parameters 里的 Parameter；
        # PEFT LoRA 等模块的 weight 可能是 property，跳过以免 KeyError。
        if 'weight' not in m._parameters or m._parameters['weight'] is None:
            continue
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.Embedding)):
            continue
        if m.weight.dim() not in (2, 4):
            continue
        n_total = m.weight.numel()
        prune.l1_unstructured(m, name='weight', amount=amount)
        # 永久化：将 mask 应用到 weight 并删除 mask
        prune.remove(m, 'weight')
        n_zero = (m.weight == 0).sum().item()
        pruned_info.append({
            'module': name,
            'type': type(m).__name__,
            'total': n_total,
            'zeros': n_zero,
            'sparsity': n_zero / n_total,
        })

    # 2) TransformerEncoderLayer 中的 MultiheadAttention.in_proj_weight 直接对 Tensor 剪枝
    for name, m in model.named_modules():
        if any(name.startswith(s) for s in skip_modules):
            continue
        if isinstance(m, nn.MultiheadAttention) and m.in_proj_weight is not None:
            n_total = m.in_proj_weight.numel()
            s = _prune_parameter_tensor(m.in_proj_weight, amount)
            n_zero = int(s * n_total)
            pruned_info.append({
                'module': name + '.in_proj_weight',
                'type': 'Parameter',
                'total': n_total,
                'zeros': n_zero,
                'sparsity': s,
            })

    return pruned_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_combined_fused_correct/checkpoint_best.pt',
                        help='原网络 checkpoint 路径')
    parser.add_argument('--data_dir', type=str, default='data/guizhou_embroidery_correct')
    parser.add_argument('--amount', type=float, default=0.3,
                        help='剪枝比例，0~1 之间')
    parser.add_argument('--skip_dinov2', action='store_true',
                        help='是否跳过 DINOv2 backbone 的剪枝（仅剪 ResNet/LoRA/Transformer/Heads）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints_combined_fused_correct_pruned')
    args = parser.parse_args()

    assert 0.0 <= args.amount < 1.0, 'amount 必须在 [0, 1) 之间'

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 模型配置必须与训练时一致（combined_fused_correct 默认值）
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

    # 1. 加载原模型
    print('\n[1/5] 加载原网络 ...')
    model = EmbroideryNetCombined(**model_kwargs).to(device)
    load_checkpoint(model, args.checkpoint, device)
    total_before, nonzero_before = count_parameters(model)
    print(f'  总参数: {total_before:,} | 非零参数: {nonzero_before:,} | 稀疏度: {1 - nonzero_before / total_before:.4%}')

    # 2. 评估原网络
    print('\n[2/5] 评估原网络 ...')
    test_ds = BlueCalicoDataset(args.data_dir, 'test', get_transforms(False, args.input_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=(device.type == 'cuda'))
    metrics_before = evaluate(model, test_loader, device)
    print(f"  auth_acc={metrics_before['auth_acc']:.3f} auth_auc={metrics_before['auth_auc']:.3f} "
          f"pat_acc={metrics_before['pat_acc']:.3f} def_acc={metrics_before['def_acc']:.3f} "
          f"def_auc={metrics_before['def_auc']:.3f}")

    # 3. 剪枝
    print(f'\n[3/5] 执行全局 L1 非结构化剪枝（amount={args.amount}） ...')
    skip = ['dinov2.dinov2'] if args.skip_dinov2 else []
    pruned_info = prune_model(model, amount=args.amount, skip_modules=skip)
    total_after, nonzero_after = count_parameters(model)
    print(f'  总参数: {total_after:,} | 非零参数: {nonzero_after:,} | 稀疏度: {1 - nonzero_after / total_after:.4%}')
    print(f'  非零参数减少: {nonzero_before - nonzero_after:,} ({1 - nonzero_after / nonzero_before:.2%})')

    # 4. 评估剪枝后网络
    print('\n[4/5] 评估剪枝后网络 ...')
    metrics_after = evaluate(model, test_loader, device)
    print(f"  auth_acc={metrics_after['auth_acc']:.3f} auth_auc={metrics_after['auth_auc']:.3f} "
          f"pat_acc={metrics_after['pat_acc']:.3f} def_acc={metrics_after['def_acc']:.3f} "
          f"def_auc={metrics_after['def_auc']:.3f}")

    # 5. 保存
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    pruned_path = os.path.join(args.output_dir, f'{base_name}_pruned_amount{args.amount}.pth')
    torch.save(model.state_dict(), pruned_path)

    report = {
        'original_checkpoint': args.checkpoint,
        'pruned_checkpoint': pruned_path,
        'prune_amount': args.amount,
        'skip_dinov2': args.skip_dinov2,
        'parameters': {
            'total': total_before,
            'nonzero_before': nonzero_before,
            'nonzero_after': nonzero_after,
            'reduction': nonzero_before - nonzero_after,
            'reduction_ratio': 1 - nonzero_after / nonzero_before,
            'sparsity_after': 1 - nonzero_after / total_after,
        },
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'pruned_modules': pruned_info,
    }
    report_path = os.path.join(args.output_dir, f'{base_name}_pruned_amount{args.amount}_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('\n[5/5] 保存完成')
    print(f'  剪枝模型: {pruned_path}')
    print(f'  评估报告: {report_path}')
    print(f'\n文件大小: {os.path.getsize(pruned_path) / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
