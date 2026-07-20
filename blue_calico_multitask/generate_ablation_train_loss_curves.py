import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def smooth(y, size=5):
    if len(y) < size:
        return y
    return uniform_filter1d(y, size=size)


def plot_train_loss_only():
    """Plot only training loss curves for ablation variants."""
    variants = {
        'Ours (full)': 'checkpoints_combined_fused_correct/history_cls.json',
        'w/o Patch Transformer': 'checkpoints_combined_fused_correct_no_patchtransformer/history_cls.json',
        'w/o ArcFace': 'checkpoints_combined_fused_correct_no_arcface/history_cls.json',
        'DINOv2-LoRA only': 'checkpoints_guizhou_emb_dinov2_lora_only_correct/history_cls.json',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, path) in enumerate(variants.items()):
        full_path = os.path.join('D:/song/kimi_DEMO/blue_calico_multitask', path)
        if not os.path.exists(full_path):
            print(f'Skipping {name}: {full_path} not found')
            continue

        with open(full_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        epochs = [h['epoch'] for h in history]
        train_total = [h['train']['total'] for h in history]

        ax = axes[idx]
        ax.plot(epochs, train_total, 'r-', alpha=0.35, label='train loss')
        ax.plot(epochs, smooth(train_total), 'g--', linewidth=2.5, label='smooth train loss')

        ax.set_title(name, fontsize=15)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Loss Curves of Ablation Variants', fontsize=17, y=1.02)
    plt.tight_layout()
    out_path = 'D:/song/kimi_DEMO/blue_calico_multitask/ablation_train_loss_curves.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved train-only loss curves to: {out_path}')


if __name__ == '__main__':
    plot_train_loss_only()
