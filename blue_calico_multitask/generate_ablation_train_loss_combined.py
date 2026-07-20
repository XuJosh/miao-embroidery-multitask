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


def plot_combined_train_loss():
    """Plot all ablation train loss curves in a single figure."""
    variants = [
        ('Ours (full)', 'checkpoints_combined_fused_correct/history_cls.json', '#E41A1C', '-', 'o'),
        ('w/o Patch Transformer', 'checkpoints_combined_fused_correct_no_patchtransformer/history_cls.json', '#377EB8', '--', 's'),
        ('w/o ArcFace', 'checkpoints_combined_fused_correct_no_arcface/history_cls.json', '#4DAF4A', '-.', '^'),
        ('DINOv2-LoRA only', 'checkpoints_guizhou_emb_dinov2_lora_only_correct/history_cls.json', '#FF7F00', ':', 'D'),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, path, color, linestyle, marker in variants:
        full_path = os.path.join('D:/song/kimi_DEMO/blue_calico_multitask', path)
        if not os.path.exists(full_path):
            print(f'Skipping {name}: {full_path} not found')
            continue

        with open(full_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        epochs = np.array([h['epoch'] for h in history])
        train_total = np.array([h['train']['total'] for h in history])

        # Only keep first 50 epochs
        mask = epochs <= 50
        epochs = epochs[mask]
        train_total = train_total[mask]
        train_smooth = smooth(train_total, size=5)

        # Plot raw with low alpha and thin line
        ax.plot(epochs, train_total, color=color, alpha=0.18, linewidth=1)
        # Plot smoothed with thicker line and marker every N epochs
        ax.plot(epochs, train_smooth, color=color, linestyle=linestyle,
                linewidth=2.5, label=name, marker=marker,
                markevery=max(1, len(epochs) // 8), markersize=6, markerfacecolor='white',
                markeredgewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Training Loss Curves of Ablation Variants (First 50 Epochs)', fontsize=15)
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0, right=50)
    ax.set_ylim(0, 2.0)

    plt.tight_layout()
    out_path = 'D:/song/kimi_DEMO/blue_calico_multitask/ablation_train_loss_combined.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved combined train loss curve to: {out_path}')


if __name__ == '__main__':
    plot_combined_train_loss()
