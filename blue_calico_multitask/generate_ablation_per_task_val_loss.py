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


def plot_per_task_val_loss():
    """Plot per-task validation loss curves for ablation variants."""
    variants = [
        ('Ours (full)', 'checkpoints_combined_fused_correct/history_cls.json', '#E41A1C', '-', 'o'),
        ('w/o Patch Transformer', 'checkpoints_combined_fused_correct_no_patchtransformer/history_cls.json', '#377EB8', '--', 's'),
        ('w/o ArcFace', 'checkpoints_combined_fused_correct_no_arcface/history_cls.json', '#4DAF4A', '-.', '^'),
        ('DINOv2-LoRA only', 'checkpoints_guizhou_emb_dinov2_lora_only_correct/history_cls.json', '#FF7F00', ':', 'D'),
    ]

    tasks = [
        ('Auth', 'auth', 'Auth Validation Loss'),
        ('Pattern', 'pattern', 'Pattern Validation Loss'),
        ('Defect', 'defect', 'Defect Validation Loss'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for col, (task_name, task_key, title) in enumerate(tasks):
        ax = axes[col]
        for name, path, color, linestyle, marker in variants:
            full_path = os.path.join('D:/song/kimi_DEMO/blue_calico_multitask', path)
            if not os.path.exists(full_path):
                continue
            with open(full_path, 'r', encoding='utf-8') as f:
                history = json.load(f)

            epochs = np.array([h['epoch'] for h in history])
            val_loss = np.array([h['val'][task_key] for h in history])
            val_smooth = smooth(val_loss, size=5)

            # Plot raw with low alpha
            ax.plot(epochs, val_loss, color=color, alpha=0.18, linewidth=1)
            # Plot smoothed
            ax.plot(epochs, val_smooth, color=color, linestyle=linestyle,
                    linewidth=2.2, label=name, marker=marker,
                    markevery=max(1, len(epochs) // 8), markersize=5,
                    markerfacecolor='white', markeredgewidth=1.2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)

    # Common legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=11,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle('Per-Task Validation Loss Curves of Ablation Variants', fontsize=15)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path = 'D:/song/kimi_DEMO/blue_calico_multitask/ablation_per_task_val_loss.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved per-task validation loss curves to: {out_path}')


if __name__ == '__main__':
    plot_per_task_val_loss()
