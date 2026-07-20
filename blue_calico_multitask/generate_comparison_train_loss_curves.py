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


def plot_comparison_train_loss():
    """Plot train loss curves for comparison networks and Ours."""
    variants = [
        ('Ours', 'checkpoints_combined_fused_correct/history_cls.json', '#E41A1C', '-', 'o'),
        ('EfficientNet-B0', 'checkpoints_external_efficientnet_b0_correct/history_cls.json', '#377EB8', '--', 's'),
        ('EfficientNet-B3', 'checkpoints_external_efficientnet_b3_correct/history_cls.json', '#4DAF4A', '-.', '^'),
        ('ViT-B/16', 'checkpoints_external_vit_b_16_correct/history_cls.json', '#FF7F00', ':', 'D'),
        ('ResNet50-CBAM', 'checkpoints_external_resnet50_cbam_correct/history_cls.json', '#984EA3', '-', 'v'),
        ('MobileNetV4-Conv-Small', 'checkpoints_external_mobilenetv4_conv_small.e1200_r224_in1k_correct/history_cls.json', '#F781BF', '--', 'p'),
        ('MambaOut', 'checkpoints_external_mambaout_small.in1k_correct/history_cls.json', '#A65628', '-.', '*'),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for name, path, color, linestyle, marker in variants:
        full_path = os.path.join('D:/song/kimi_DEMO/blue_calico_multitask', path)
        if not os.path.exists(full_path):
            print(f'Skipping {name}: {full_path} not found')
            continue

        with open(full_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        epochs = np.array([h['epoch'] for h in history])
        train_total = np.array([h['train']['total'] for h in history])
        train_smooth = smooth(train_total, size=5)

        # Plot raw with low alpha
        ax.plot(epochs, train_total, color=color, alpha=0.15, linewidth=1)
        # Plot smoothed
        ax.plot(epochs, train_smooth, color=color, linestyle=linestyle,
                linewidth=2.2, label=name, marker=marker,
                markevery=max(1, len(epochs) // 8), markersize=5,
                markerfacecolor='white', markeredgewidth=1.2)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Training Loss Curves of Comparison Networks', fontsize=15)
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    out_path = 'D:/song/kimi_DEMO/blue_calico_multitask/comparison_train_loss_curves.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved comparison train loss curves to: {out_path}')


if __name__ == '__main__':
    plot_comparison_train_loss()
