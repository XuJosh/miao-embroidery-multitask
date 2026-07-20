import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_comparison_heatmap():
    """Plot a heatmap comparing metrics across different networks."""
    methods = [
        'EfficientNet-B0',
        'EfficientNet-B3',
        'ViT-B/16',
        'ResNet50-CBAM',
        'MobileNetV4-Conv-Small',
        'MambaOut',
        'Ours',
    ]
    metrics = ['Auth Acc', 'Auth AUC', 'Pat Acc', 'Def Acc', 'Def AUC']
    data = np.array([
        [0.928, 0.965, 0.973, 0.908, 0.948],  # EfficientNet-B0
        [0.938, 0.979, 0.996, 0.928, 0.950],  # EfficientNet-B3
        [0.640, 0.698, 0.893, 0.777, 0.693],  # ViT-B/16
        [0.952, 0.989, 0.965, 0.923, 0.962],  # ResNet50-CBAM
        [0.833, 0.894, 0.982, 0.875, 0.890],  # MobileNetV4-Conv-Small
        [0.951, 0.985, 0.993, 0.920, 0.960],  # MambaOut
        [0.969, 0.990, 0.996, 0.931, 0.948],  # Ours
    ])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(data, cmap='YlGn', aspect='auto', vmin=0.6, vmax=1.0)

    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticklabels(methods, fontsize=11)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics)):
            val = data[i, j]
            text_color = 'white' if val > 0.85 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=text_color, fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Score', fontsize=12)

    # Highlight Ours row
    ax.add_patch(plt.Rectangle((-0.5, len(methods)-1.5), len(metrics), 1,
                                 fill=False, edgecolor='red', linewidth=3))

    ax.set_title('Performance Heatmap of Comparison Networks', fontsize=15, pad=15)
    plt.tight_layout()

    out_path = 'D:/song/kimi_DEMO/blue_calico_multitask/comparison_performance_heatmap.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved comparison performance heatmap to: {out_path}')


if __name__ == '__main__':
    plot_comparison_heatmap()
