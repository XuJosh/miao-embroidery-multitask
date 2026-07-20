import os
import re
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUTPUT_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/gradcam_comparison_outputs'
BATCH_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/gradcam_comparison_batch'
NET_NAMES = [
    'Ours',
    'ResNet50-CBAM',
    'EfficientNet-B3',
    'EfficientNet-B0',
    'MobileNetV4-Conv-Small',
    'MambaOut',
    'ViT-B16',
]


def parse_sample_index(filename):
    """Extract sample index from filename like sample_002_pred_1.png."""
    m = re.search(r'sample_(\d+)_pred', filename)
    return int(m.group(1)) if m else None


def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img) / 255.0


def get_common_indices(net_names, batch_dir):
    """Return sample indices that exist for every network."""
    index_sets = []
    for net in net_names:
        net_dir = os.path.join(batch_dir, net)
        files = [f for f in os.listdir(net_dir) if f.endswith('.png')]
        indices = set(parse_sample_index(f) for f in files if parse_sample_index(f) is not None)
        index_sets.append(indices)
    common = sorted(set.intersection(*index_sets))
    return common


def build_grid(net_names, batch_dir, selected_indices, save_path,
               title='Grad-CAM Visualization Comparison across Networks (Real Samples)'):
    """Build a grid: rows=samples, cols=each network."""
    n_rows = len(selected_indices)
    n_cols = len(net_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(selected_indices):
        for j, net in enumerate(net_names):
            img_path = os.path.join(batch_dir, net, f'sample_{idx:03d}_pred_1.png')
            if os.path.exists(img_path):
                net_img = load_image(img_path)
            else:
                # fallback: any pred value
                net_dir = os.path.join(batch_dir, net)
                candidates = [f for f in os.listdir(net_dir) if f.startswith(f'sample_{idx:03d}_pred')]
                net_img = load_image(os.path.join(net_dir, candidates[0])) if candidates else np.ones((224, 224, 3))
            axes[i, j].imshow(net_img)
            title = net.replace('ViT-B16', 'ViT-B/16') if i == 0 else ''
            axes[i, j].set_title(title, fontsize=12)
            axes[i, j].axis('off')

    plt.suptitle(title, fontsize=15, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved grid to: {save_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    common = get_common_indices(NET_NAMES, BATCH_DIR)
    print(f'Common sample indices across {len(NET_NAMES)} networks: {len(common)}')
    print(common[:20])

    # Default: first 6 samples for preview
    selected = common[:6]
    print(f'Building grid for samples: {selected}')

    save_path = os.path.join(OUTPUT_DIR, 'gradcam_comparison_grid_7nets.png')
    build_grid(NET_NAMES, BATCH_DIR, selected, save_path)


if __name__ == '__main__':
    main()
