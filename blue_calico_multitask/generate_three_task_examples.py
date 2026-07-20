import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
from PIL import Image
import numpy as np
import os


INPUT_DIR = 'C:/Users/admin/Desktop/图片'
OUTPUT_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/result_visualization_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image file, class info, and probabilities from user's table
SAMPLES = [
    {
        'file': 'color_278.jpg',
        'cls_str': 'authentic/bx/no defect',
        'auth': 0.998,
        'pat': 0.999,
        'def': 0.947,
    },
    {
        'file': 'color_586.jpg',
        'cls_str': 'authentic/dx/no defect',
        'auth': 0.999,
        'pat': 1.0,
        'def': 0.917,
    },
    {
        'file': 'basic_96.jpg',
        'cls_str': 'authentic/mx/no defect',
        'auth': 1.0,
        'pat': 1.0,
        'def': 0.969,
    },
    {
        'file': 'color_30.jpg',
        'cls_str': 'authentic/ssx/no defect',
        'auth': 0.997,
        'pat': 1.0,
        'def': 0.903,
    },
    {
        'file': 'color_999.png',
        'cls_str': 'authentic/other/no defect',
        'auth': 0.969,
        'pat': 0.988,
        'def': 0.863,
    },
    {
        'file': 'test_00035.jpg',
        'cls_str': 'authentic/bx/defect',
        'auth': 0.998,
        'pat': 1.0,
        'def': 0.876,
    },
    {
        'file': 'test_00047.jpg',
        'cls_str': 'authentic/mx/defect',
        'auth': 0.999,
        'pat': 1.0,
        'def': 1.0,
    },
    {
        'file': 'taxd_002.png',
        'cls_str': 'fake/-/-',
        'auth': 0.993,
        'pat': None,
        'def': None,
    },
]

PATTERN_NAMES = {
    'bx': 'BX',
    'dx': 'DX',
    'mx': 'MX',
    'ssx': 'SSX',
    'other': 'Other',
}


def make_title(sample):
    parts = sample['cls_str'].split('/')
    auth_label = 'Authentic' if parts[0] == 'authentic' else 'Fake'
    if parts[0] == 'fake':
        return f'{auth_label}\nauth={sample["auth"]:.3f}'
    pattern = PATTERN_NAMES.get(parts[1], parts[1])
    defect_label = 'No Defect' if parts[2] == 'no defect' else 'Defect'
    return (
        f'{auth_label} / {pattern} / {defect_label}\n'
        f'auth={sample["auth"]:.3f}, pat={sample["pat"]:.3f}, def={sample["def"]:.3f}'
    )


def main():
    n_samples = len(SAMPLES)
    n_cols = 4
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axes = axes.flatten()

    for i, sample in enumerate(SAMPLES):
        img_path = os.path.join(INPUT_DIR, sample['file'])
        img = Image.open(img_path).convert('RGB')
        axes[i].imshow(img)
        axes[i].set_title(make_title(sample), fontsize=11)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')

    plt.suptitle('Recognition Results of the Proposed Method on Miao Embroidery Authenticity, Pattern and Defect Detection', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.45)

    save_path = os.path.join(OUTPUT_DIR, 'three_task_recognition_examples.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved figure to: {save_path}')


if __name__ == '__main__':
    main()
