import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, 'D:/song/kimi_DEMO/blue_calico_multitask')

from dataset import BlueCalicoDataset, get_transforms
from model_combined import EmbroideryNetCombined


DATA_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/data/guizhou_embroidery_correct'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32


def load_model(ckpt_path, use_arcface=True, use_patch_transformer=True, auth_defect_use_fused=True):
    model = EmbroideryNetCombined(
        num_patterns=5,
        defect_mode='cls',
        use_patch_transformer=use_patch_transformer,
        use_arcface=use_arcface,
        auth_defect_use_fused=auth_defect_use_fused,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, loader):
    auth_true, auth_pred = [], []
    pat_true, pat_pred = [], []
    def_true, def_pred = [], []

    for batch in loader:
        images = batch['image'].to(DEVICE)
        labels = batch['labels'] if 'labels' in batch else batch
        out = model(images)

        auth_true.extend(labels['auth'].cpu().numpy())
        auth_pred.extend(out['auth'].argmax(1).cpu().numpy())
        pat_true.extend(labels['pattern'].cpu().numpy())
        pat_pred.extend(out['pattern'].argmax(1).cpu().numpy())
        def_true.extend(labels['defect_label'].cpu().numpy())
        def_pred.extend(out['defect'].argmax(1).cpu().numpy())

    return (np.array(auth_true), np.array(auth_pred),
            np.array(pat_true), np.array(pat_pred),
            np.array(def_true), np.array(def_pred))


def plot_confusion_matrix(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.Greys
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14, fontweight='bold')

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('True label', fontsize=12)
    ax.set_ylabel('Predicted label', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved confusion matrix to: {out_path}')


def build_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[p, t] += 1
    return cm


def main():
    test_ds = BlueCalicoDataset(DATA_DIR, 'test', get_transforms(False, 224))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    variants = [
        {
            'name': 'Ours (full)',
            'ckpt': 'D:/song/kimi_DEMO/blue_calico_multitask/checkpoints_combined_fused_correct/embroidery_cls_best.pth',
            'use_arcface': True,
            'use_patch_transformer': True,
            'auth_defect_use_fused': True,
        },
        {
            'name': 'w/o Patch Transformer',
            'ckpt': 'D:/song/kimi_DEMO/blue_calico_multitask/checkpoints_combined_fused_correct_no_patchtransformer/embroidery_cls_best.pth',
            'use_arcface': True,
            'use_patch_transformer': False,
            'auth_defect_use_fused': True,
        },
        {
            'name': 'w/o ArcFace',
            'ckpt': 'D:/song/kimi_DEMO/blue_calico_multitask/checkpoints_combined_fused_correct_no_arcface/embroidery_cls_best.pth',
            'use_arcface': False,
            'use_patch_transformer': True,
            'auth_defect_use_fused': True,
        },
    ]

    auth_classes = ['fake', 'real']
    defect_classes = ['no defect', 'defect']
    pattern_classes = ['bx', 'dx', 'mx', 'other', 'ssx']

    output_dir = 'D:/song/kimi_DEMO/blue_calico_multitask/ablation_confusion_matrices'
    os.makedirs(output_dir, exist_ok=True)

    results_summary = []

    for v in variants:
        print(f'\nEvaluating {v["name"]} ...')
        model = load_model(v['ckpt'], v['use_arcface'], v['use_patch_transformer'], v.get('auth_defect_use_fused', True))
        auth_true, auth_pred, pat_true, pat_pred, def_true, def_pred = get_predictions(model, test_loader)

        # Auth confusion matrix
        cm_auth = build_confusion_matrix(auth_true, auth_pred, 2)
        plot_confusion_matrix(cm_auth, auth_classes, f'{v["name"]} - Authenticity',
                              os.path.join(output_dir, f'{v["name"].replace(" ", "_").replace("/", "_")}_auth_cm.png'))

        # Defect confusion matrix
        cm_def = build_confusion_matrix(def_true, def_pred, 2)
        plot_confusion_matrix(cm_def, defect_classes, f'{v["name"]} - Defect',
                              os.path.join(output_dir, f'{v["name"].replace(" ", "_").replace("/", "_")}_defect_cm.png'))

        # Pattern confusion matrix
        cm_pat = build_confusion_matrix(pat_true, pat_pred, 5)
        plot_confusion_matrix(cm_pat, pattern_classes, f'{v["name"]} - Pattern',
                              os.path.join(output_dir, f'{v["name"].replace(" ", "_").replace("/", "_")}_pattern_cm.png'))

        # Compute accuracy
        auth_acc = (auth_true == auth_pred).mean()
        pat_acc = (pat_true == pat_pred).mean()
        def_acc = (def_true == def_pred).mean()
        results_summary.append({
            'name': v['name'],
            'auth_acc': float(auth_acc),
            'pat_acc': float(pat_acc),
            'def_acc': float(def_acc),
        })
        print(f'  auth_acc={auth_acc:.3f} pat_acc={pat_acc:.3f} def_acc={def_acc:.3f}')

    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f'\nSaved summary to {os.path.join(output_dir, "summary.json")}')


if __name__ == '__main__':
    main()
