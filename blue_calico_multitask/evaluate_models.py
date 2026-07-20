"""
Unified evaluation script for trained models.
Produces per-task accuracy, confusion matrices, classification reports, and ROC-AUC.
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, roc_auc_score,
)

from dataset import BlueCalicoDataset, get_transforms
from model_dinov2_mamba import EmbroideryNet
from model_enhanced import EmbroideryNetEnhanced
from model_external import MultiTaskBackbone


def build_model(model_type, backbone='resnet50', use_frequency=True, use_arcface=True, use_dynamic_loss=True):
    common = {
        'num_patterns': 5,
        'defect_mode': 'cls',
        'seq_encoder': 'transformer',
        'seq_layers': 2,
        'seq_nhead': 8,
        'seq_dim_feedforward': 512,
        'dropout': 0.3,
        'dinov2_checkpoint': False,
    }
    if model_type == 'base':
        return EmbroideryNet(**common)
    if model_type == 'external':
        return MultiTaskBackbone(backbone=backbone, num_patterns=5, defect_mode='cls')
    return EmbroideryNetEnhanced(
        **common,
        use_frequency=use_frequency,
        use_arcface=use_arcface,
    )


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_auth_true, all_auth_prob, all_auth_pred = [], [], []
    all_pat_true, all_pat_pred = [], []
    all_def_true, all_def_prob, all_def_pred = [], [], []

    for batch in loader:
        images = batch['image'].to(device)
        auth = batch['auth'].cpu().numpy()
        pattern = batch['pattern'].cpu().numpy()
        defect_label = batch['defect_label'].cpu().numpy()

        with torch.cuda.amp.autocast():
            out = model(images)

        auth_logits = out['auth'].float().cpu()
        pat_logits = out['pattern'].float().cpu()
        def_logits = out['defect'].float().cpu()

        auth_prob = F.softmax(auth_logits, dim=1)[:, 1].numpy()
        auth_pred = auth_logits.argmax(dim=1).numpy()

        pat_pred = pat_logits.argmax(dim=1).numpy()

        def_prob = F.softmax(def_logits, dim=1)[:, 1].numpy()
        def_pred = def_logits.argmax(dim=1).numpy()

        all_auth_true.append(auth)
        all_auth_prob.append(auth_prob)
        all_auth_pred.append(auth_pred)

        all_pat_true.append(pattern)
        all_pat_pred.append(pat_pred)

        all_def_true.append(defect_label)
        all_def_prob.append(def_prob)
        all_def_pred.append(def_pred)

    results = {
        'auth_true': np.concatenate(all_auth_true),
        'auth_prob': np.concatenate(all_auth_prob),
        'auth_pred': np.concatenate(all_auth_pred),
        'pat_true': np.concatenate(all_pat_true),
        'pat_pred': np.concatenate(all_pat_pred),
        'def_true': np.concatenate(all_def_true),
        'def_prob': np.concatenate(all_def_prob),
        'def_pred': np.concatenate(all_def_pred),
    }
    return results


def compute_metrics(results):
    metrics = {}
    for task in ['auth', 'pat', 'def']:
        true = results[f'{task}_true']
        pred = results[f'{task}_pred']
        metrics[task] = {
            'accuracy': float(accuracy_score(true, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(true, pred)),
        }
        if task in ('auth', 'def'):
            prob = results[f'{task}_prob']
            try:
                metrics[task]['roc_auc'] = float(roc_auc_score(true, prob))
            except Exception as e:
                metrics[task]['roc_auc'] = None

    # Per-class metrics for pattern
    metrics['pattern_report'] = classification_report(
        results['pat_true'], results['pat_pred'], output_dict=True, zero_division=0
    )
    # Binary reports
    metrics['auth_report'] = classification_report(
        results['auth_true'], results['auth_pred'], output_dict=True, zero_division=0
    )
    metrics['defect_report'] = classification_report(
        results['def_true'], results['def_pred'], output_dict=True, zero_division=0
    )
    return metrics


def plot_confusion_matrices(results, name, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    tasks = [
        ('auth', 'Auth (0=fake, 1=real)'),
        ('pat', 'Pattern (0-4)'),
        ('def', 'Defect (0=clean, 1=defect)'),
    ]
    for ax, (task, title) in zip(axes, tasks):
        cm = confusion_matrix(results[f'{task}_true'], results[f'{task}_pred'])
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(title)
        ax.set_ylabel('True')
        ax.set_xlabel('Pred')
        # annotate
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        fig.colorbar(im, ax=ax)
    plt.suptitle(f'Confusion matrices: {name}', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'confusion_{name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='eval_config.json')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else (args.device if args.device != 'auto' else 'cpu'))
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    summary = {}
    for entry in config:
        name = entry['name']
        checkpoint = entry['checkpoint']
        if not os.path.exists(checkpoint):
            print(f'[SKIP] {name}: checkpoint not found {checkpoint}')
            continue

        print(f'\n=== Evaluating {name} ===')
        model_type = entry.get('model_type', 'base')
        backbone = entry.get('backbone', 'resnet50')
        use_freq = entry.get('use_frequency', True)
        use_arc = entry.get('use_arcface', True)

        model = build_model(model_type, backbone=backbone, use_frequency=use_freq, use_arcface=use_arc).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)

        ds = BlueCalicoDataset(entry['data_dir'], args.split, get_transforms(False, 224))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        results = evaluate_model(model, loader, device)
        metrics = compute_metrics(results)
        cm_path = plot_confusion_matrices(results, name, args.out_dir)

        summary[name] = {
            'checkpoint': checkpoint,
            'data_dir': entry['data_dir'],
            'metrics': metrics,
            'confusion_matrix_plot': cm_path,
        }

        print(f"  auth_acc={metrics['auth']['accuracy']:.3f} auc={metrics['auth'].get('roc_auc')}")
        print(f"  pat_acc={metrics['pat']['accuracy']:.3f}")
        print(f"  def_acc={metrics['def']['accuracy']:.3f} auc={metrics['def'].get('roc_auc')}")

    report_path = os.path.join(args.out_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nReport saved to {report_path}')


if __name__ == '__main__':
    main()
