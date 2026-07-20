import re
import os
import json
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = 'logs/train_dinov2_lora_transformer_200.log'
OUT_DIR = 'results'


def parse_log(log_path):
    """Parse all epoch summary lines, keep the last occurrence of each epoch."""
    pattern = re.compile(
        r'Epoch \[(?P<epoch>\d{3})/200\] time=(?P<time>[\d.]+)s \| '
        r'train_total=(?P<train_total>[\d.]+) auth_acc=(?P<train_auth>[\d.]+) pat_acc=(?P<train_pat>[\d.]+) defect_acc=(?P<train_defect>[\d.]+) \| '
        r'val_total=(?P<val_total>[\d.]+) auth_acc=(?P<val_auth>[\d.]+) pat_acc=(?P<val_pat>[\d.]+) defect_acc=(?P<val_defect>[\d.]+)'
    )
    records = {}
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                d = m.groupdict()
                epoch = int(d['epoch'])
                records[epoch] = {k: float(v) for k, v in d.items() if k != 'epoch'}
                records[epoch]['time'] = float(d['time'])
    epochs = sorted(records.keys())
    data = [records[e] for e in epochs]
    return epochs, data


def plot_loss(epochs, data, out_path):
    train_total = [d['train_total'] for d in data]
    val_total = [d['val_total'] for d in data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_total, label='train_total', linewidth=1.5)
    plt.plot(epochs, val_total, label='val_total', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Baseline DINOv2+LoRA+Transformer (full log)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved loss curve to {out_path}')


def plot_metrics(epochs, data, out_path):
    metrics = [('auth', 'Auth'), ('pat', 'Pattern'), ('defect', 'Defect')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, title) in zip(axes, metrics):
        train_vals = [d[f'train_{metric}'] for d in data]
        val_vals = [d[f'val_{metric}'] for d in data]
        ax.plot(epochs, train_vals, label='train', linewidth=1.2)
        ax.plot(epochs, val_vals, label='val', linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle('Baseline DINOv2+LoRA+Transformer task metrics (full log)', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved metrics curve to {out_path}')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    epochs, data = parse_log(LOG_FILE)
    print(f'Parsed {len(epochs)} unique epochs (range {min(epochs)}-{max(epochs)})')

    plot_loss(epochs, data, os.path.join(OUT_DIR, 'loss_curve_baseline_full.png'))
    plot_metrics(epochs, data, os.path.join(OUT_DIR, 'metrics_curve_baseline_full.png'))

    # Save JSON for easy reuse
    summary = {
        'epochs': epochs,
        'records': data,
    }
    with open(os.path.join(OUT_DIR, 'baseline_full_log.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved JSON summary to', os.path.join(OUT_DIR, 'baseline_full_log.json'))
