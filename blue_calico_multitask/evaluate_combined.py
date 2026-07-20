"""
Evaluate EmbroideryNetCombined checkpoints on the test set.
Outputs accuracy and ROC-AUC for auth/defect, and pattern accuracy.
"""

import os
import argparse
import glob
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from dataset import BlueCalicoDataset, get_transforms
from model_combined import EmbroideryNetCombined


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_auth_true, all_auth_prob, all_auth_pred = [], [], []
    all_pat_true, all_pat_pred = [], []
    all_def_true, all_def_prob, all_def_pred = [], [], []

    for batch in tqdm(loader, desc='Test', leave=False):
        images = batch['image'].to(device, non_blocking=True)
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

    auth_true = np.concatenate(all_auth_true)
    auth_prob = np.concatenate(all_auth_prob)
    auth_pred = np.concatenate(all_auth_pred)
    pat_true = np.concatenate(all_pat_true)
    pat_pred = np.concatenate(all_pat_pred)
    def_true = np.concatenate(all_def_true)
    def_prob = np.concatenate(all_def_prob)
    def_pred = np.concatenate(all_def_pred)

    return {
        'auth_acc': float(accuracy_score(auth_true, auth_pred)),
        'auth_auc': float(roc_auc_score(auth_true, auth_prob)),
        'pat_acc': float(accuracy_score(pat_true, pat_pred)),
        'def_acc': float(accuracy_score(def_true, def_pred)),
        'def_auc': float(roc_auc_score(def_true, def_prob)),
    }


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        epoch = ckpt.get('epoch', None)
    else:
        state_dict = ckpt
        epoch = None
    # strip "module." if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/guizhou_embroidery_correct')
    parser.add_argument('--save_dir', type=str, default='checkpoints_combined_fused_correct')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoints', type=str, nargs='*', default=None,
                        help='Specific checkpoint paths; if omitted, evaluate best, last, and all embroidery_cls_epoch*.pth in save_dir')
    parser.add_argument('--output', type=str, default='combined_fused_test_results.json')
    parser.add_argument('--auth_defect_use_fused', action='store_true')
    parser.add_argument('--use_patch_transformer', action='store_true', default=True)
    parser.add_argument('--no_patch_transformer', dest='use_patch_transformer', action='store_false')
    parser.add_argument('--use_arcface', action='store_true', default=True)
    parser.add_argument('--no_arcface', dest='use_arcface', action='store_false')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = EmbroideryNetCombined(
        num_patterns=5,
        defect_mode='cls',
        lora_r=8,
        lora_alpha=16,
        seq_layers=2,
        seq_nhead=8,
        seq_dim_feedforward=512,
        dropout=0.3,
        use_frequency=False,
        use_arcface=args.use_arcface,
        auth_defect_use_fused=args.auth_defect_use_fused,
        use_patch_transformer=args.use_patch_transformer,
    ).to(device)

    test_ds = BlueCalicoDataset(args.data_dir, 'test', get_transforms(False, args.input_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    if args.checkpoints:
        paths = args.checkpoints
    else:
        paths = []
        for name in ('checkpoint_best.pt', 'checkpoint_last.pt'):
            p = os.path.join(args.save_dir, name)
            if os.path.exists(p):
                paths.append(p)
        paths += sorted(glob.glob(os.path.join(args.save_dir, 'embroidery_cls_epoch*.pth')))

    results = []
    for path in paths:
        print(f'\nEvaluating {path} ...')
        epoch = load_checkpoint(model, path, device)
        metrics = evaluate(model, test_loader, device)
        metrics['path'] = path
        metrics['epoch'] = epoch
        print(f"  epoch={epoch} auth_acc={metrics['auth_acc']:.3f} auth_auc={metrics['auth_auc']:.3f} "
              f"pat_acc={metrics['pat_acc']:.3f} def_acc={metrics['def_acc']:.3f} def_auc={metrics['def_auc']:.3f}")
        results.append(metrics)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nSaved results to {args.output}')

    if results:
        best_def = max(results, key=lambda x: x['def_acc'])
        print(f"Best defect accuracy: {best_def['def_acc']:.3f} @ {best_def['path']} (epoch={best_def['epoch']})")


if __name__ == '__main__':
    main()
