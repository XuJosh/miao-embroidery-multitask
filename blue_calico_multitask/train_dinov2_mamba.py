"""
Training script for EmbroideryNet (DINOv2 + LoRA + Vision Mamba).
"""

import os
import json
import argparse
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_dinov2_mamba import EmbroideryNet, MultiTaskLoss
from dataset import BlueCalicoDataset, get_transforms


def _build_targets(batch, device, defect_mode):
    targets = {
        'auth': batch['auth'].to(device),
        'pattern': batch['pattern'].to(device),
    }
    if defect_mode == 'seg':
        targets['defect'] = batch['defect'].to(device)
    else:
        targets['defect_label'] = batch['defect_label'].to(device)
    return targets


def _defect_metric(preds, targets, defect_mode):
    if defect_mode == 'seg':
        pred_mask = (torch.sigmoid(preds['defect']) > 0.5).float()
        inter = (pred_mask * targets['defect']).sum(dim=(1, 2, 3))
        union = ((pred_mask + targets['defect']) > 0).float().sum(dim=(1, 2, 3))
        return (inter / (union + 1e-6)).mean().item()
    else:
        return (preds['defect'].argmax(1) == targets['defect_label']).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, defect_mode, scaler=None, desc='Train'):
    model.train()
    metric_key = 'defect_iou' if defect_mode == 'seg' else 'defect_acc'
    metrics = {'total': 0.0, 'auth': 0.0, 'pattern': 0.0, 'defect': 0.0,
               'auth_acc': 0, 'pattern_acc': 0, metric_key: 0, 'n': 0}

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        targets = _build_targets(batch, device, defect_mode)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(images)
                losses = criterion(preds, targets)
            scaler.scale(losses['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            losses = criterion(preds, targets)
            losses['total'].backward()
            optimizer.step()

        b = images.size(0)
        metrics['total'] += losses['total'].item() * b
        metrics['auth'] += losses['auth'] * b
        metrics['pattern'] += losses['pattern'] * b
        metrics['defect'] += losses['defect'] * b
        metrics['n'] += b

        with torch.no_grad():
            metrics['auth_acc'] += (preds['auth'].argmax(1) == targets['auth']).float().mean().item() * b
            metrics['pattern_acc'] += (preds['pattern'].argmax(1) == targets['pattern']).float().mean().item() * b
            metrics[metric_key] += _defect_metric(preds, targets, defect_mode) * b

        pbar.set_postfix({
            'loss': f"{metrics['total']/metrics['n']:.3f}",
            'auth': f"{metrics['auth_acc']/metrics['n']:.3f}",
            'pat': f"{metrics['pattern_acc']/metrics['n']:.3f}",
            metric_key[:6]: f"{metrics[metric_key]/metrics['n']:.3f}",
        })

    for k in metrics:
        if k != 'n':
            metrics[k] /= metrics['n']
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, defect_mode, desc='Val'):
    model.eval()
    metric_key = 'defect_iou' if defect_mode == 'seg' else 'defect_acc'
    metrics = {'total': 0.0, 'auth': 0.0, 'pattern': 0.0, 'defect': 0.0,
               'auth_acc': 0, 'pattern_acc': 0, metric_key: 0, 'n': 0}

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        targets = _build_targets(batch, device, defect_mode)

        with torch.cuda.amp.autocast():
            preds = model(images)
            losses = criterion(preds, targets)

        b = images.size(0)
        metrics['total'] += losses['total'].item() * b
        metrics['auth'] += losses['auth'] * b
        metrics['pattern'] += losses['pattern'] * b
        metrics['defect'] += losses['defect'] * b
        metrics['n'] += b

        metrics['auth_acc'] += (preds['auth'].argmax(1) == targets['auth']).float().mean().item() * b
        metrics['pattern_acc'] += (preds['pattern'].argmax(1) == targets['pattern']).float().mean().item() * b
        metrics[metric_key] += _defect_metric(preds, targets, defect_mode) * b

        pbar.set_postfix({
            'loss': f"{metrics['total']/metrics['n']:.3f}",
            'auth': f"{metrics['auth_acc']/metrics['n']:.3f}",
            'pat': f"{metrics['pattern_acc']/metrics['n']:.3f}",
            metric_key[:6]: f"{metrics[metric_key]/metrics['n']:.3f}",
        })

    for k in metrics:
        if k != 'n':
            metrics[k] /= metrics['n']
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/guizhou_embroidery')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lora_lr', type=float, default=1e-4)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--defect_mode', type=str, default='cls', choices=['seg', 'cls'])
    parser.add_argument('--w_auth', type=float, default=1.0)
    parser.add_argument('--w_pattern', type=float, default=0.5)
    parser.add_argument('--w_defect', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints_guizhou_emb_dinov2_lora_seq')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--seq_encoder', type=str, default='transformer',
                        choices=['vim', 'gru', 'transformer'])
    parser.add_argument('--seq_layers', type=int, default=2)
    parser.add_argument('--seq_nhead', type=int, default=8)
    parser.add_argument('--seq_dim_feedforward', type=int, default=512)
    parser.add_argument('--vim_d_state', type=int, default=16)
    parser.add_argument('--vim_expand', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dinov2_checkpoint', action='store_true',
                        help='Enable DINOv2 gradient checkpointing (saves memory but slower)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a checkpoint (.pth weights or .pt full checkpoint)')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else (args.device if args.device != 'auto' else 'cpu'))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f'Using device: {device}')
    print(f'Defect learning mode: {args.defect_mode}')

    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = BlueCalicoDataset(args.data_dir, 'train', get_transforms(True, args.input_size))
    val_ds = BlueCalicoDataset(args.data_dir, 'val', get_transforms(False, args.input_size))
    test_ds = BlueCalicoDataset(args.data_dir, 'test', get_transforms(False, args.input_size))

    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True, 'persistent_workers': args.workers > 0}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    start_epoch = 1
    best_val_total = float('inf')
    history = []
    resume_optimizer_state = None

    common_model_kwargs = {
        'num_patterns': 5,
        'defect_mode': args.defect_mode,
        'dinov2_name': 'dinov2_vits14',
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'seq_encoder': args.seq_encoder,
        'seq_layers': args.seq_layers,
        'seq_nhead': args.seq_nhead,
        'seq_dim_feedforward': args.seq_dim_feedforward,
        'vim_d_state': args.vim_d_state,
        'vim_expand': args.vim_expand,
        'dropout': args.dropout,
        'dinov2_checkpoint': args.dinov2_checkpoint,
    }

    if args.resume and os.path.exists(args.resume):
        print(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = EmbroideryNet(**common_model_kwargs).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            resume_optimizer_state = checkpoint.get('optimizer_state_dict', None)
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_total = checkpoint.get('best_val_total', float('inf'))
            history = checkpoint.get('history', [])
            print(f'  -> resuming from epoch {start_epoch}, best_val_total={best_val_total:.4f}')
        else:
            # weights-only resume
            model = EmbroideryNet(**common_model_kwargs).to(device)
            model.load_state_dict(checkpoint)
            print('  -> loaded weights-only checkpoint, starting from epoch 1')
    else:
        model = EmbroideryNet(**common_model_kwargs).to(device)

    # Print LoRA trainable parameters
    try:
        model.dinov2.dinov2.print_trainable_parameters()
    except Exception:
        pass

    criterion = MultiTaskLoss(w_auth=args.w_auth, w_pattern=args.w_pattern,
                              w_defect=args.w_defect, defect_mode=args.defect_mode)

    # Parameter-grouped optimizer: LoRA uses smaller LR, new layers use larger LR
    lora_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_' in name:
                lora_params.append(param)
            else:
                other_params.append(param)

    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.lr},
        {'params': lora_params, 'lr': args.lora_lr},
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print('  -> optimizer state loaded')
        except Exception as e:
            print(f'  -> could not load optimizer state: {e}')
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    metric_key = 'defect_iou' if args.defect_mode == 'seg' else 'defect_acc'

    print('\nStarting training...')
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                  args.defect_mode, scaler=scaler,
                                  desc=f'Train epoch {epoch}/{args.epochs}')
        val_m = evaluate(model, val_loader, criterion, device, args.defect_mode,
                         desc=f'Val epoch {epoch}/{args.epochs}')
        scheduler.step()

        print(f"Epoch [{epoch:03d}/{args.epochs}] time={time.time()-t0:.1f}s | "
              f"train_total={train_m['total']:.4f} "
              f"auth_acc={train_m['auth_acc']:.3f} pat_acc={train_m['pattern_acc']:.3f} "
              f"{metric_key}={train_m[metric_key]:.3f} | "
              f"val_total={val_m['total']:.4f} "
              f"auth_acc={val_m['auth_acc']:.3f} pat_acc={val_m['pattern_acc']:.3f} "
              f"{metric_key}={val_m[metric_key]:.3f}")

        history.append({'epoch': epoch, 'train': train_m, 'val': val_m})

        # Save weights every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'embroidery_{args.defect_mode}_epoch{epoch}.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_total': best_val_total,
                'history': history,
            }, os.path.join(args.save_dir, 'checkpoint_last.pt'))

        if val_m['total'] < best_val_total:
            best_val_total = val_m['total']
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'embroidery_{args.defect_mode}_best.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_total': best_val_total,
                'history': history,
            }, os.path.join(args.save_dir, 'checkpoint_best.pt'))
            print(f"  -> New best model saved (val_total={best_val_total:.4f})")

    # Final test
    best_path = os.path.join(args.save_dir, f'embroidery_{args.defect_mode}_best.pth')
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_m = evaluate(model, test_loader, criterion, device, args.defect_mode, desc='Test')
    print(f"\nTest results: "
          f"auth_acc={test_m['auth_acc']:.3f} "
          f"pat_acc={test_m['pattern_acc']:.3f} "
          f"{metric_key}={test_m[metric_key]:.3f} "
          f"total_loss={test_m['total']:.4f}")

    with open(os.path.join(args.save_dir, f'history_{args.defect_mode}.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
