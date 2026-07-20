"""
Train external multi-task baseline models (ResNet, EfficientNet, ViT, frozen DINOv2).
"""

import os
import json
import argparse
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_external import MultiTaskBackbone
from model_dinov2_mamba import MultiTaskLoss
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


def train_one_epoch(model, loader, criterion, optimizer, device, defect_mode, scaler=None, grad_clip=1.0, desc='Train'):
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
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            losses = criterion(preds, targets)
            losses['total'].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    parser.add_argument('--backbone', type=str, required=True,
                        choices=['resnet50', 'resnet50_cbam',
                                 'mobilenetv4_conv_small.e1200_r224_in1k',
                                 'ghostnetv3_100.in1k',
                                 'mambaout_small.in1k',
                                 'mambaout_tiny.in1k',
                                 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4',
                                 'convnext_tiny', 'swin_t', 'vit_b_16', 'dinov2_vits14_frozen',
                                 'dinov2_vits14_lora'])
    parser.add_argument('--data_dir', type=str, default='data/guizhou_embroidery_correct')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--defect_mode', type=str, default='cls', choices=['seg', 'cls'])
    parser.add_argument('--w_auth', type=float, default=1.0)
    parser.add_argument('--w_pattern', type=float, default=0.5)
    parser.add_argument('--w_defect', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else (args.device if args.device != 'auto' else 'cpu'))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if args.save_dir is None:
        args.save_dir = f'checkpoints_external_{args.backbone}_correct'
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'Using device: {device}')
    print(f'Backbone: {args.backbone}')
    print(f'Save dir: {args.save_dir}')
    print(f'AMP: {not args.no_amp and device.type == "cuda"} | grad_clip: {args.grad_clip}')

    train_ds = BlueCalicoDataset(args.data_dir, 'train', get_transforms(True, args.input_size))
    val_ds = BlueCalicoDataset(args.data_dir, 'val', get_transforms(False, args.input_size))
    test_ds = BlueCalicoDataset(args.data_dir, 'test', get_transforms(False, args.input_size))

    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True, 'persistent_workers': args.workers > 0}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    start_epoch = 1
    best_val_total = float('inf')
    history = []
    resume_optimizer_state = None

    model = MultiTaskBackbone(backbone=args.backbone, num_patterns=5,
                              defect_mode=args.defect_mode, dropout=args.dropout).to(device)

    if args.resume and os.path.exists(args.resume):
        print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            resume_optimizer_state = ckpt.get('optimizer_state_dict', None)
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_total = ckpt.get('best_val_total', float('inf'))
            history = ckpt.get('history', [])
            print(f'  -> resuming from epoch {start_epoch}, best_val_total={best_val_total:.4f}')
        else:
            model.load_state_dict(ckpt)
            print('  -> loaded weights-only checkpoint')

    criterion = MultiTaskLoss(w_auth=args.w_auth, w_pattern=args.w_pattern,
                              w_defect=args.w_defect, defect_mode=args.defect_mode)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print('  -> optimizer state loaded')
        except Exception as e:
            print(f'  -> could not load optimizer state: {e}')
    scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda' and not args.no_amp) else None

    metric_key = 'defect_iou' if args.defect_mode == 'seg' else 'defect_acc'

    print('\nStarting training...')
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                  args.defect_mode, scaler=scaler, grad_clip=args.grad_clip,
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
