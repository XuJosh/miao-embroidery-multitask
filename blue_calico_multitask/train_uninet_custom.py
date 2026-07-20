"""
Custom training loop for UniNet (anomalib) to bypass Lightning overhead and
Windows dataloader issues.  Trains on a normal-only subset and evaluates on the
full anomaly-detection test set.
"""
import os
import argparse
import json
import random
import numpy as np
import torch
from torchvision.transforms.v2 import Compose, Resize, Normalize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_preprocessor(image_size):
    return Compose([
        Resize((image_size, image_size), antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/guizhou_embroidery_dinomaly')
    parser.add_argument('--save_dir', type=str, default='checkpoints_uninet_custom')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_subset', type=int, default=2000, help='0 = use all normal training images')
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_amp', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from anomalib.data import Folder
    from anomalib.models import UniNet

    transform = build_preprocessor(args.image_size)

    datamodule = Folder(
        name='guizhou_embroidery',
        root=args.data_root,
        normal_dir='train/good',
        abnormal_dir='test/defect',
        normal_test_dir='test/good',
        mask_dir='ground_truth/defect',
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        val_split_mode='same_as_test',
    )
    datamodule.setup()

    # Attach the same transform used by the model so dataloader returns preprocessed tensors.
    datamodule.train_data.augmentations = transform
    datamodule.val_data.augmentations = transform
    datamodule.test_data.augmentations = transform

    print(f'Train samples: {len(datamodule.train_data)}')
    if 0 < args.train_subset < len(datamodule.train_data):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(datamodule.train_data), size=args.train_subset, replace=False).tolist()
        datamodule.train_data = datamodule.train_data.subsample(indices)
        print(f'Using subset: {len(datamodule.train_data)}')

    model = UniNet(
        student_backbone='wide_resnet50_2',
        teacher_backbone='wide_resnet50_2',
        pre_processor=False,
        post_processor=True,
        evaluator=False,
        visualizer=False,
    ).to(device)

    # Same optimizer/scheduler as anomalib UniNet Lightning model.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.model.student.parameters()},
            {"params": model.model.bottleneck.parameters()},
            {"params": model.model.dfs.parameters()},
            {"params": model.model.teachers.target_teacher.parameters(), "lr": 1e-6},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
        eps=1e-10,
        amsgrad=True,
    )
    milestones = [int(args.epochs * 0.8)] if args.epochs > 1 else [args.epochs]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    use_amp = (not args.no_amp) and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f'Device: {device} | AMP: {use_amp} | image_size: {args.image_size} | batch: {args.batch_size}')

    train_loader = datamodule.train_dataloader()
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for batch_idx, batch in enumerate(train_loader):
            images = batch.image.to(device)
            labels = batch.gt_label.to(device).float()
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = model.model(images=images, labels=labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.model(images=images, labels=labels)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f'Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] loss={loss.item():.4f}')
        t1.record()
        torch.cuda.synchronize()
        print(f'Epoch {epoch} avg loss={np.mean(losses):.4f} time={t0.elapsed_time(t1)/1000:.1f}s')
        scheduler.step()

    # Evaluation on test set.
    print('Evaluating on test set...')
    model.eval()
    test_loader = datamodule.test_dataloader()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch.image.to(device)
            labels = batch.gt_label.to(device).long().cpu().numpy()
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model.model(images)
            else:
                out = model.model(images)
            scores = out.pred_score.cpu().numpy()
            all_scores.append(np.atleast_1d(scores).ravel())
            all_labels.append(np.atleast_1d(labels).ravel())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    from sklearn.metrics import roc_auc_score, accuracy_score

    def best_acc(y_true, y_prob):
        best = 0.0
        best_t = 0.5
        for t in np.linspace(y_prob.min(), y_prob.max(), 200):
            acc = accuracy_score(y_true, (y_prob >= t).astype(int))
            if acc > best:
                best = acc
                best_t = t
        return best, best_t

    auc = roc_auc_score(labels, scores)
    acc, thr = best_acc(labels, scores)
    print(f'\nUniNet defect classification: auc={auc:.4f} acc={acc:.4f} (thr={thr:.4f})')

    out = {
        'defect_auc': float(auc),
        'defect_acc': float(acc),
        'threshold': float(thr),
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'train_subset': len(datamodule.train_data),
        'lr': args.lr,
    }
    with open(os.path.join(args.save_dir, 'uninet_results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Save model state dict.
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'uninet_last.pth'))
    print(f'Results saved to {args.save_dir}')


if __name__ == '__main__':
    main()
