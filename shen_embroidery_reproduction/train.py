"""
Training script for the Shen Embroidery CNN reproduction.

Usage examples:
    # Train the improved MobileNet V1 (with SPP) using synthetic data
    python train.py --data_dir data/synthetic --model improved_mobilenet_v1 \
                    --epochs 200 --batch_size 4 --lr 0.001

    # Train the baseline MobileNet V1
    python train.py --data_dir data/synthetic --model mobilenet_v1

    # Compare with ResNet50
    python train.py --data_dir data/synthetic --model resnet50
"""

import os
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from model import create_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(input_size=224, is_training=True):
    """Data augmentation used in the paper: flip, rotate, color variation."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_dataloaders(data_dir, batch_size=4, input_size=224, num_workers=0):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(input_size, True))
    val_ds = datasets.ImageFolder(val_dir, transform=get_transforms(input_size, False))
    test_ds = datasets.ImageFolder(test_dir, transform=get_transforms(input_size, False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Shen Embroidery CNN training')
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--model', type=str, default='improved_mobilenet_v1',
                        choices=['improved_mobilenet_v1', 'mobilenet_v1',
                                 'alexnet', 'vgg16', 'resnet50', 'inception_v3'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone (transfer learning)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Disable pretrained backbone')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone and only train classifier')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout before final classifier')
    parser.add_argument('--workers', type=int, default=0,
                        help='DataLoader num_workers (use 0 on Windows)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto/cpu/cuda')
    args = parser.parse_args()

    if args.no_pretrained:
        args.pretrained = False

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir, args.batch_size, args.input_size, args.workers)

    model = create_model(
        model_name=args.model,
        num_classes=len(classes),
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    model = model.to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print("Backbone frozen; only classifier will be trained.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

    best_val_acc = 0.0
    history = []

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - t0
        print(f"Epoch [{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
              f"lr={optimizer.param_groups[0]['lr']:.6f} | "
              f"time={epoch_time:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })

        # Save checkpoint every 5 epochs (as in the paper)
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.save_dir, f'{args.model}_epoch{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best val acc: {best_val_acc:.2f}% saved to {best_path}")

    # Final test evaluation
    best_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test loss: {test_loss:.4f}, test accuracy: {test_acc:.2f}%")

    # Save history and test results
    result = {
        'args': vars(args),
        'history': history,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'classes': classes,
    }
    result_path = os.path.join(args.save_dir, f'{args.model}_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == '__main__':
    main()
