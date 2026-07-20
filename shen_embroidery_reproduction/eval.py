"""
Evaluation script: load a trained model and report accuracy + confusion matrix.

Example:
    python eval.py --data_dir data/synthetic \
                   --model improved_mobilenet_v1 \
                   --checkpoint checkpoints/improved_mobilenet_v1_best.pth
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from model import create_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_test_loader(data_dir, batch_size=4, input_size=224, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_dir = os.path.join(data_dir, 'test')
    ds = datasets.ImageFolder(test_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True), ds.classes


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_probs.append(probs.cpu().numpy())
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return (avg_loss, acc,
            np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


def main():
    parser = argparse.ArgumentParser(description='Evaluate Shen Embroidery model')
    parser.add_argument('--data_dir', type=str, default='data/synthetic')
    parser.add_argument('--model', type=str, default='improved_mobilenet_v1',
                        choices=['improved_mobilenet_v1', 'mobilenet_v1',
                                 'alexnet', 'vgg16', 'resnet50', 'inception_v3'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    test_loader, classes = get_test_loader(
        args.data_dir, args.batch_size, args.input_size, args.workers)

    model = create_model(args.model, num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)

    loss, acc, probs, preds, labels = evaluate(model, test_loader, device)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.2f}%\n")

    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=classes, digits=4))

    # Show confidence for a few examples
    print("\nSample predictions (class, confidence):")
    for i in range(min(10, len(labels))):
        true_name = classes[int(labels[i])]
        pred_name = classes[int(preds[i])]
        conf = probs[i][int(preds[i])]
        print(f"  #{i}: true={true_name}, pred={pred_name}, conf={conf:.4f}")


if __name__ == '__main__':
    main()
