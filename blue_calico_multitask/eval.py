"""
Evaluation + Grad-CAM visualization for BlueCalicoNet.
Supports both segmentation and image-level defect mode.
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from model import BlueCalicoNet
from dataset import BlueCalicoDataset, get_transforms
from gradcam import generate_gradcam, save_cam_overlay


def evaluate(model, loader, device, defect_mode):
    model.eval()
    auth_preds, auth_labels = [], []
    pat_preds, pat_labels = [], []
    defect_preds, defect_labels = [], []
    ious = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            out = model(images)

            auth_preds.extend(out['auth'].argmax(1).cpu().tolist())
            auth_labels.extend(batch['auth'].tolist())
            pat_preds.extend(out['pattern'].argmax(1).cpu().tolist())
            pat_labels.extend(batch['pattern'].tolist())

            if defect_mode == 'seg':
                pred_mask = (torch.sigmoid(out['defect']) > 0.5).float()
                gt_mask = batch['defect'].to(device)
                inter = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
                union = ((pred_mask + gt_mask) > 0).float().sum(dim=(1, 2, 3))
                iou = (inter / (union + 1e-6)).cpu().numpy()
                ious.append(iou)
            else:
                defect_preds.extend(out['defect'].argmax(1).cpu().tolist())
                defect_labels.extend(batch['defect_label'].tolist())

    auth_acc = 100.0 * np.mean(np.array(auth_preds) == np.array(auth_labels))
    pat_acc = 100.0 * np.mean(np.array(pat_preds) == np.array(pat_labels))

    if defect_mode == 'seg':
        mean_defect = np.concatenate(ious).mean()
        return auth_acc, pat_acc, mean_defect, auth_preds, auth_labels, pat_preds, pat_labels, None, None
    else:
        def_acc = 100.0 * np.mean(np.array(defect_preds) == np.array(defect_labels))
        return auth_acc, pat_acc, def_acc, auth_preds, auth_labels, pat_preds, pat_labels, defect_preds, defect_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/blue_calico')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--defect_mode', type=str, default='seg', choices=['seg', 'cls'])
    parser.add_argument('--save_dir', type=str, default='cam_outputs')
    parser.add_argument('--num_cam', type=int, default=8,
                        help='Number of Grad-CAM images to generate')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    test_ds = BlueCalicoDataset(args.data_dir, 'test', get_transforms(False))
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    model = BlueCalicoNet(num_patterns=5, defect_mode=args.defect_mode).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    (auth_acc, pat_acc, defect_score, ap, al, pp, pl,
     defect_preds, defect_labels) = evaluate(model, test_loader, device, args.defect_mode)

    print(f"Test authenticity accuracy: {auth_acc:.2f}%")
    print(f"Test pattern accuracy: {pat_acc:.2f}%")
    if args.defect_mode == 'seg':
        print(f"Test defect mean IoU: {defect_score:.4f}\n")
    else:
        print(f"Test defect classification accuracy: {defect_score:.2f}%\n")

    print("Authenticity confusion matrix:")
    print(confusion_matrix(al, ap))
    print("\nAuthenticity report:")
    print(classification_report(al, ap, target_names=['fake', 'real'], digits=4))

    print("\nPattern confusion matrix:")
    print(confusion_matrix(pl, pp))
    print("\nPattern report:")
    print(classification_report(pl, pp, target_names=['fish','flower','geometry','cloud','phoenix'], digits=4))

    if args.defect_mode == 'cls':
        print("\nDefect classification confusion matrix:")
        print(confusion_matrix(defect_labels, defect_preds))
        print("\nDefect classification report:")
        print(classification_report(defect_labels, defect_preds, target_names=['no_defect','defect'], digits=4))

    # Grad-CAM
    os.makedirs(args.save_dir, exist_ok=True)
    count = 0
    for batch in test_loader:
        images = batch['image'].to(device)
        for i in range(images.size(0)):
            if count >= args.num_cam:
                break
            img = images[i:i+1]
            heatmap, pred_class = generate_gradcam(model, img)
            label_str = 'real' if batch['auth'][i].item() == 1 else 'fake'
            pred_str = 'real' if pred_class == 1 else 'fake'
            fname = os.path.join(args.save_dir,
                                 f"cam_{count}_true{label_str}_pred{pred_str}.png")
            save_cam_overlay(images[i], heatmap, fname)
            count += 1
        if count >= args.num_cam:
            break
    print(f"\nGrad-CAM images saved to {args.save_dir}")


if __name__ == '__main__':
    main()
