"""
Weakly-supervised defect localization using a defect-classification model + Grad-CAM.

The model is trained only with image-level labels (defect_mode=cls).
At inference, Grad-CAM on the "defect" head produces a coarse localization map,
which is thresholded to obtain a pixel-level defect mask.
"""

import os
import argparse

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from model import BlueCalicoNet
from dataset import BlueCalicoDataset, get_transforms
from gradcam import generate_gradcam, save_cam_overlay, cam_to_mask


def compute_mask_metrics(pred_mask, gt_mask):
    """pred_mask, gt_mask: numpy arrays of shape (H, W) with values 0/1."""
    pred = pred_mask.astype(np.bool_)
    gt = gt_mask.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / (union + 1e-8)
    precision = inter / (pred.sum() + 1e-8)
    recall = inter / (gt.sum() + 1e-8)
    pixel_acc = (pred == gt).mean()
    return iou, precision, recall, pixel_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/blue_calico')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='localization_outputs')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for converting CAM to binary mask')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of defective images to visualize')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)
    mask_dir = os.path.join(args.save_dir, 'masks')
    overlay_dir = os.path.join(args.save_dir, 'overlays')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    test_ds = BlueCalicoDataset(args.data_dir, 'test', get_transforms(False))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = BlueCalicoNet(num_patterns=5, defect_mode='cls').to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    metrics = []
    vis_count = 0

    print('\nRunning weakly-supervised defect localization...')
    for idx, batch in enumerate(test_loader):
        image = batch['image'].to(device)
        with torch.no_grad():
            out = model(image)
        defect_pred = out['defect'].argmax(dim=1).item()
        defect_prob = torch.softmax(out['defect'], dim=1)[0, 1].item()

        # Ground-truth mask if available
        gt_mask = batch['defect'][0, 0].cpu().numpy().astype(np.uint8)

        # Localize ground-truth defective images (or predicted defective if no GT)
        has_gt_defect = gt_mask.sum() > 0
        if not (has_gt_defect or defect_pred == 1):
            continue

        # Grad-CAM targeting the defect head, class 1 (defect)
        cam, _ = generate_gradcam(model, image, head='defect', target_class=1)
        pred_mask = cam_to_mask(cam, threshold=args.threshold)

        if has_gt_defect:
            iou, prec, rec, pacc = compute_mask_metrics(pred_mask, gt_mask)
            metrics.append((iou, prec, rec, pacc))

        # Save visualizations
        if vis_count < args.num_vis:
            name = batch['name'][0]
            overlay_path = os.path.join(overlay_dir, f'{name}_cam.png')
            save_cam_overlay(image[0], cam, overlay_path)

            pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            pred_mask_img.save(os.path.join(mask_dir, f'{name}_pred.png'))

            gt_mask_img = Image.fromarray((gt_mask * 255).astype(np.uint8))
            gt_mask_img.save(os.path.join(mask_dir, f'{name}_gt.png'))
            vis_count += 1

    if metrics:
        metrics = np.array(metrics)
        print(f"\nLocalization metrics over {len(metrics)} defective images:")
        print(f"  Mean IoU:      {metrics[:,0].mean():.4f}")
        print(f"  Mean Precision:{metrics[:,1].mean():.4f}")
        print(f"  Mean Recall:   {metrics[:,2].mean():.4f}")
        print(f"  Mean Pixel Acc:{metrics[:,3].mean():.4f}")
    else:
        print("\nNo ground-truth defect masks available for metric computation.")

    print(f"\nSaved {vis_count} localization visualizations to {args.save_dir}")


if __name__ == '__main__':
    main()
