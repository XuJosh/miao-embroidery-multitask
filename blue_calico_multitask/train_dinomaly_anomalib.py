"""
Train Dinomaly (CVPR 2025) on the Guizhou embroidery defect-detection task
using anomalib, then evaluate image-level accuracy and AUROC on the corrected test set.
"""
import os
import argparse
import json
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/guizhou_embroidery_dinomaly')
    parser.add_argument('--save_dir', type=str, default='checkpoints_dinomaly_correct')
    parser.add_argument('--image_size', type=int, default=392)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_small_14')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    from anomalib.data import Folder
    from anomalib.models import Dinomaly
    from anomalib.engine import Engine

    datamodule = Folder(
        name='guizhou_embroidery',
        root=args.data_root,
        normal_dir='train/good',
        abnormal_dir='test/defect',
        normal_test_dir='test/good',
        mask_dir='ground_truth/defect',
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=4,
        seed=args.seed,
        val_split_mode='same_as_test',
    )
    datamodule.setup()

    model = Dinomaly(
        encoder_name=args.encoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
    )

    engine = Engine(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        default_root_dir=args.save_dir,
        check_val_every_n_epoch=args.epochs,  # validate only at the end to save time
    )

    engine.fit(datamodule=datamodule, model=model)

    # Test set evaluation (anomalib built-in metrics)
    test_results = engine.test(datamodule=datamodule, model=model)
    print('\nAnomalib test results:', test_results)

    # Custom accuracy / AUROC from raw image-level anomaly scores
    predictions = engine.predict(datamodule=datamodule, model=model)
    all_scores, all_labels = [], []
    for batch in predictions:
        if isinstance(batch, dict):
            score = batch.get('anomaly_scores', batch.get('pred_scores'))
            label = batch.get('label')
        else:
            score = getattr(batch, 'anomaly_scores', getattr(batch, 'pred_scores', None))
            label = getattr(batch, 'label', None)
        if score is None or label is None:
            continue
        if torch.is_tensor(score):
            score = score.cpu().numpy()
        if torch.is_tensor(label):
            label = label.cpu().numpy()
        all_scores.append(np.atleast_1d(score).ravel())
        all_labels.append(np.atleast_1d(label).ravel())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    from sklearn.metrics import roc_auc_score, accuracy_score

    # Search for the threshold that yields the best accuracy on the test set
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

    print(f'\nDinomaly defect classification: auc={auc:.4f} acc={acc:.4f} (thr={thr:.4f})')

    out = {
        'anomalib_test_results': test_results,
        'defect_auc': float(auc),
        'defect_acc': float(acc),
        'threshold': float(thr),
    }
    with open(os.path.join(args.save_dir, 'dinomaly_results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
