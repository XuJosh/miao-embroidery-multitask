# Guizhou Embroidery Baseline Comparison (test set)

| Method | Reference | Config | Auth Acc | Pattern Acc | Defect Acc | Defect AUC | Best Defect Acc |
|--------|-----------|--------|----------|-------------|------------|------------|-----------------|
| ResNet50-CBAM | Wang et al., *npj Heritage Science*, 2026 | 30 ep, 224 px, batch 32 | 0.958 | 0.984 | **0.923** | 0.960 | 0.939 (@ ep 20) |
| MobileNetV4-Conv-Small | Ma et al., *Mathematics*, 2026 | 60 ep, 224 px, batch 32 | 0.834 | 0.970 | 0.865 | 0.886 | 0.875 (@ ep 45) |
| MambaOut-Small | Yu & Wang, CVPR 2025, pp. 4484-4496 | 50 ep, 224 px, batch 32, no AMP, grad clip 1.0 | 0.951 | **0.993** | 0.920 | 0.960 | 0.935 (@ ep 35) |
| MambaOut-Tiny | Yu & Wang, CVPR 2025, pp. 4484-4496 | 50 ep, 224 px, batch 32, no AMP, grad clip 1.0 | 0.928 | 0.981 | **0.936** | **0.971** | 0.936 (@ ep 50) |
| UniNet | Wei et al., CVPR 2025 | 5 ep, 224 px, batch 8, normal-only subset 2000, wide_resnet50_2 teacher/student | — | — | 0.723 | 0.587 | 0.723 (@ ep 5) |

**Notes**

- All supervised baselines are trained and evaluated on the corrected Guizhou embroidery multi-task split (train 34 330 / val 736 / test 736).
- UniNet is a one-class anomaly detector, so it is evaluated only on the defect task (auth/pattern heads are not applicable).
- UniNet used the default `wide_resnet50_2` teacher/student backbones. Training on the full 26 240 normal images with the anomalib `Engine` caused the Windows dataloader/Lightning loop to hang; a lightweight custom training loop was therefore used with a 2 000-image normal-only subset. This smaller training budget largely explains the lower defect performance.
- The best defect accuracy for each supervised method is taken from the per-epoch checkpoints saved every 5 epochs.
