#!/usr/bin/env bash
# Train external mainstream baselines on the corrected dataset and evaluate all models.

set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

LOG_ALL="logs/run_external_comparison.log"
mkdir -p logs results

echo "[$(date)] ===== Starting external comparison experiments =====" | tee -a "$LOG_ALL"

for BACKBONE in resnet50 efficientnet_b0 vit_b_16 dinov2_vits14_frozen; do
    echo "[$(date)] Training external baseline: $BACKBONE ..." | tee -a "$LOG_ALL"
    bash run_external_baseline.sh "$BACKBONE"
done

echo "[$(date)] Evaluating all models including external baselines ..." | tee -a "$LOG_ALL"
python evaluate_models.py --config eval_config.json --split test --batch_size 64 --out_dir results

echo "[$(date)] ===== External comparison experiments completed =====" | tee -a "$LOG_ALL"
