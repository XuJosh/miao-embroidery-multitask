#!/usr/bin/env bash
# Train EfficientNet-B3/B4, ConvNeXt-Tiny, Swin-Tiny for 50 epochs each on corrected data.
set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

DATA_DIR="data/guizhou_embroidery_correct"
EPOCHS=50
BATCH_SIZE=32
LR=1e-3
WORKERS=4
DROPOUT=0.3
LOG_ALL="logs/run_external_50epochs.log"

mkdir -p logs

echo "[$(date)] ===== Starting 50-epoch external comparisons =====" | tee -a "$LOG_ALL"

for BACKBONE in efficientnet_b3 efficientnet_b4 convnext_tiny swin_t; do
    SAVE_DIR="checkpoints_external_${BACKBONE}_correct"
    LOG_FILE="logs/train_external_${BACKBONE}_correct_50.log"
    mkdir -p "$SAVE_DIR"

    echo "[$(date)] Training $BACKBONE (50 epochs, batch=$BATCH_SIZE) ..." | tee -a "$LOG_ALL"

    python -u train_external.py \
        --backbone "$BACKBONE" \
        --data_dir "$DATA_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --device cuda \
        --save_dir "$SAVE_DIR" \
        --workers "$WORKERS" \
        --dropout "$DROPOUT" \
        2>&1 | tee -a "$LOG_FILE"

    echo "[$(date)] $BACKBONE training finished." | tee -a "$LOG_ALL"
done

echo "[$(date)] ===== Training completed; evaluating all checkpoints =====" | tee -a "$LOG_ALL"

for BACKBONE in efficientnet_b3 efficientnet_b4 convnext_tiny swin_t; do
    SAVE_DIR="checkpoints_external_${BACKBONE}_correct"
    echo "[$(date)] Evaluating $BACKBONE checkpoints on test set ..." | tee -a "$LOG_ALL"
    python -u evaluate_external_checkpoints.py \
        --backbone "$BACKBONE" \
        --save_dir "$SAVE_DIR" \
        --data_dir "$DATA_DIR" \
        --batch_size 64 \
        --split test \
        2>&1 | tee -a "logs/eval_external_${BACKBONE}_correct_50.log"
done

echo "[$(date)] ===== All done =====" | tee -a "$LOG_ALL"
