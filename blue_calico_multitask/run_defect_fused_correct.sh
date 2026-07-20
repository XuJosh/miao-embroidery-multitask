#!/usr/bin/env bash
# Train the enhanced model on corrected data, but use the fused feature
# (instead of cnn_vec) as the input to the defect classification head.

set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

SAVE_DIR="checkpoints_enhanced_correct_defect_fused"
LOG_FILE="logs/train_enhanced_correct_defect_fused_150.log"
DATA_DIR="data/guizhou_embroidery_correct"
EPOCHS=150
MAX_ATTEMPTS=50

mkdir -p "$SAVE_DIR" logs

attempt=0
while true; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
        echo "[$(date)] Too many restart attempts ($MAX_ATTEMPTS). Giving up."
        exit 1
    fi

    RESUME=""
    if [ -f "$SAVE_DIR/checkpoint_best.pt" ]; then
        RESUME="$SAVE_DIR/checkpoint_best.pt"
    elif [ -f "$SAVE_DIR/checkpoint_last.pt" ]; then
        RESUME="$SAVE_DIR/checkpoint_last.pt"
    fi

    if [ -n "$RESUME" ]; then
        EPOCH=$(python - <<PY
import torch
try:
    ckpt = torch.load(r'$RESUME', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception as e:
    print(0)
PY
        )
        if [ "$EPOCH" -ge "$EPOCHS" ]; then
            echo "[$(date)] defect_fused already completed $EPOCH epochs. Exiting."
            break
        fi
        echo "[$(date)] Attempt $attempt: resuming defect_fused from $RESUME (epoch $EPOCH)"
    else
        echo "[$(date)] Attempt $attempt: starting defect_fused from scratch"
    fi

    CMD="python -u train_enhanced.py \
        --data_dir $DATA_DIR \
        --defect_mode cls \
        --epochs $EPOCHS \
        --batch_size 48 \
        --lr 1e-3 \
        --lora_lr 1e-4 \
        --input_size 224 \
        --device cuda \
        --save_dir $SAVE_DIR \
        --workers 4 \
        --lora_r 8 \
        --lora_alpha 16 \
        --seq_encoder transformer \
        --seq_layers 2 \
        --seq_nhead 8 \
        --seq_dim_feedforward 512 \
        --dropout 0.3 \
        --use_frequency \
        --use_arcface \
        --use_dynamic_loss \
        --defect_use_fused"

    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi

    echo "[$(date)] Starting: $CMD"
    set +e
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] defect_fused training exited cleanly."
        break
    fi

    echo "[$(date)] defect_fused exited with code $EXIT_CODE. Restarting in 30 seconds..."
    sleep 30
done
