#!/usr/bin/env bash
# Train one external multi-task baseline on the corrected dataset.
# Usage: bash run_external_baseline.sh <backbone>

set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

BACKBONE="${1:-resnet50}"
SAVE_DIR="checkpoints_external_${BACKBONE}_correct"
LOG_FILE="logs/train_external_${BACKBONE}_correct_150.log"
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
            echo "[$(date)] External $BACKBONE already completed $EPOCH epochs. Exiting."
            break
        fi
        echo "[$(date)] Attempt $attempt: resuming $BACKBONE from $RESUME (epoch $EPOCH)"
    else
        echo "[$(date)] Attempt $attempt: starting $BACKBONE from scratch"
    fi

    CMD="python -u train_external.py \
        --backbone $BACKBONE \
        --data_dir $DATA_DIR \
        --epochs $EPOCHS \
        --batch_size 48 \
        --lr 1e-3 \
        --device cuda \
        --save_dir $SAVE_DIR \
        --workers 4 \
        --dropout 0.3"

    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi

    echo "[$(date)] Starting: $CMD"
    set +e
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] External $BACKBONE training exited cleanly."
        break
    fi

    echo "[$(date)] External $BACKBONE exited with code $EXIT_CODE. Restarting in 30 seconds..."
    sleep 30
done
