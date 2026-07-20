#!/usr/bin/env bash
# Train one ablation variant of the enhanced model on the corrected dataset.
# Usage: bash run_ablation_correct.sh <name> [extra_flags]
# Example: bash run_ablation_correct.sh no_freq --no_frequency

set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

NAME="${1:-ablation}"
shift || true
EXTRA_FLAGS="$@"

SAVE_DIR="checkpoints_enhanced_correct_${NAME}"
LOG_FILE="logs/train_enhanced_correct_${NAME}_150.log"
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
            echo "[$(date)] Ablation $NAME already completed $EPOCH epochs. Exiting."
            break
        fi
        echo "[$(date)] Attempt $attempt: resuming $NAME from $RESUME (epoch $EPOCH)"
    else
        echo "[$(date)] Attempt $attempt: starting $NAME from scratch"
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
        $EXTRA_FLAGS"

    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi

    echo "[$(date)] Starting: $CMD"
    set +e
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Ablation $NAME training exited cleanly."
        break
    fi

    echo "[$(date)] Ablation $NAME exited with code $EXIT_CODE. Restarting in 30 seconds..."
    sleep 30
done
