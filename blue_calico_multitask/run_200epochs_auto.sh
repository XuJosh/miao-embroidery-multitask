#!/usr/bin/env bash
# Auto-restarting 200-epoch training for DINOv2+LoRA+Transformer
# Resumes from the latest checkpoint if training crashes or is interrupted.

set -e

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

SAVE_DIR="checkpoints_guizhou_emb_dinov2_lora_seq"
LOG_FILE="logs/train_dinov2_lora_transformer_200.log"
EPOCHS=200

mkdir -p "$SAVE_DIR" logs

while true; do
    RESUME=""
    if [ -f "$SAVE_DIR/checkpoint_best.pt" ]; then
        RESUME="$SAVE_DIR/checkpoint_best.pt"
    elif [ -f "$SAVE_DIR/checkpoint_last.pt" ]; then
        RESUME="$SAVE_DIR/checkpoint_last.pt"
    fi

    # Check whether we already finished 200 epochs
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
            echo "[$(date)] Training already completed $EPOCH epochs. Exiting."
            break
        fi
        echo "[$(date)] Resuming from $RESUME (completed epoch $EPOCH)"
    else
        echo "[$(date)] No checkpoint found, starting fresh"
    fi

    CMD="python -u train_dinov2_mamba.py \
        --data_dir data/guizhou_embroidery \
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
        --dropout 0.3"

    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi

    echo "[$(date)] Starting: $CMD"
    set +e
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    echo "[$(date)] Training process exited with code $EXIT_CODE. Will restart in 10 seconds..."
    sleep 10
done
