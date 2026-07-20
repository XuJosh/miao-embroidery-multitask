#!/usr/bin/env bash
# Stage 2: automatically start the enhanced model after the baseline 200-epoch run finishes.
# Then run explainability visualizations on both models.

set -e

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

BASELINE_DIR="checkpoints_guizhou_emb_dinov2_lora_seq"
ENHANCED_DIR="checkpoints_guizhou_emb_dinov2_lora_enhanced"
BASELINE_CKPT="$BASELINE_DIR/checkpoint_best.pt"
ENHANCED_CKPT="$ENHANCED_DIR/checkpoint_best.pt"
EPOCHS=200
VIS_DIR="visualizations"

mkdir -p "$ENHANCED_DIR" "$VIS_DIR"

# ---------------------------------------------------------------------------
# Wait for baseline model to finish 200 epochs
# ---------------------------------------------------------------------------
echo "[$(date)] Stage 2 watcher started. Waiting for baseline model to reach epoch $EPOCHS..."
while true; do
    EPOCH=0
    for CKPT in "$BASELINE_CKPT" "$BASELINE_DIR/checkpoint_last.pt"; do
        if [ -f "$CKPT" ]; then
            CUR=$(python - <<PY
import torch
try:
    ckpt = torch.load(r'$CKPT', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception as e:
    print(0)
PY
            )
            if [ "$CUR" -gt "$EPOCH" ]; then
                EPOCH=$CUR
            fi
        fi
    done
    if [ "$EPOCH" -ge "$EPOCHS" ]; then
        echo "[$(date)] Baseline model finished at epoch $EPOCH. Starting enhanced training."
        break
    fi
    echo "[$(date)] Baseline at epoch $EPOCH, still waiting..."
    sleep 60
done

# ---------------------------------------------------------------------------
# Train enhanced model (with auto-restart on crash)
# ---------------------------------------------------------------------------
while true; do
    RESUME=""
    if [ -f "$ENHANCED_CKPT" ]; then
        RESUME="$ENHANCED_CKPT"
    elif [ -f "$ENHANCED_DIR/checkpoint_last.pt" ]; then
        RESUME="$ENHANCED_DIR/checkpoint_last.pt"
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
            echo "[$(date)] Enhanced model already completed $EPOCH epochs."
            break
        fi
        echo "[$(date)] Resuming enhanced model from $RESUME (epoch $EPOCH)"
    else
        echo "[$(date)] Starting enhanced model from scratch"
    fi

    CMD="python -u train_enhanced.py \
        --data_dir data/guizhou_embroidery \
        --defect_mode cls \
        --epochs $EPOCHS \
        --batch_size 48 \
        --lr 1e-3 \
        --lora_lr 1e-4 \
        --input_size 224 \
        --device cuda \
        --save_dir $ENHANCED_DIR \
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
        --use_dynamic_loss"

    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi

    echo "[$(date)] Starting enhanced training: $CMD"
    set +e
    eval "$CMD" 2>&1 | tee -a "logs/train_enhanced_200.log"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Enhanced training exited cleanly."
        break
    fi

    echo "[$(date)] Enhanced training exited with code $EXIT_CODE. Restarting in 30 seconds..."
    sleep 30
done

# ---------------------------------------------------------------------------
# Generate explainability visualizations for both models
# ---------------------------------------------------------------------------
echo "[$(date)] Generating explainability visualizations..."

BASELINE_WEIGHTS="$BASELINE_DIR/embroidery_cls_best.pth"
ENHANCED_WEIGHTS="$ENHANCED_DIR/embroidery_cls_best.pth"

if [ -f "$BASELINE_WEIGHTS" ]; then
    python visualize_explain.py \
        --checkpoint "$BASELINE_WEIGHTS" \
        --model_type base \
        --split test \
        --num_images 30 \
        --save_dir "$VIS_DIR/base" \
        --device cuda
fi

if [ -f "$ENHANCED_WEIGHTS" ]; then
    python visualize_explain.py \
        --checkpoint "$ENHANCED_WEIGHTS" \
        --model_type enhanced \
        --split test \
        --num_images 30 \
        --save_dir "$VIS_DIR/enhanced" \
        --device cuda
fi

echo "[$(date)] Stage 2 complete. Results saved in $ENHANCED_DIR and $VIS_DIR."
