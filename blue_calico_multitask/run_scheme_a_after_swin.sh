#!/usr/bin/env bash
# Scheme A ablations after Swin-Tiny finishes:
#   1) Evaluate Swin-Tiny (lr=1e-4) on corrected test
#   2) ResNet50 external baseline without ArcFace (50 epochs)
#   3) Evaluate ResNet50 on corrected test
#   4) combined_fused_correct without ArcFace (50 epochs)
#   5) Evaluate combined_fused_correct (no ArcFace) on corrected test

set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

DATA_DIR="data/guizhou_embroidery_correct"
EPOCHS=50
MAX_ATTEMPTS=50

# ---------------------------------------------------------------------------
# 1) Swin-Tiny evaluation (rerun with lr=1e-4)
# ---------------------------------------------------------------------------
SWIN_DIR="checkpoints_external_swin_t_correct"
if [ -f "$SWIN_DIR/best_model.pth" ]; then
    echo "[$(date)] Evaluating Swin-Tiny (lr=1e-4) on corrected test..."
    python evaluate_external.py \
        --backbone swin_t \
        --data_dir $DATA_DIR \
        --split test \
        --checkpoint "$SWIN_DIR/best_model.pth" \
        --batch_size 64 \
        --device cuda \
        > "logs/eval_swin_t_correct_lr1e-4.log" 2>&1
    tail -n 20 "logs/eval_swin_t_correct_lr1e-4.log"
fi

# ---------------------------------------------------------------------------
# 2) ResNet50 external baseline (no ArcFace) -- external backbones never use ArcFace
# ---------------------------------------------------------------------------
SAVE_DIR="checkpoints_external_resnet50_correct_50epochs"
LOG_FILE="logs/train_external_resnet50_correct_50.log"
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
            echo "[$(date)] ResNet50 already completed $EPOCH epochs. Skipping training."
            break
        fi
        echo "[$(date)] Attempt $attempt: resuming ResNet50 from $RESUME (epoch $EPOCH)"
    else
        echo "[$(date)] Attempt $attempt: starting ResNet50 from scratch"
    fi

    CMD="python -u train_external.py \
        --backbone resnet50 \
        --data_dir $DATA_DIR \
        --epochs $EPOCHS \
        --batch_size 48 \
        --lr 1e-3 \
        --input_size 224 \
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
        echo "[$(date)] ResNet50 training exited cleanly."
        break
    fi

    echo "[$(date)] ResNet50 exited with code $EXIT_CODE. Restarting in 30 seconds..."
    sleep 30
done

# Evaluate ResNet50
if [ -f "$SAVE_DIR/best_model.pth" ]; then
    echo "[$(date)] Evaluating ResNet50 (50 epochs) on corrected test..."
    python evaluate_external.py \
        --backbone resnet50 \
        --data_dir $DATA_DIR \
        --split test \
        --checkpoint "$SAVE_DIR/best_model.pth" \
        --batch_size 64 \
        --device cuda \
        > "logs/eval_resnet50_correct_50.log" 2>&1
    tail -n 20 "logs/eval_resnet50_correct_50.log"
fi

# ---------------------------------------------------------------------------
# 3) combined_fused_correct without ArcFace
# ---------------------------------------------------------------------------
SAVE_DIR="checkpoints_combined_fused_correct_no_arcface"
LOG_FILE="logs/train_combined_fused_correct_no_arcface_50.log"
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
            echo "[$(date)] Combined fused (no ArcFace) already completed $EPOCH epochs. Skipping training."
            break
        fi
        echo "[$(date)] Attempt $attempt: resuming combined fused (no ArcFace) from $RESUME (epoch $EPOCH)"
    else
        echo "[$(date)] Attempt $attempt: starting combined fused (no ArcFace) from scratch"
    fi

    CMD="python -u train_combined.py \
        --data_dir $DATA_DIR \
        --epochs $EPOCHS \
        --batch_size 32 \
        --lr 1e-3 \
        --resnet_lr 1e-4 \
        --lora_lr 1e-4 \
        --input_size 224 \
        --device cuda \
        --save_dir $SAVE_DIR \
        --workers 4 \
        --lora_r 8 \
        --lora_alpha 16 \
        --seq_layers 2 \
        --seq_nhead 8 \
        --seq_dim_feedforward 512 \
        --dropout 0.3 \
        --no_arcface \
        --use_dynamic_loss \
        --auth_defect_use_fused"

    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi

    echo "[$(date)] Starting: $CMD"
    set +e
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Combined fused (no ArcFace) training exited cleanly."
        break
    fi

    echo "[$(date)] Combined fused (no ArcFace) exited with code $EXIT_CODE. Restarting in 30 seconds..."
    sleep 30
done

# Evaluate combined_fused_correct (no ArcFace)
if [ -f "$SAVE_DIR/best_model.pth" ]; then
    echo "[$(date)] Evaluating combined_fused_correct (no ArcFace) on corrected test..."
    python evaluate_combined.py \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR \
        --checkpoint "$SAVE_DIR/best_model.pth" \
        --batch_size 32 \
        --device cuda \
        --auth_defect_use_fused \
        --no_arcface \
        --output "combined_fused_correct_no_arcface_test_results.json" \
        > "logs/eval_combined_fused_correct_no_arcface.log" 2>&1
    tail -n 20 "logs/eval_combined_fused_correct_no_arcface.log"
fi

echo "[$(date)] All Scheme A ablations and evaluations finished."
