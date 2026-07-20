#!/usr/bin/env bash
# Run ablation experiments on corrected dataset, then evaluate all models and generate visualizations.

set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

LOG_ALL="logs/run_ablation_and_visualize.log"
mkdir -p logs visualizations results

echo "[$(date)] ===== Starting ablation + visualization =====" | tee -a "$LOG_ALL"

# ---------------------------------------------------------------------------
# Ablation 1: no frequency branch
# ---------------------------------------------------------------------------
echo "[$(date)] Step 1/5: Enhanced without frequency branch (150 epochs) ..." | tee -a "$LOG_ALL"
bash run_ablation_correct.sh no_freq --no_frequency

# ---------------------------------------------------------------------------
# Ablation 2: no ArcFace
# ---------------------------------------------------------------------------
echo "[$(date)] Step 2/5: Enhanced without ArcFace (150 epochs) ..." | tee -a "$LOG_ALL"
bash run_ablation_correct.sh no_arcface --no_arcface

# ---------------------------------------------------------------------------
# Ablation 3: no dynamic loss
# ---------------------------------------------------------------------------
echo "[$(date)] Step 3/5: Enhanced without dynamic loss (150 epochs) ..." | tee -a "$LOG_ALL"
bash run_ablation_correct.sh no_dynamic --no_dynamic_loss

# ---------------------------------------------------------------------------
# Evaluate all available models
# ---------------------------------------------------------------------------
echo "[$(date)] Step 4/5: Evaluating all models on test set ..." | tee -a "$LOG_ALL"
python evaluate_models.py --config eval_config.json --split test --batch_size 64 --out_dir results

# ---------------------------------------------------------------------------
# Visualizations: baseline vs enhanced on corrected test set
# ---------------------------------------------------------------------------
echo "[$(date)] Step 5/5: Generating explainability visualizations ..." | tee -a "$LOG_ALL"

python visualize_explain.py \
    --checkpoint checkpoints_guizhou_emb_dinov2_lora_seq_correct/embroidery_cls_best.pth \
    --model_type base \
    --data_dir data/guizhou_embroidery_correct \
    --split test \
    --num_images 30 \
    --save_dir visualizations/correct_baseline \
    --device cuda

python visualize_explain.py \
    --checkpoint checkpoints_guizhou_emb_dinov2_lora_enhanced_correct/embroidery_cls_best.pth \
    --model_type enhanced \
    --data_dir data/guizhou_embroidery_correct \
    --split test \
    --num_images 30 \
    --save_dir visualizations/correct_enhanced \
    --device cuda

echo "[$(date)] ===== Ablation + visualization completed =====" | tee -a "$LOG_ALL"
