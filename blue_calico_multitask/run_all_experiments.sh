#!/usr/bin/env bash
# Master script to run the full experiment sequence automatically:
#   1. Preserve the aug-first dataset and baseline model.
#   2. Train enhanced model on aug-first dataset (150 epochs).
#   3. Create corrected-split dataset (split originals first, augment train only).
#   4. Train baseline model on corrected dataset (150 epochs).
#   5. Train enhanced model on corrected dataset (150 epochs).

set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

LOG_ALL="logs/run_all_experiments.log"
mkdir -p logs

echo "[$(date)] ===== Starting all experiments =====" | tee -a "$LOG_ALL"

# ---------------------------------------------------------------------------
# Preserve aug-first assets with explicit names
# ---------------------------------------------------------------------------
echo "[$(date)] Preserving aug-first dataset to data/guizhou_embroidery_augfirst ..." | tee -a "$LOG_ALL"
rm -rf data/guizhou_embroidery_augfirst
cp -r data/guizhou_embroidery data/guizhou_embroidery_augfirst

echo "[$(date)] Preserving aug-first baseline model to checkpoints_guizhou_emb_dinov2_lora_seq_augfirst ..." | tee -a "$LOG_ALL"
rm -rf checkpoints_guizhou_emb_dinov2_lora_seq_augfirst
cp -r checkpoints_guizhou_emb_dinov2_lora_seq checkpoints_guizhou_emb_dinov2_lora_seq_augfirst

# ---------------------------------------------------------------------------
# Step 1: Enhanced model on aug-first dataset
# ---------------------------------------------------------------------------
echo "[$(date)] Step 1/4: Training enhanced model on aug-first dataset (150 epochs) ..." | tee -a "$LOG_ALL"
bash run_enhanced_150_augfirst.sh

# ---------------------------------------------------------------------------
# Step 2: Create corrected-split dataset
# ---------------------------------------------------------------------------
echo "[$(date)] Step 2/4: Creating corrected-split dataset ..." | tee -a "$LOG_ALL"
python convert_guizhou_embroidery_correct.py \
    --src "D:\song\newdata_en" \
    --dst data/guizhou_embroidery_correct

# ---------------------------------------------------------------------------
# Step 3: Baseline model on corrected dataset
# ---------------------------------------------------------------------------
echo "[$(date)] Step 3/4: Training baseline model on corrected dataset (150 epochs) ..." | tee -a "$LOG_ALL"
bash run_baseline_150_correct.sh

# ---------------------------------------------------------------------------
# Step 4: Enhanced model on corrected dataset
# ---------------------------------------------------------------------------
echo "[$(date)] Step 4/4: Training enhanced model on corrected dataset (150 epochs) ..." | tee -a "$LOG_ALL"
bash run_enhanced_150_correct.sh

echo "[$(date)] ===== All experiments completed =====" | tee -a "$LOG_ALL"
