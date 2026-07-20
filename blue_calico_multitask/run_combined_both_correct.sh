#!/usr/bin/env bash
# Run both combined-model experiments sequentially on the corrected dataset.
#   1. combined_correct:       auth/defect from ResNet50, pattern from fusion
#   2. combined_fused_correct: auth/defect/pattern all from fusion

set -e
set -o pipefail

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

echo "[$(date)] ===== Starting combined experiment sequence ====="

# 1. Non-fused variant (resume if checkpoint exists)
echo "[$(date)] Step 1/2: combined_correct (auth/defect from ResNet50)"
bash run_combined_correct.sh

# 2. Fused variant
echo "[$(date)] Step 2/2: combined_fused_correct (auth/defect from fused feature)"
bash run_combined_fused_correct.sh

echo "[$(date)] ===== Combined experiment sequence completed ====="
