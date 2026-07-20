#!/usr/bin/env bash
# Wait for combined_fused_correct no ArcFace training to finish,
# then run correct evaluations for Swin, ResNet50, and combined no ArcFace.

set -e

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

DATA_DIR="data/guizhou_embroidery_correct"
TARGET_EPOCH=50
POLL_INTERVAL=120

get_combined_epoch() {
    python - <<PY
import torch, os
p = r'checkpoints_combined_fused_correct_no_arcface/checkpoint_last.pt'
if os.path.exists(p):
    try:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        print(ckpt.get('epoch', 0))
    except Exception:
        print(0)
else:
    print(0)
PY
}

echo "[$(date)] Monitoring combined_fused_correct no ArcFace training..."
while true; do
    EPOCH=$(get_combined_epoch)
    echo "[$(date)] Combined no ArcFace checkpoint_last epoch $EPOCH / $TARGET_EPOCH"
    if [ "$EPOCH" -ge "$TARGET_EPOCH" ]; then
        echo "[$(date)] Combined no ArcFace appears finished. Waiting 60s for cleanup..."
        sleep 60
        break
    fi
    sleep $POLL_INTERVAL
done

echo "[$(date)] Running Scheme A evaluations..."

# Swin-Tiny (lr=1e-4)
python evaluate_external_checkpoints.py \
    --backbone swin_t \
    --data_dir $DATA_DIR \
    --save_dir checkpoints_external_swin_t_correct \
    --batch_size 64 \
    --device cuda \
    --output results_external_swin_t_correct_lr1e-4_test.json \
    > logs/eval_swin_t_correct_lr1e-4.log 2>&1

# ResNet50 (50 epochs)
python evaluate_external_checkpoints.py \
    --backbone resnet50 \
    --data_dir $DATA_DIR \
    --save_dir checkpoints_external_resnet50_correct_50epochs \
    --batch_size 64 \
    --device cuda \
    --output results_external_resnet50_correct_50_test.json \
    > logs/eval_resnet50_correct_50.log 2>&1

# combined_fused_correct no ArcFace
python evaluate_combined.py \
    --data_dir $DATA_DIR \
    --save_dir checkpoints_combined_fused_correct_no_arcface \
    --batch_size 32 \
    --device cuda \
    --auth_defect_use_fused \
    --no_arcface \
    --output results_combined_fused_correct_no_arcface_test.json \
    > logs/eval_combined_fused_correct_no_arcface.log 2>&1

echo "[$(date)] All Scheme A evaluations finished."

# Print concise summary
python - <<PY
import json, os
files = {
    'Swin-Tiny': 'results_external_swin_t_correct_lr1e-4_test.json',
    'ResNet50': 'results_external_resnet50_correct_50_test.json',
    'combined_fused_correct (no ArcFace)': 'results_combined_fused_correct_no_arcface_test.json',
}
print("\n=== Scheme A corrected-test summary ===")
for name, f in files.items():
    if not os.path.exists(f):
        print(f"{name}: results not found ({f})")
        continue
    data = json.load(open(f, encoding='utf-8'))
    best = max(data, key=lambda x: x.get('def_acc', 0))
    print(f"{name}: auth_acc={best['auth_acc']:.3f} pat_acc={best['pat_acc']:.3f} def_acc={best['def_acc']:.3f}  @ epoch={best.get('epoch','?')} ({os.path.basename(best['path'])})")
PY
