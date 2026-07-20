#!/usr/bin/env bash
# Wait for Swin-Tiny rerun to finish, then launch Scheme A ablations.

set -e

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

SWIN_DIR="checkpoints_external_swin_t_correct"
TARGET_EPOCH=50
POLL_INTERVAL=120

# Backup the failed lr=1e-3 run if it still exists (no-op if already moved).
if [ -f "checkpoints_external_swin_t_correct_failed/best_model.pth" ] && [ ! -f "checkpoints_external_swin_t_correct/best_model.pth" ]; then
    echo "[$(date)] Old failed checkpoint backup already in place."
fi

echo "[$(date)] Monitoring Swin-Tiny training in $SWIN_DIR..."

get_max_epoch() {
    python - <<PY
import torch, os
best = r'$SWIN_DIR/checkpoint_best.pt'
last = r'$SWIN_DIR/checkpoint_last.pt'
epochs = []
for p in [best, last]:
    if os.path.exists(p):
        try:
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            epochs.append(ckpt.get('epoch', 0))
        except Exception:
            epochs.append(0)
    else:
        epochs.append(0)
print(max(epochs))
PY
}

while true; do
    EPOCH=$(get_max_epoch)
    echo "[$(date)] Swin-Tiny max checkpoint epoch $EPOCH / $TARGET_EPOCH"
    if [ "$EPOCH" -ge "$TARGET_EPOCH" ]; then
        echo "[$(date)] Swin-Tiny appears to have finished. Waiting an extra 60s for process cleanup..."
        sleep 60
        break
    fi
    sleep $POLL_INTERVAL
done

echo "[$(date)] Launching Scheme A ablations (ResNet50 50ep + combined_fused_correct no ArcFace 50ep + evaluations)..."
bash run_scheme_a_after_swin.sh
