#!/usr/bin/env bash
# Monitor the Swin-Tiny training process and automatically launch the two
# ablation experiments once it has completed 50 epochs successfully.

set -e

cd /d/song/kimi_DEMO/blue_calico_multitask
source /c/Anaconda/etc/profile.d/conda.sh
conda activate guizhou_emb

SAVE_DIR="checkpoints_external_swin_t_correct"
CHECKPOINT="$SAVE_DIR/checkpoint_last.pt"
EPOCHS=50
POLL_INTERVAL=60
MAX_WAIT_HOURS=8
MAX_ITER=$((MAX_WAIT_HOURS * 3600 / POLL_INTERVAL))

is_swin_running() {
    cmd //c wmic process where "name='python.exe'" get ProcessId,CommandLine /format:csv 2>/dev/null \
        | tr -d '\r' \
        | grep -i "train_external.py" \
        | grep -i "swin_t" \
        | head -n 1
}

echo "[$(date)] Monitoring Swin-Tiny training..."

for i in $(seq 1 $MAX_ITER); do
    RUNNING=$(is_swin_running || true)

    if [ -z "$RUNNING" ]; then
        echo "[$(date)] Swin-Tiny process not found. Checking checkpoint..."
        if [ -f "$CHECKPOINT" ]; then
            EPOCH=$(python - <<PY
import torch
try:
    ckpt = torch.load(r'$CHECKPOINT', map_location='cpu', weights_only=False)
    print(ckpt.get('epoch', 0))
except Exception as e:
    print(0)
PY
            )
            if [ "$EPOCH" -ge "$EPOCHS" ]; then
                echo "[$(date)] Swin-Tiny completed epoch $EPOCH. Starting ablations."
                bash run_ablations_after_swin.sh
                exit 0
            else
                echo "[$(date)] Swin-Tiny checkpoint epoch $EPOCH < $EPOCHS. Aborting ablation launch."
                exit 1
            fi
        else
            echo "[$(date)] No checkpoint found. Aborting ablation launch."
            exit 1
        fi
    fi

    sleep $POLL_INTERVAL
done

echo "[$(date)] Monitor timed out after $MAX_WAIT_HOURS hours."
exit 1
