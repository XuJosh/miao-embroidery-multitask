"""
Wait for Dinomaly to finish, then continue MobileNetV4-Conv-Small from 30 -> 60 epochs.
"""
import os
import shutil
import subprocess
import sys
import time

dinomaly_result = "checkpoints_dinomaly_correct/dinomaly_results.json"
mobilenet_dir = "checkpoints_external_mobilenetv4_conv_small.e1200_r224_in1k_correct"
mobilenet_ckpt = os.path.join(mobilenet_dir, "checkpoint_last.pt")

print(f"[MobileNetV4 60ep launcher] Waiting for Dinomaly result: {dinomaly_result}")

max_wait_hours = 2
interval = 60
checked = 0
while not os.path.exists(dinomaly_result):
    time.sleep(interval)
    checked += 1
    if checked % 5 == 0:
        print(f"[MobileNetV4 60ep launcher] Still waiting... ({checked * interval // 60} min elapsed)")
    if checked * interval > max_wait_hours * 3600:
        print("[MobileNetV4 60ep launcher] Timeout: Dinomaly result not found within 2 hours. Exiting.")
        sys.exit(1)

print("[MobileNetV4 60ep launcher] Dinomaly finished. Preparing to resume MobileNetV4 training.")

# Backup the epoch-30 checkpoint before it gets overwritten
if os.path.exists(mobilenet_ckpt):
    backup_path = os.path.join(mobilenet_dir, "checkpoint_epoch30.pt")
    shutil.copy2(mobilenet_ckpt, backup_path)
    print(f"[MobileNetV4 60ep launcher] Backed up epoch-30 checkpoint to {backup_path}")

cmd = [
    "conda", "run", "-n", "guizhou_emb", "python", "train_external.py",
    "--backbone", "mobilenetv4_conv_small.e1200_r224_in1k",
    "--epochs", "60",
    "--batch_size", "48",
    "--lr", "1e-3",
    "--input_size", "224",
    "--workers", "4",
    "--resume", mobilenet_ckpt,
]

with open("mobilenetv4_60ep_train.log", "w", encoding="utf-8") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    proc.wait()

print(f"[MobileNetV4 60ep launcher] Training finished with exit code {proc.returncode}")
sys.exit(proc.returncode)
