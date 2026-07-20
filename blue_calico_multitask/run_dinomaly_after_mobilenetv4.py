"""
Wait for MobileNetV4 training to finish, then launch Dinomaly training.
"""
import os
import subprocess
import sys
import time

mobilenet_ckpt = "checkpoints_external_mobilenetv4_conv_small.e1200_r224_in1k_correct/checkpoint_last.pt"
print(f"[Dinomaly launcher] Waiting for MobileNetV4 checkpoint: {mobilenet_ckpt}")

max_wait_hours = 2
interval = 60
checked = 0
while not os.path.exists(mobilenet_ckpt):
    time.sleep(interval)
    checked += 1
    if checked % 5 == 0:
        print(f"[Dinomaly launcher] Still waiting... ({checked * interval // 60} min elapsed)")
    if checked * interval > max_wait_hours * 3600:
        print("[Dinomaly launcher] Timeout: MobileNetV4 checkpoint not found within 2 hours. Exiting.")
        sys.exit(1)

print("[Dinomaly launcher] MobileNetV4 checkpoint found. Starting Dinomaly training.")

cmd = [
    "conda", "run", "-n", "guizhou_emb", "python", "train_dinomaly_anomalib.py",
    "--data_root", "data/guizhou_embroidery_dinomaly",
    "--save_dir", "checkpoints_dinomaly_correct",
    "--image_size", "392",
    "--batch_size", "16",
    "--epochs", "2",
    "--encoder", "dinov2reg_vit_small_14",
    "--seed", "42",
]

with open("dinomaly_train.log", "w", encoding="utf-8") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    proc.wait()

print(f"[Dinomaly launcher] Dinomaly finished with exit code {proc.returncode}")
sys.exit(proc.returncode)
