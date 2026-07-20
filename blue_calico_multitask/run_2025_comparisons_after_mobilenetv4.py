"""
After MobileNetV4 finishes 60 epochs, run three additional 2025 comparison methods:
    1. MambaOut-Small (CVPR 2025) multi-task, 50 epochs
    2. MambaOut-Tiny (CVPR 2025) multi-task, 50 epochs
    3. UniNet (CVPR 2025 anomaly detection), 50 epochs
"""
import os
import subprocess
import sys
import time

PYTHON = r"C:\Anaconda\envs\guizhou_emb\python.exe"


def run_cmd(cmd, log_file):
    print(f"[2025 comparisons] Running: {' '.join(cmd)}  ->  log: {log_file}")
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['HF_HUB_OFFLINE'] = '1'
    with open(log_file, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        proc.wait()
    print(f"[2025 comparisons] Finished with exit code {proc.returncode}")
    return proc.returncode


def wait_for_mobilenetv4_60(timeout_hours=3):
    marker = "checkpoints_external_mobilenetv4_conv_small.e1200_r224_in1k_correct/embroidery_cls_epoch60.pth"
    print(f"[2025 comparisons] Waiting for MobileNetV4 60-epoch marker: {marker}")
    waited = 0
    interval = 60
    while not os.path.exists(marker):
        time.sleep(interval)
        waited += interval
        if waited % 300 == 0:
            print(f"[2025 comparisons] Still waiting... {waited // 60} min elapsed")
        if waited > timeout_hours * 3600:
            print("[2025 comparisons] Timeout waiting for MobileNetV4. Exiting.")
            sys.exit(1)
    print("[2025 comparisons] MobileNetV4 60-epoch marker found.")


def main():
    wait_for_mobilenetv4_60()

    base_train = [
        PYTHON, "train_external.py",
        "--epochs", "50", "--batch_size", "48", "--lr", "1e-3",
        "--input_size", "224", "--workers", "4",
    ]

    # 1) MambaOut-Small
    backbone = "mambaout_small.in1k"
    save_dir = f"checkpoints_external_{backbone}_correct"
    ret = run_cmd(base_train + ["--backbone", backbone, "--no_amp"], "mambaout_small_train.log")
    if ret == 0:
        run_cmd(
            [PYTHON, "evaluate_external_checkpoints.py",
             "--backbone", backbone,
             "--data_dir", "data/guizhou_embroidery_correct",
             "--save_dir", save_dir,
             "--batch_size", "64",
             "--output", "mambaout_small_test_results.json"],
            "mambaout_small_eval.log",
        )

    # 2) MambaOut-Tiny
    backbone = "mambaout_tiny.in1k"
    save_dir = f"checkpoints_external_{backbone}_correct"
    ret = run_cmd(base_train + ["--backbone", backbone, "--no_amp"], "mambaout_tiny_train.log")
    if ret == 0:
        run_cmd(
            [PYTHON, "evaluate_external_checkpoints.py",
             "--backbone", backbone,
             "--data_dir", "data/guizhou_embroidery_correct",
             "--save_dir", save_dir,
             "--batch_size", "64",
             "--output", "mambaout_tiny_test_results.json"],
            "mambaout_tiny_eval.log",
        )

    # 3) UniNet (CVPR 2025 anomaly detection)
    run_cmd(
        [PYTHON, "train_uninet_anomalib.py",
         "--data_root", "data/guizhou_embroidery_dinomaly",
         "--save_dir", "checkpoints_uninet_correct",
         "--image_size", "224",
         "--batch_size", "16",
         "--epochs", "50",
         "--seed", "42"],
        "uninet_train.log",
    )

    print("[2025 comparisons] All 2025 comparison experiments completed.")


if __name__ == '__main__':
    main()
