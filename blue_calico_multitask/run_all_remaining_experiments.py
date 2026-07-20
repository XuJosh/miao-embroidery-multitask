"""
Run all remaining comparison experiments sequentially:
    1. Dinomaly 2 epoch
    2. MobileNetV4 31-60 epoch resume
    3. MambaOut-Small (CVPR 2025) 50 epoch + eval
    4. MambaOut-Tiny (CVPR 2025) 50 epoch + eval
    5. UniNet 50 epoch

Uses the conda env's python directly and sets PYTHONIOENCODING=utf-8 to avoid
conda-run UnicodeEncodeError on Windows.
"""
import os
import subprocess
import sys

PYTHON = r"C:\Anaconda\envs\guizhou_emb\python.exe"


def run(cmd, log_file):
    print(f"\n[run_all] {'='*60}")
    print(f"[run_all] Running: {' '.join(cmd)}")
    print(f"[run_all] Log: {log_file}")
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['HF_HUB_OFFLINE'] = '1'
    with open(log_file, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
        proc.wait()
    print(f"[run_all] Exit code: {proc.returncode}")
    return proc.returncode


def main():
    steps = []

    # 1) Dinomaly
    steps.append((
        [PYTHON, "train_dinomaly_anomalib.py",
         "--data_root", "data/guizhou_embroidery_dinomaly",
         "--save_dir", "checkpoints_dinomaly_correct",
         "--image_size", "392", "--batch_size", "16",
         "--epochs", "2", "--encoder", "dinov2reg_vit_small_14", "--seed", "42"],
        "dinomaly_train.log"
    ))

    # 2) MobileNetV4 resume 31-60
    steps.append((
        [PYTHON, "train_external.py",
         "--backbone", "mobilenetv4_conv_small.e1200_r224_in1k",
         "--epochs", "60", "--batch_size", "48", "--lr", "1e-3",
         "--input_size", "224", "--workers", "4",
         "--resume", "checkpoints_external_mobilenetv4_conv_small.e1200_r224_in1k_correct/checkpoint_last.pt"],
        "mobilenetv4_60ep_train.log"
    ))

    # 3) MambaOut-Small 50ep + eval
    backbone = "mambaout_small.in1k"
    save_dir = f"checkpoints_external_{backbone}_correct"
    steps.append((
        [PYTHON, "train_external.py",
         "--backbone", backbone,
         "--epochs", "50", "--batch_size", "48", "--lr", "1e-3",
         "--input_size", "224", "--workers", "4",
         "--no_amp"],
        "mambaout_small_train.log"
    ))
    steps.append((
        [PYTHON, "evaluate_external_checkpoints.py",
         "--backbone", backbone,
         "--data_dir", "data/guizhou_embroidery_correct",
         "--save_dir", save_dir,
         "--batch_size", "64",
         "--output", "mambaout_small_test_results.json"],
        "mambaout_small_eval.log"
    ))

    # 4) MambaOut-Tiny 50ep + eval
    backbone = "mambaout_tiny.in1k"
    save_dir = f"checkpoints_external_{backbone}_correct"
    steps.append((
        [PYTHON, "train_external.py",
         "--backbone", backbone,
         "--epochs", "50", "--batch_size", "48", "--lr", "1e-3",
         "--input_size", "224", "--workers", "4",
         "--no_amp"],
        "mambaout_tiny_train.log"
    ))
    steps.append((
        [PYTHON, "evaluate_external_checkpoints.py",
         "--backbone", backbone,
         "--data_dir", "data/guizhou_embroidery_correct",
         "--save_dir", save_dir,
         "--batch_size", "64",
         "--output", "mambaout_tiny_test_results.json"],
        "mambaout_tiny_eval.log"
    ))

    # 5) UniNet 50ep
    steps.append((
        [PYTHON, "train_uninet_anomalib.py",
         "--data_root", "data/guizhou_embroidery_dinomaly",
         "--save_dir", "checkpoints_uninet_correct",
         "--image_size", "224", "--batch_size", "16",
         "--epochs", "50", "--seed", "42"],
        "uninet_train.log"
    ))

    results = {}
    for i, (cmd, log) in enumerate(steps, 1):
        ret = run(cmd, log)
        results[i] = ret
        if ret != 0:
            print(f"[run_all] Step {i} failed; continuing to next step if any.")

    print("\n[run_all] Summary:")
    for i, ret in results.items():
        print(f"  Step {i}: {'OK' if ret == 0 else 'FAILED'} ({ret})")


if __name__ == '__main__':
    main()
