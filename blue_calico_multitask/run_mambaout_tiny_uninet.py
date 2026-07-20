"""
Resume the remaining experiments after MambaOut-Small is done:
    1. MambaOut-Tiny (CVPR 2025) 50 epoch + eval
    2. UniNet (CVPR 2025) anomaly-detection training with custom loop
"""
import os
import subprocess

PYTHON = r"C:\Anaconda\envs\guizhou_emb\python.exe"


def run(cmd, log_file):
    print(f"\n[remaining] {'='*60}")
    print(f"[remaining] Running: {' '.join(cmd)}")
    print(f"[remaining] Log: {log_file}")
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['HF_HUB_OFFLINE'] = '1'
    with open(log_file, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
        proc.wait()
    print(f"[remaining] Exit code: {proc.returncode}")
    return proc.returncode


def main():
    steps = [
        # 1) MambaOut Tiny 50ep + eval
        ([PYTHON, "train_external.py",
          "--backbone", "mambaout_tiny.in1k",
          "--epochs", "50", "--batch_size", "48", "--lr", "1e-3",
          "--input_size", "224", "--workers", "4",
          "--no_amp"],
         "mambaout_tiny_train.log"),
        ([PYTHON, "evaluate_external_checkpoints.py",
          "--backbone", "mambaout_tiny.in1k",
          "--data_dir", "data/guizhou_embroidery_correct",
          "--save_dir", "checkpoints_external_mambaout_tiny.in1k_correct",
          "--batch_size", "64",
          "--output", "mambaout_tiny_test_results.json",
          "--no_amp"],
         "mambaout_tiny_eval.log"),

        # 2) UniNet anomaly detection (custom loop to avoid Lightning hang)
        ([PYTHON, "train_uninet_custom.py",
          "--data_root", "data/guizhou_embroidery_dinomaly",
          "--save_dir", "checkpoints_uninet_224",
          "--image_size", "224", "--batch_size", "8",
          "--epochs", "5", "--train_subset", "2000", "--seed", "42"],
         "uninet_train.log"),
    ]

    results = {}
    for i, (cmd, log) in enumerate(steps, 1):
        ret = run(cmd, log)
        results[i] = ret
        if ret != 0:
            print(f"[remaining] Step {i} failed; continuing to next step if any.")

    print("\n[remaining] Summary:")
    for i, ret in results.items():
        print(f"  Step {i}: {'OK' if ret == 0 else 'FAILED'} ({ret})")


if __name__ == '__main__':
    main()
