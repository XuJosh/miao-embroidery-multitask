"""Robust downloader with resume and frequent progress output."""
import os
import sys
import time
import urllib.request

URL = "https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.3.1%2Bcu118-cp38-cp38-win_amd64.whl"
OUT = "wheelhouse/torch-2.3.1+cu118-cp38-cp38-win_amd64.whl"

os.makedirs("wheelhouse", exist_ok=True)

headers = {}
start = 0
if os.path.exists(OUT):
    start = os.path.getsize(OUT)
    headers['Range'] = f'bytes={start}-'
    print(f"Resuming from {start} bytes")

req = urllib.request.Request(URL, headers=headers)
with urllib.request.urlopen(req, timeout=60) as resp:
    total = int(resp.headers.get('Content-Length', 0)) + start
    print(f"Total size: {total} bytes ({total/1024**3:.2f} GB)")
    mode = 'ab' if start else 'wb'
    downloaded = start
    t0 = time.time()
    with open(OUT, mode) as f:
        while True:
            chunk = resp.read(1024*1024)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed = downloaded / elapsed / 1024**2 if elapsed > 0 else 0
            pct = downloaded / total * 100 if total else 0
            print(f"{pct:.1f}% {downloaded}/{total} bytes  {speed:.2f} MB/s")
            sys.stdout.flush()

print("Download complete:", OUT)
