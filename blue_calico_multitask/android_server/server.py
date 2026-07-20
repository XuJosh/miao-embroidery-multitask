"""
苗绣识别 Android 客户端的后端服务。
启动后监听 HTTP 请求，接收手机上传的图片，返回真伪/纹样/疵点识别结果。

用法：
    python android_server/server.py
    # 然后在同一局域网内的 Android 客户端中填入本机 IP:5000
"""

import os
import sys
import tempfile
import json
from pathlib import Path

from flask import Flask, request, jsonify

# 让服务可以找到 original_package 里的模型加载代码
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'original_package'))

from model_loader import build_model, load_checkpoint, predict


app = Flask(__name__)

# ---------- 服务启动时加载模型 ----------
CHECKPOINT = os.path.normpath(
    Path(__file__).resolve().parent.parent / 'original_package' / 'checkpoints' / 'embroidery_cls_best.pth'
)
print(f'Loading model from {CHECKPOINT} ...')
MODEL, DEVICE = build_model('auto')
load_checkpoint(MODEL, CHECKPOINT, DEVICE)
print(f'Model loaded on {DEVICE}')


@app.route('/')
def index():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})


@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # 保存到临时文件
    suffix = Path(file.filename).suffix or '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict(MODEL, tmp_path, DEVICE)
        # 按“真品才给纹样和疵点”的逻辑组织返回结果
        response = {
            'auth': result['auth'],
            'is_real': result['auth']['label'] == 1,
        }
        if response['is_real']:
            response['pattern'] = result['pattern']
            response['defect'] = result['defect']
        else:
            response['pattern'] = None
            response['defect'] = None
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == '__main__':
    # host='0.0.0.0' 允许局域网内其他设备访问
    app.run(host='0.0.0.0', port=5000, debug=False)
