# 苗绣识别 Android 客户端

由于当前 `combined_fused_correct` 模型（DINOv2 + ResNet50 + Transformer，约 190MB）直接转换为 TorchScript/Android 原生模型失败，这里提供了一套更实用的方案：

> **手机拍照 → 通过局域网把图片发给电脑上的 Python 服务端 → 服务端用原网络推理 → 返回结果到手机显示**

电脑端的 `original_package` 软件保留不动。

## 目录结构

```
android_client/          # Android Studio 项目
├── app/src/main/...
│   ├── java/com/example/embroidery/MainActivity.kt
│   ├── res/layout/activity_main.xml
│   └── AndroidManifest.xml
└── README.md

android_server/
└── server.py            # Python 后端服务
```

## 使用步骤

### 1. 启动 Python 后端服务

在电脑（已安装 `guizhou_emb` 环境）上运行：

```bash
cd android_server
python server.py
```

启动后会显示电脑监听的 IP 和端口（默认 `0.0.0.0:5000`）。

查看本机局域网 IP：

```bash
ipconfig
```

假设电脑 IP 为 `192.168.1.5`，服务地址就是 `192.168.1.5:5000`。

### 2. 导入 Android 项目

1. 用 Android Studio 打开 `android_client` 文件夹。
2. 等待 Gradle 同步完成。
3. 用 USB 连接手机，或启动模拟器。
4. 点击运行。

### 3. 手机端操作

1. 确保手机和电脑连接**同一个 Wi-Fi**。
2. 打开 App，在顶部输入框填入电脑 IP，例如 `192.168.1.5:5000`。
3. 点击 **📷 拍照** 或 **🖼️ 相册** 选择一张苗绣图片。
4. 点击 **▶ 开始识别**。
5. 等待几秒后，下方卡片显示结果：
   - 伪品：仅显示真伪结果
   - 真品：显示真伪 + 纹样 + 疵点

## 注意事项

- 首次启动后端时需要加载 DINOv2 权重，耗时约 10–30 秒。
- Android 9+ 需要 `android:usesCleartextTraffic="true"` 才能访问 `http://`（已在 AndroidManifest.xml 中开启）。
- 如果无法访问，请检查防火墙是否允许 5000 端口。
- 由于模型较大，推理约需 1–3 秒（取决于电脑 GPU/CPU）。

## 后续做成纯离线 Android 的方案

如果希望未来不依赖电脑服务端，需要：

1. 把模型骨干换成更轻量的网络（如 MobileNetV4、EfficientNet-B0），重新训练。
2. 将最终模型导出为 TorchScript / ONNX / TFLite。
3. 在 Android 端用 PyTorch Mobile / ONNX Runtime Mobile 推理。

当前项目里的 `external_mobilenetv4` 等实验已经证明轻量骨干可行，可以在此基础上继续。
