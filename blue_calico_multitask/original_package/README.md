# 苗绣多任务识别系统（原网络版）

本项目使用**未剪枝的 `combined_fused_correct` 原网络**封装成一个可运行的 GUI 软件。

## 目录结构

```
original_package/
├── app.py                              # GUI 主程序
├── predict_cli.py                      # 命令行推理工具
├── model_loader.py                     # 模型加载 / 推理接口
├── model_combined.py                   # 原网络结构（复制）
├── model_enhanced.py                   # 组件定义（复制）
├── model_dinov2_mamba.py               # DINOv2 / Transformer 组件（复制）
├── checkpoints/
│   └── embroidery_cls_best.pth         # 原网络权重（state_dict）
├── requirements.txt
└── README.md
```

## 快速开始

1. 激活环境：
   ```bash
   conda activate guizhou_emb
   ```

2. 运行 GUI：
   ```bash
   cd original_package
   python app.py
   ```

3. 点击 **加载模型**，选择测试图片，点击 **开始识别**。

## 界面说明

新版 GUI 基于 `customtkinter`，采用现代扁平化设计：

- 左侧蓝色侧边栏：模型选择、图片选择、**大按钮「▶ 开始识别」**
- 右侧主区域：上方图片预览，下方三个结果卡片（真伪 / 纹样 / 疵点）
- 底部状态栏 + 进度条

加载模型和推理时会显示进度条，避免界面假死。

## 依赖安装

```bash
pip install -r requirements.txt
```

主要新增 `customtkinter`，用于美化界面。

## 命令行推理

```bash
cd original_package
python predict_cli.py --image ../data/guizhou_embroidery_correct/images/test/test_00000_a0.jpg
```

## 模型性能（测试集）

| 指标 | 数值 |
|---|---|
| 总参数 | 49,524,418 |
| auth_acc | 0.969 |
| auth_auc | 0.990 |
| pat_acc | 0.996 |
| def_acc | 0.931 |
| def_auc | 0.948 |

## 注意事项

- 第一次加载模型时会从本地或网络加载 DINOv2 预训练权重，耗时约 10–30 秒。
- 本包内的 `model_*.py` 为项目源码的复制件，用于保证包可独立导入，未修改原项目文件。
- 如需加载其他 checkpoint，可在 GUI 中点击“浏览”选择 `.pt` / `.pth` 文件。
