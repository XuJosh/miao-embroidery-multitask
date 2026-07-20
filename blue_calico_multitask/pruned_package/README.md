# 苗绣多任务识别系统（剪枝版）

本项目对 `combined_fused_correct` 网络进行了**全局 L1 非结构化剪枝**（30% 权重置零），并打包为一个可运行的 GUI 软件。原始网络文件与 checkpoint **未被删除**，仍保留在 `../checkpoints_combined_fused_correct/`。

## 目录结构

```
pruned_package/
├── app.py                              # GUI 主程序
├── model_loader.py                     # 模型加载 / 推理接口
├── model_combined.py                   # 原网络结构（复制）
├── model_enhanced.py                   # 组件定义（复制）
├── model_dinov2_mamba.py               # DINOv2 / Transformer 组件（复制）
├── checkpoints/
│   └── checkpoint_best_pruned_amount0.3.pth.gz   # 剪枝后 checkpoint
├── requirements.txt
└── README.md
```

## 快速开始

1. 确保已激活 `guizhou_emb` 环境（或安装了 `requirements.txt` 中的包）：
   ```bash
   conda activate guizhou_emb
   ```

2. 运行 GUI：
   ```bash
   cd pruned_package
   python app.py
   ```

3. 界面默认加载“剪枝网络”。点击 **加载模型**，然后选择测试图片，点击 **开始识别** 即可查看结果。

## 命令行单张图片推理

```bash
cd pruned_package
python -c "from model_loader import predict_from_path; \
import json; \
print(json.dumps(predict_from_path('checkpoints/checkpoint_best_pruned_amount0.3.pth.gz', \
'../data/guizhou_embroidery_correct/images/test/test_00000_a0.jpg'), \
ensure_ascii=False, indent=2))"
```

## 加载原网络对比

在 GUI 中：
- 网络类型选择 **原始网络**
- Checkpoint 路径默认指向 `../checkpoints_combined_fused_correct/checkpoint_best.pt`
- 点击 **加载模型** 即可使用未剪枝网络推理。

## 剪枝效果（在测试集上）

| 指标 | 原网络 | 剪枝网络（30%） | 变化 |
|---|---|---|---|
| 总参数量 | 49,524,418 | 49,524,418 | 不变（架构相同） |
| 非零参数量 | 49,524,034 | 34,923,451 | ↓ 29.5% |
| 真伪准确率 | 0.969 | 0.933 | ↓ 0.036 |
| 纹样准确率 | 0.996 | 0.957 | ↓ 0.039 |
| 疵点准确率 | 0.931 | 0.875 | ↓ 0.056 |
| 疵点 AUC | 0.948 | 0.917 | ↓ 0.031 |

> 详细数据见 `checkpoints/checkpoint_best_pruned_amount0.3_report.json`。

## 重新生成其他剪枝比例

如需生成 10%/50% 等比例的剪枝模型，可回到项目根目录运行：

```bash
python prune_combined_fused.py --amount 0.5 --device cuda
```

生成的 checkpoint 会保存在 `../checkpoints_combined_fused_correct_pruned/`。

## 注意事项

- 第一次加载模型时会从本地或网络加载 DINOv2 预训练权重，耗时约 10–30 秒。
- 剪枝为 **非结构化剪枝**，对普通 PyTorch 推理加速有限；如需真正加速，建议进一步做结构化剪枝、量化或替换更轻量骨干。
- 本包内的 `model_*.py` 为项目源码的复制件，仅用于保证包可独立导入，未修改原项目文件。
