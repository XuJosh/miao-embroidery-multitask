# 沈绣识别网络复现（PyTorch）

本项目复现了论文 **《Research on the innovative application of Shen Embroidery cultural heritage based on convolutional neural network》** 中的核心网络结构：

- **基线网络**：MobileNet V1
- **改进网络**：MobileNet V1 + 空间金字塔池化（SPP）
- **训练策略**：数据增强 + 迁移学习 + 微调

## 项目结构

```
shen_embroidery_reproduction/
├── model.py                 # 网络定义（MobileNet V1、SPP、对比模型）
├── train.py                 # 训练脚本
├── eval.py                  # 评估脚本（准确率、混淆矩阵）
├── generate_synthetic_data.py  # 生成合成数据用于跑通流程
├── requirements.txt         # 依赖
└── README.md                # 本说明
```

## 环境准备

已经在项目根目录创建并激活了虚拟环境（`../.venv`）。如需重新安装依赖：

```bash
cd d:\song\kimi_DEMO
.venv\Scripts\activate
pip install -r shen_embroidery_reproduction\requirements.txt
```

> 注：当前已安装 `torch`、`torchvision`。若希望使用迁移学习，`timm` 可提供 ImageNet 预训练的 MobileNet-V1 权重。

## 快速跑通（合成数据）

1. **生成合成数据集**（仅用于验证代码，不能复现论文 98.45% 精度）：

```bash
python shen_embroidery_reproduction\generate_synthetic_data.py --root data/synthetic
```

2. **训练改进版 MobileNet V1（不加载预训练权重，快速验证）**：

```bash
python shen_embroidery_reproduction\train.py \
    --data_dir data/synthetic \
    --model improved_mobilenet_v1 \
    --epochs 10 \
    --batch_size 4 \
    --lr 0.001 \
    --no_pretrained
```

3. **训练基线 MobileNet V1**：

```bash
python shen_embroidery_reproduction\train.py \
    --data_dir data/synthetic \
    --model mobilenet_v1 \
    --epochs 10 \
    --batch_size 4 \
    --lr 0.001 \
    --no_pretrained
```

4. **评估**：

```bash
python shen_embroidery_reproduction\eval.py \
    --data_dir data/synthetic \
    --model improved_mobilenet_v1 \
    --checkpoint checkpoints/improved_mobilenet_v1_best.pth
```

## 使用真实沈绣数据复现论文结果

将数据集按如下目录结构放置：

```
data/shen_embroidery/
    train/
        shenxiu/          # 沈绣训练图像
        fei/              # 非沈绣训练图像
    val/
        shenxiu/
        fei/
    test/
        shenxiu/
        fei/
```

论文参数：

- 输入尺寸：224×224
- batch size：4
- 初始学习率：0.001
- epoch：200
- 数据增强：翻转、旋转、颜色变换

**启用迁移学习（推荐，可接近论文 98.45% 精度）**：

```bash
python shen_embroidery_reproduction\train.py \
    --data_dir data/shen_embroidery \
    --model improved_mobilenet_v1 \
    --epochs 200 \
    --batch_size 4 \
    --lr 0.001 \
    --pretrained
```

**仅训练分类层（快速微调）**：

```bash
python shen_embroidery_reproduction\train.py \
    --data_dir data/shen_embroidery \
    --model improved_mobilenet_v1 \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.001 \
    --pretrained \
    --freeze_backbone
```

## 模型说明

### MobileNet V1 结构

按照论文表 1 实现：

1. 标准卷积（3×3，stride=2，32 通道）
2. 13 组深度可分离卷积（depthwise + pointwise）
3. 输出 7×7×1024 特征图

### SPP 模块

将原本的全局平均池化替换为 **空间金字塔池化**，使用 1×1、3×3、5×5 三个尺度的 adaptive max pooling，拼接后送入全连接层：

```
输入特征：(B, 1024, 7, 7)
SPP 输出维度：1024 × (1² + 3² + 5²) = 35,840
```

### 对比模型

`train.py` 也支持论文中对比的模型：

- `alexnet`
- `vgg16`
- `resnet50`
- `inception_v3`

示例：

```bash
python shen_embroidery_reproduction\train.py \
    --data_dir data/shen_embroidery \
    --model resnet50 \
    --epochs 200 \
    --batch_size 4 \
    --lr 0.001 \
    --pretrained
```

## 注意事项

1. **完整复现需要真实沈绣数据集**。论文数据集主要来自南通沈绣博物馆，部分数据可在 https://www.scidb.cn/en/s/JR7ZJn 申请。
2. **迁移学习权重**：默认尝试通过 `timm` 加载 ImageNet 预训练的 MobileNet-V1。如果网络受限，会回退到随机初始化，并在训练日志中提示。
3. **Windows 用户**：`--workers` 建议保持 0，避免多进程报错。
4. **GPU/CPU**：脚本会自动检测 CUDA，无 GPU 时使用 CPU。

## 参考

- Zhu J., Zhu C. *Research on the innovative application of Shen Embroidery cultural heritage based on convolutional neural network*. Scientific Reports, 2024.
- Howard A. G. et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. 2017.
