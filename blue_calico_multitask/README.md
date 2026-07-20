# 南通蓝印花布智能鉴真多任务网络

基于选题三的升级版思路实现：

- **任务**：真伪鉴别 + 纹样分类 + 疵点分割
- **结构**：CNN 纹理分支 + ViT 结构分支 + 多任务头
- **可解释性**：Grad-CAM 可视化鉴真依据
- **数据**：合成蓝印花布数据集（可替换为真实图像）

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | BlueCalicoNet 网络、损失函数 |
| `dataset.py` | PyTorch Dataset |
| `generate_data.py` | 生成合成蓝印花布数据集 |
| `train.py` | 多任务训练 |
| `eval.py` | 评估 + 混淆矩阵 + Grad-CAM |
| `gradcam.py` | Grad-CAM 实现（支持 auth / defect 头） |
| `localize_defects.py` | 弱监督疵点定位：cls 模型 + Grad-CAM |

## 环境

已在 `../.venv` 中安装 PyTorch。如需额外依赖：

```bash
pip install matplotlib scikit-learn Pillow numpy
```

## 快速运行

```bash
cd d:\song\kimi_DEMO
.venv\Scripts\activate

# 1. 生成合成数据
python blue_calico_multitask\generate_data.py --root data/blue_calico

# 2a. 需要像素级疵点 mask（默认 --defect_mode seg）
python blue_calico_multitask\train.py \
    --data_dir data/blue_calico \
    --defect_mode seg \
    --epochs 30 --batch_size 8 --lr 0.001 \
    --w_auth 1.0 --w_pattern 0.5 --w_defect 1.0 \
    --save_dir checkpoints_bc

# 2b. 只有图像级“是否有疵点”标签（不需要 mask）
python blue_calico_multitask\train.py \
    --data_dir data/blue_calico \
    --defect_mode cls \
    --epochs 30 --batch_size 8 --lr 0.001 \
    --w_auth 1.0 --w_pattern 0.5 --w_defect 1.0 \
    --save_dir checkpoints_bc_cls

# 3. 评估 + Grad-CAM（真伪头）
python blue_calico_multitask\eval.py \
    --data_dir data/blue_calico \
    --defect_mode cls \
    --checkpoint checkpoints_bc_cls\bluecalico_cls_best.pth \
    --save_dir cam_outputs

# 4. 弱监督疵点定位（用 defect 头做 Grad-CAM）
python blue_calico_multitask\localize_defects.py \
    --data_dir data/blue_calico \
    --checkpoint checkpoints_bc_cls\bluecalico_cls_best.pth \
    --save_dir localization_outputs \
    --threshold 0.5 \
    --num_vis 10
```

> 说明：`--defect_mode cls` 时，模型把疵点检测退化为**图像级二分类**（有疵点 / 无疵点），只需要在 `labels.json` 里给每张图标 `defect: 0/1`，**完全不需要像素级 mask**。
> 
> `localize_defects.py` 进一步用 **Grad-CAM on defect head** 把图像级标签“升级”为像素级定位图，实现弱监督疵点定位。

## 网络结构

```
输入图像 (3, 224, 224)
    ├─ CNN 纹理分支 ──┐
    │   → 输出特征图 (256, 14, 14)
    │   → 输出向量 256-d
    │
    └─ ViT 结构分支 ──┘
        → patch embedding
        → Transformer encoder
        → 输出向量 128-d

融合层 (256+128 → 256)
    ├─ 真伪分类头 → 2 类
    ├─ 纹样分类头 → 5 类
    ├─ 疵点分割头（--defect_mode seg）→ (1, 224, 224)
    └─ 疵点分类头（--defect_mode cls）→ 2 类（有/无疵点）
```

## 可扩展方向

1. **真实数据**：把 `data/blue_calico/images` 和 `masks` 替换为真实蓝印花布图像与疵点标注。
2. **生成式增强**：在 `generate_data.py` 中加入 GAN/Diffusion 生成缺陷样本。
3. **轻量化部署**：训练后用 `torch.quantization` 或 ONNX 导出到移动端。
4. **注意力可视化**：除了 Grad-CAM，还可以可视化 ViT 的 attention map。
5. **异常检测**：把真伪鉴别改为 One-Class / 自编码器异常检测。

## 参考文献思路

- 原论文：沈绣 + MobileNet V1 + SPP
- 本代码：蓝印花布 + CNN-ViT 双分支 + 多任务 + Grad-CAM
