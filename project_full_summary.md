# 贵州苗绣三分支动态多任务识别项目备忘录

**生成时间**：2026-07-20  
**工作目录**：`D:/song/kimi_DEMO`  
**核心项目**：`blue_calico_multitask/`

---

## 1. 项目概述

### 1.1 研究目标
设计一个**三分支动态多任务深度网络**，对**贵州苗绣/蓝印花布**图像同时进行三项任务：

1. **真伪鉴别（Auth）**：判断输入图像是真品还是伪品
2. **纹样识别（Pattern）**：识别苗绣的五种纹样类别
3. **疵点检测（Defect）**：检测图像中是否存在瑕疵

### 1.2 核心创新点
- **三分支特征提取**：
  - 局部纹理分支：捕获针脚、丝线等细观纹理
  - 全局语义分支：捕获纹样整体语义
  - 结构关系分支：通过 Patch Transformer 建模上下文关系
- **动态多任务损失**：基于不确定性加权，避免三个任务目标冲突
- **ArcFace 加性角度边际损失**：放大细粒度类别差异，提升真伪鉴别和纹样分类能力
- **多任务融合**：三个任务共享融合特征，相互辅助

### 1.3 主要难点
1. 纹理细微、疵点尺度小
2. 同类纹样差异大，异类共享主题
3. 缺陷与手工自然噪声难以区分
4. 伪品与真品边界模糊
5. 真实数据集有限，伪品/有瑕疵样本稀缺

---

## 2. 网络架构

### 2.1 任务头
- **Auth Head**：真伪二分类（真品/伪品）
- **Pattern Head**：纹样多分类（5 类）
- **Defect Head**：疵点检测（分割/二分类）

### 2.2 ArcFace 任务头
ArcFace 主要用于 **Auth Head** 和 **Pattern Head**，通过加性角度边际 `m` 拉大类别间角度差异：

```
L_ArcFace = -log(exp(s·cos(θ_y + m)) / Σ exp(s·cos(θ_i)))
```

其中：
- `s`：特征缩放因子
- `θ_y`：样本特征与真实类别权重向量的夹角
- `m`：角度边际

### 2.3 动态多任务损失
基于 Kendall 等 2018（参考文献 [17]）的不确定性加权策略：

```
L_total = Σ (1 / (2σ_i²)) · L_i + log(σ_i)
```

其中 `σ_i` 为每个任务的不确定性，网络自动学习权重。

**注意**：ArcFace 损失是**分类头**的边际损失，动态多任务损失是**整体多任务**的加权策略。两者作用层级不同。

### 2.4 网络描述文件
详见：`D:/song/kimi_DEMO/blue_calico_multitask/network_description.md`

---

## 3. 数据集

### 3.1 原始数据集
- **路径**：`D:/song/newdata_en`
- **总量**：5,000 张原始图像
- **每类数量**：1,000 张
- **类别目录**：
  - `byx`：辫绣
  - `dx`：堆绣
  - `mx`：马尾绣
  - `qt`：其他
  - `szmwx`：数纱马尾绣

### 3.2 数据转换脚本
- **生成扩增数据集**：`blue_calico_multitask/convert_guizhou_embroidery.py`
- **生成正确划分版本（推荐）**：`blue_calico_multitask/convert_guizhou_embroidery_correct.py`
  - 先按原始图像划分 train/val/test，避免数据泄漏
  - 仅对训练集做数据扩增（aug_factor=10）
  - 验证集和测试集保留原始图像，不扩增

### 3.3 扩增后数据集（`guizhou_embroidery` 与 `guizhou_embroidery_augfirst`）

| 子集 | 原始图像数 | 扩增后图像数 | 说明 |
|------|------------|--------------|------|
| 训练集 | 3,433 | 34,330 | 每张扩增 10 倍 |
| 验证集 | 736 | 736 | 不扩增 |
| 测试集 | 736 | 736 | 不扩增 |
| **合计** | **4,905** | **35,802** | 原始 5,000 张中约 95 张因分辨率过低被过滤 |

### 3.4 论文实验所用版本（`guizhou_embroidery_correct`）

#### 测试集分布（736 张）

| 统计项 | 数量 |
|--------|------|
| 测试集总数 | 736 |
| 真品 | 428 |
| 伪品 | 308 |
| 有瑕疵 | 204 |
| 无瑕疵 | 532 |

#### 五种纹样测试分布

| 纹样 | 缩写 | 测试集数量 | 真品-无瑕疵 | 真品-有瑕疵 | 伪品-无瑕疵 |
|------|------|------------|-------------|-------------|-------------|
| 辫绣 | bx | 145 | 44 | 29 | 72 |
| 堆绣 | dx | 150 | 48 | 47 | 55 |
| 马尾绣 | mx | 141 | 42 | 37 | 62 |
| 其他 | other | 150 | 41 | 50 | 59 |
| 数纱绣 | ssx | 150 | 49 | 41 | 60 |

### 3.5 伪品生成方式
伪品不是收集的，而是**从真品合成**而来。转换脚本以 40% 概率将真品转为伪品，采用三种策略之一：

1. **机绣模拟（machine）**：高斯模糊 + 降低饱和度 + 网格纹理
2. **低质量复制（low_quality）**：颜色通道偏移 + 高斯噪声 + 强 JPEG 压缩
3. **数字篡改（tamper）**：随机复制图像块并粘贴到另一位置

伪品保留原始纹样标签，因此五种纹样均有对应伪品。

### 3.6 学术可行性说明
- **可行**：人脸活体检测、指纹活体检测等领域广泛使用合成攻击样本。
- **必须注意**：论文中必须明确说明伪品是**合成生成**，不能冒充真实伪品数据集。
- **风险**：合成伪品分布可能与真实伪品存在差异，模型可能过拟合到生成痕迹上。
- **建议**：在论文局限性中说明，并尽可能补充真实伪品测试。

---

## 4. 实验结果

### 4.1 对比实验结果

| 方法 | auth_acc | auth_auc | pat_acc | def_acc | def_auc |
|------|----------|----------|---------|---------|---------|
| EfficientNet-B0 | 0.928 | 0.965 | 0.973 | 0.908 | 0.940 |
| EfficientNet-B3 | 0.938 | 0.979 | 0.996 | 0.928 | **0.950** |
| ViT-B/16 | 0.640 | 0.698 | 0.893 | 0.777 | 0.693 |
| ResNet50-CBAM [2026] | 0.952 | 0.989 | 0.965 | 0.923 | **0.962** |
| MobileNetV4-Conv-Small [2026] | **0.833** | 0.894 | **0.982** | **0.875** | 0.890 |
| MambaOut [2025] | 0.951 | 0.985 | 0.993 | 0.920 | **0.960** |
| **Ours（my net）** | **0.969** | **0.990** | **0.996** | **0.931** | 0.948 |

### 4.2 热力图分析结论
- **Ours**：关注区域与关键纹样、结构轮廓、局部纹理细节一致
- **ResNet50-CBAM**：能定位主要目标，但热力图响应较弥散
- **EfficientNet-B0/B3**：更关注局部纹理块，缺乏全局纹样语义
- **MobileNetV4-Conv-Small**：受模型容量限制，关注高对比度像素
- **MambaOut**：全局建模能力强，但对细纹理苗绣关注仍显粗糙
- **ViT-B/16**：直接切分 16×16 patch，缺乏刺绣纹理预训练与局部归纳偏置，注意力分散

### 4.3 训练损失曲线
- 已生成 7 个网络对比训练损失曲线（50 epochs，纵轴限制 2.5）
- Ours 收敛快且最终损失较低
- ViT-B/16 前期下降慢，需要更多轮次收敛

---

## 5. 已生成的重要文件与图表

### 5.1 目录结构
- `blue_calico_multitask/checkpoints_combined_fused_correct/`：Ours 模型检查点
- `blue_calico_multitask/ablation_confusion_matrices/`：消融实验混淆矩阵
- `blue_calico_multitask/cam_outputs_test/` 等：Grad-CAM 热力图
- `selected_defect_images_600x600/`：挑选的 20 张疵点图像（600×600）
- `localization_outputs_test/` / `localization_outputs_v2/`：疵点定位输出

### 5.2 Word 文档
- 消融实验分析 Word 文档（多个版本 v6/v7/v10/v13/v14）
- 对比实验分析 Word 文档
- 论文正文：`D:/学校资料/论文相关/论文-文字/论文正文`（已在复制文件上修改）

### 5.3 关键图表
- 7 网络 Grad-CAM 对比图（含用户选定样本）
- 对比实验训练损失曲线（50 epochs）
- 三任务识别示例图（英文缩写）
- 纹样分类混淆矩阵（bx/dx/mx/ssx/other）
- 疵点检测图像与掩码（600×600）

---

## 6. 论文写作进展

### 6.1 已完成
- 论文主体内容撰写
- 标题拟定
- 参考文献整理（22 条）并核查
- 总结与展望添加
- 对比实验与消融实验引用标注
- 参考文献一致性检查：
  - [1] 已改为英文
  - [13] DINOv2 已改为 TMLR 2024 正式期刊版
  - [18] 已替换为 EfficientNet 2019
  - [17] 与 [18] 重复问题已解决
- 对比实验图和文字分析（含 Grad-CAM 描述）
- 消融实验图和文字分析

### 6.2 论文题目方向
围绕“三分支动态多任务网络用于贵州苗绣真伪鉴别、纹样识别与疵点检测”拟定。

### 6.3 参考文献（22 条，已核查）

| 编号 | 文献 | 类型 | 状态 |
|------|------|------|------|
| [1] | Gu, M. M. & Patkin, J. Heritage and identity... | Linguistics and Education 2013 | 已确认 |
| [2] | Zhang et al. Identification of Miao embroidery... | Autex Research Journal 2021 | 已确认 |
| [3] | Guan et al. Automatic embroidery texture synthesis... | The Visual Computer 2021 | 已确认 |
| [4] | Zhong et al. Restoring intricate Miao embroidery patterns... | The Visual Computer 2025 | 已确认 |
| [5] | Liu & Zhou. Innovative design of Chinese traditional textile patterns... | HCI 2022 (Springer) | 已确认 |
| [6] | Zhu & Zhu. Research on Shen embroidery... | Scientific Reports 2024 | 已确认 |
| [7] | Zhuo et al. Combined query embroidery image retrieval... | Scientific Reports 2024 | 已确认 |
| [8] | Atoun et al. Face anti-spoofing... | IJCB 2017 | 已确认 |
| [9] | Özkıper et al. Fingerprint liveness detection... | FiCloud 2022 | 已确认 |
| [10] | Jing et al. Mobile-Unet... | Textile Research Journal 2022 | 已确认 |
| [11] | Tao et al. Recent advances in deep learning for surface defect detection... | IEEE TIM 2021 | 已确认 |
| [12] | Luo et al. Automated visual defect detection for flat steel surface... | IEEE TIM 2020 | 已确认 |
| [13] | Oquab et al. DINOv2... | TMLR 2024 | 已改为正式版 |
| [14] | Hu et al. LoRA... | ICLR 2022 | 已确认 |
| [15] | He et al. Deep residual learning... | CVPR 2016 | 已确认 |
| [16] | Deng et al. ArcFace... | CVPR 2019 | 已确认 |
| [17] | Kendall et al. Multi-task learning using uncertainty... | CVPR 2018 | 已确认 |
| [18] | Tan & Le. EfficientNet... | ICML 2019 | 已确认 |
| [19] | Dosovitskiy et al. An image is worth 16x16 words... | ICLR 2021 | 已确认 |
| [20] | Zhao et al. Deep learning-based pattern classification for embroidery in Asia... | npj Heritage Science 2026 | 已确认 |
| [21] | Sun et al. Deep learning-based recognition of Jin Cang embroidery stitches... | Mathematics 2026 | 已确认 |
| [22] | Yu & Wang. MambaOut... | CVPR 2025 | 已确认 |

**注意**：[10] 作者名 `R?TSCH` 应改为 `Rätsch`；[11] 和 [12] 可补全 DOI。

---

## 7. 关键概念解释

### 7.1 AUC 原理
AUC（Area Under the ROC Curve）衡量的是模型将正样本排在负样本前面的能力。对于真伪鉴别、疵点检测等二分类任务，AUC 对类别不平衡不敏感，因此比准确率更能反映模型的排序能力。

### 7.2 验证集为什么有 loss
验证集虽然不参与反向传播，但会计算 loss 用于：
- 监控模型是否过拟合
- 早停（early stopping）
- 选择最佳 epoch 的模型

---

## 8. 遗留事项与建议

1. **伪品真实性**：如果可能，补充少量真实伪品样本进行独立测试，增强论文说服力。
2. **参考文献格式**：统一 [11]、[12] 的 DOI，修正 [10] 作者名拼写。
3. **论文最终检查**：
   - 检查“苗绣”与“蓝印花布”表述是否统一
   - 检查图表编号与正文引用是否一致
   - 检查公式符号说明是否完整
4. **开源/软件**：网络已封装为软件（Android 客户端），相关位置可进一步整理。

---

## 9. 相关路径速查

| 用途 | 路径 |
|------|------|
| 项目根目录 | `D:/song/kimi_DEMO` |
| 核心代码 | `D:/song/kimi_DEMO/blue_calico_multitask/` |
| 扩增数据集 | `D:/song/kimi_DEMO/blue_calico_multitask/data/guizhou_embroidery/` |
| 正确划分数据集 | `D:/song/kimi_DEMO/blue_calico_multitask/data/guizhou_embroidery_correct/` |
| 网络描述文档 | `D:/song/kimi_DEMO/blue_calico_multitask/network_description.md` |
| 论文正文 | `D:/学校资料/论文相关/论文-文字/论文正文` |
| 原始数据集 | `D:/song/newdata_en` |

---

*本文件由 Kimi 根据项目对话整理生成，用于项目回顾与后续参考。*
