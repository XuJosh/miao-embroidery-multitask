# combined_fused_correct：面向苗绣多任务识别的融合网络设计与贡献

> 本稿对应论文“研究方法 / 模型设计”章节，结合苗绣分类的 5 个核心难点，详细说明 `combined_fused_correct` 网络（即 `EmbroideryNetCombined(auth_defect_use_fused=True, use_patch_transformer=True, use_arcface=True)`）的结构、每一层的作用以及各分支如何解决具体问题。

---

## 1. 从苗绣难点到网络设计动机

苗绣图像的鉴别与识别面临以下相互交织的困难：

| 难点 | 核心表现 | 对网络设计的要求 |
|---|---|---|
| (6) 纹理细微、缺陷尺度小 | 针脚、丝线、光泽变化在 224×224 输入下非常细小，真伪差异和缺陷往往只是局部纹理异常 | 需要能够保留高分辨率局部细节的特征提取器，并在小尺度上捕捉异常 |
| (7) 同类纹样差异大、异类共享 motif | 同一类苗绣在构图、颜色、尺度上变化大；不同类别又共享花卉、动物、几何等局部元素 | 需要全局语义理解 + 局部结构建模，抑制噪声并强化类别判别边界 |
| (8) 缺陷与正常手工噪声难区分 | 手工绣品本身存在不均匀、毛刺、线头等“自然噪声” | 需要“正常纹样应该长什么样”的全局先验，才能区分正常波动与真实缺陷 |
| (9) 伪品与真品边界模糊 | 伪品只是对真品的轻度退化，视觉上非常接近真品 | 需要细粒度度量学习，扩大类间、压缩类内角度 margin |
| (10) 三个任务目标不同但特征可共享 | 伪鉴别和缺陷检测都关注“异常区域”；纹样分类提供正常先验 | 需要在统一特征空间中联合优化三个任务，同时保留任务-specific 头 |

针对上述难点，本文提出 **`combined_fused_correct`**：一个**三分支特征提取 + 统一融合 + 任务特定头 + ArcFace 边际损失 + 动态多任务损失**的端到端网络。其设计逻辑可概括为：

- **ResNet50** 负责局部纹理与细观针脚特征，解决难点 (6)(8)；
- **DINOv2-LoRA** 提供经过大规模无监督预训练的全局视觉语义，解决难点 (7)；
- **Patch Transformer** 在 DINOv2 patch token 序列上建模空间关系，解决难点 (7)(10)；
- **统一融合层** 让三个任务共享一个 256 维特征空间，实现“正常先验”与“异常检测”的相互约束，解决难点 (8)(10)；
- **ArcFace 头** 通过加性角度边际放大细粒度类别差异，解决难点 (7)(9)；
- **Kendall 不确定性多任务损失** 自动平衡真伪、纹样、疵点三个损失，解决难点 (10)。

---

## 2. 网络总体结构

`combined_fused_correct` 的输入为一张 RGB 图像 \(x \in \mathbb{R}^{B \times 3 \times 224 \times 224}\)，输出为三个任务的对数几率（logits）：

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: 3 × 224 × 224                                            │
├──────────────┬─────────────────────┬────────────────────────────┤
│ ResNet50     │ DINOv2_vits14 + LoRA│ Patch Transformer            │
│ (local CNN)  │ (global ViT)        │ (patch-level sequence)       │
│ ↓            │ ↓                   │ ↓                            │
│ res_vec: 2048│ dinov2_cls: 384     │ seq_vec: 384                 │
└──────────────┴─────────────────────┴────────────────────────────┘
│                         Concat + pattern_fusion MLP              │
│                              ↓ fused: 256                        │
├─────────────────┬───────────────────┬────────────────────────────┤
│ auth_head       │ pattern_head      │ defect_head                │
│ ArcFace(256,2)  │ ArcFace(256,5)    │ MLP(256→256→2)             │
└─────────────────┴───────────────────┴────────────────────────────┘
```

关键超参数：

- `dinov2_name='dinov2_vits14'`，`lora_r=8`，`lora_alpha=16`，`lora_dropout=0.05`；
- `seq_layers=2`，`seq_nhead=8`，`seq_dim_feedforward=512`；
- `dropout=0.3`；
- `use_arcface=True`，`arcface_s=30.0`，`arcface_m=0.30`；
- `auth_defect_use_fused=True`（即 auth、defect 与 pattern 共享 fused 特征）。

---

## 3. 各分支逐层详解

### 3.1 ResNet50Backbone：局部纹理与细观异常提取

```python
class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        resnet = torchvision.models.resnet50(weights=weights)
        # 逐层保留
        self.conv1 = resnet.conv1      # 7×7, stride 2, 64
        self.bn1  = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 3×3, stride 2
        self.layer1 = resnet.layer1    # 256 channels, 56×56
        self.layer2 = resnet.layer2    # 512 channels, 28×28
        self.layer3 = resnet.layer3    # 1024 channels, 14×14
        self.layer4 = resnet.layer4    # 2048 channels, 7×7
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 64×112×112
        x = self.maxpool(x)                     # 64×56×56
        x = self.layer1(x)                      # 256×56×56
        x = self.layer2(x)                      # 512×28×28
        x = self.layer3(x)                      # 1024×14×14
        feat = self.layer4(x)                   # 2048×7×7
        vec = self.avgpool(feat).view(B, -1)    # (B, 2048)
        return feat, vec
```

**作用与对应难点**：

- ResNet50 在 ImageNet-1K-V2 上预训练，具备丰富的自然图像纹理先验；
- `layer4` 输出 2048×7×7 的特征图，对应原图 32×32 的局部感受野，能够保留针脚、丝线走向、光泽变化等细观纹理，直接对应难点 **(6)**；
- 全局平均池化后的 `res_vec`（2048 维）作为局部纹理的紧凑向量，供后续融合使用；
- 在缺陷检测中，局部 CNN 特征是发现“局部纹理异常”的基础，对应难点 **(8)**。

### 3.2 DINOv2-LoRA：全局语义与 motif 理解

```python
class DinoV2LoRA(nn.Module):
    def __init__(self, model_name='dinov2_vits14',
                 lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 target_modules=['qkv']):
        # 加载 dinov2_vits14 (embed_dim=384, 12 blocks, patch size=14)
        self.dinov2 = torch.hub.load('facebookresearch/dinov2',
                                     model_name, pretrained=True)
        # 冻结基座，注入 LoRA
        config = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                            target_modules=target_modules,
                            lora_dropout=lora_dropout, bias='none')
        self.dinov2 = get_peft_model(self.dinov2, config)

    def forward(self, x):
        out = self._forward_features(x)
        cls_token = out['x_norm_clstoken']       # (B, 384)
        patch_tokens = out['x_norm_patchtokens'] # (B, 256, 384)
        return cls_token, patch_tokens
```

**输入输出尺寸**：

- 输入 \(x\)：\(B \times 3 \times 224 \times 224\)；
- patch size = 14，因此 token 数为 \((224/14)^2 = 256\)；
- `cls_token`：\(B \times 384\)，代表整幅图像的全局语义；
- `patch_tokens`：\(B \times 256 \times 384\)，代表每个图像块的空间特征。

**作用与对应难点**：

- DINOv2 在 1.42 亿张图像上通过自监督训练，其特征对物体部分、场景几何具有良好刻画<sup>[18]</sup>；
- 对于苗绣，DINOv2 的全局 cls token 能够把握“花卉/动物/几何 motif”的整体结构，缓解同类纹样构图、颜色、尺度变化大以及异类共享局部 motif 带来的分类边界模糊，对应难点 **(7)**；
- 采用 **LoRA**（低秩适配）而非全量微调：在冻结的 DINOv2 基座上仅训练 qkv 投影的低秩增量（r=8, α=16），既保留预训练知识，又避免在苗绣小样本上过度拟合；
- 训练时支持梯度检查点（gradient checkpointing），降低显存占用。

### 3.3 Patch Transformer：patch 之间的空间关系建模

```python
class PatchTransformer(nn.Module):
    def __init__(self, d_model=384, n_layers=2, nhead=8,
                 dim_feedforward=512, dropout=0.3, max_len=512):
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=384, nhead=8, dim_feedforward=512,
            dropout=0.3, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, patch_tokens):
        b, n, d = patch_tokens.shape          # (B, 256, 384)
        x = patch_tokens + self.pos_embed[:, :n, :]
        out = self.encoder(x)                  # (B, 256, 384)
        return out.mean(dim=1)                 # (B, 384)
```

**作用与对应难点**：

- DINOv2 的 cls token 虽然全局，但对“多个 motif 如何组合成某一类纹样”的显式空间关系建模不足；
- Patch Transformer 在 256 个 patch token 上运行 2 层自注意力，显式学习“左上方龙纹 + 右下方几何边框 = 某类苗绣”的组合关系，强化纹样分类，对应难点 **(7)**；
- 自注意力还能在全局范围内比较 patch 特征，发现与周围区域不一致的异常 patch，为缺陷检测提供空间上下文，对应难点 **(6)(8)**；
- 平均池化后得到 `seq_vec`（384 维），与 ResNet/DINOv2 特征拼接。

### 3.4 FrequencyEncoder（预留扩展，本工作未启用）

代码中 `use_frequency=False`，因此频域分支未参与 `combined_fused_correct`。该分支通过 FFT 幅度图捕获刺绣针脚的周期性纹理，可作为未来进一步提升细观纹理特征的扩展方向。

---

## 4. 特征融合层：统一表征空间

```python
fused_dim = 2048 + 384 + 384 = 2816
self.pattern_fusion = nn.Sequential(
    nn.Linear(2816, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
    nn.Linear(512, 256),  nn.ReLU(inplace=True), nn.Dropout(0.3),
)

feats = [res_vec, dinov2_cls, seq_vec]
fused = self.pattern_fusion(torch.cat(feats, dim=1))  # (B, 256)
```

**设计要点**：

- **Concat** 将三种互补特征拼接：
  - `res_vec`（2048）：局部纹理、针脚细节；
  - `dinov2_cls`（384）：全局 motif 语义；
  - `seq_vec`（384）：patch 间空间组合关系。
- **两层 MLP** 将 2816 维压缩到 256 维，ReLU + Dropout(0.3) 引入非线性与正则化；
- 256 维 `fused` 是一个**任务共享的紧凑表征**。

**关键参数 `auth_defect_use_fused=True`**：

```python
auth_input   = fused   # 而非 res_vec
defect_input = fused   # 而非 res_vec
```

这意味着真伪鉴别头和疵点检测头不再只依赖 CNN 局部特征，而是能够利用 DINOv2 提供的“正常纹样全局先验”。例如：

- 缺陷检测：知道“正常苗绣应该包含哪些 motif”，才能判断局部异常是“手工噪声”还是“真实疵点”，对应难点 **(8)**；
- 真伪鉴别：全局语义 + 局部异常共同决定“伪品”，对应难点 **(9)(10)**。

---

## 5. 多任务头设计

### 5.1 ArcFace 边际分类头

```python
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        cos = F.linear(F.normalize(x), F.normalize(self.weight))
        if label is None:
            return self.s * cos
        # 仅对目标类别加 margin
        one_hot = torch.zeros_like(cos).scatter_(1, label.view(-1,1), 1.0)
        sin = torch.sqrt(1.0 - cos.pow(2) + 1e-6)
        phi = cos * cos(m) - sin * sin(m)
        output = one_hot * phi + (1 - one_hot) * cos
        return self.s * output
```

**作用**：ArcFace 将分类问题转化为角度空间中的度量学习<sup>[19]</sup>。通过给目标类别施加角度 margin \(m=0.30\)，它强制同类样本在特征空间中更紧凑、异类样本更分散。对于苗绣：

- **纹样分类**：5 类苗绣往往共享花卉/动物 motif，ArcFace 能拉开易混淆类别的决策边界，对应难点 **(7)**；
- **真伪鉴别**：真品/伪品视觉边界模糊，ArcFace 的 margin 能放大细微差异，对应难点 **(9)**。

### 5.2 真伪鉴别头

```python
self.auth_head = ArcMarginProduct(in_features=256, out_features=2,
                                  s=30.0, m=0.30)
```

- 输入：256 维 fused 特征；
- 输出：2 维 logits（真品 / 伪作）；
- 训练时需要 `labels['auth']` 计算 margin；推理时 `label=None`，输出为归一化余弦相似度乘以 scale，再经 Softmax 得到概率。

### 5.3 纹样分类头

```python
self.pattern_head = ArcMarginProduct(in_features=256, out_features=5,
                                     s=30.0, m=0.30)
```

- 输入：256 维 fused 特征；
- 输出：5 维 logits，对应 辫绣、堆绣、马尾绣、其他、数纱马尾绣；
- 与 auth 头共享 fused 空间，使“纹样先验”能够约束真伪/缺陷判断。

### 5.4 疵点检测头

```python
self.defect_head = nn.Sequential(
    nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
    nn.Linear(256, 2),
)
```

- 输入：256 维 fused 特征；
- 输出：2 维 logits（有疵点 / 无疵点）；
- 不采用 ArcFace，因为缺陷检测是异常检测风格，关注“与正常模式的偏离”而非类别 margin；
- 使用 fused 特征而非纯 CNN 特征，使疵点判断能够参考纹样先验，对应难点 **(8)**。

---

## 6. 多任务损失与训练策略

### 6.1 单任务损失

三个任务均使用交叉熵损失：

\[
\mathcal{L}_{\text{auth}} = \text{CE}(\hat{y}_{\text{auth}}, y_{\text{auth}})
\]
\[
\mathcal{L}_{\text{pattern}} = \text{CE}(\hat{y}_{\text{pattern}}, y_{\text{pattern}})
\]
\[
\mathcal{L}_{\text{defect}} = \text{CE}(\hat{y}_{\text{defect}}, y_{\text{defect}})
\]

### 6.2 动态多任务损失

为避免手动调权，采用 Kendall 等提出的基于同方差不确定性的自动加权<sup>[21]</sup>：

```python
self.log_vars = nn.Parameter(torch.zeros(3))  # 可学习

losses = torch.stack([auth_loss, pattern_loss, defect_loss])
precisions = torch.exp(-self.log_vars)
total = torch.sum(precisions * losses + self.log_vars)
```

**物理意义**：

- \(\log \sigma_i^2\) 越大，表示任务 \(i\) 的不确定性越高，其有效权重 \(e^{-\log \sigma_i^2}\) 越小；
- 网络自动降低“难/噪声大”任务的权重，提升“可靠”任务的权重；
- 这使得三个任务目标不同但能在同一框架下稳定优化，对应难点 **(10)**。

---

## 7. 推理流程

```
输入图像 x (3×224×224)
    ↓
[ResNet50] ──→ res_vec (2048)
[DINOv2-LoRA] ──→ dinov2_cls (384) + patch_tokens (256×384)
    ↓
[Patch Transformer] ──→ seq_vec (384)
    ↓
Concat([res_vec, dinov2_cls, seq_vec]) → MLP → fused (256)
    ↓
auth_head(fused)     → 真/伪概率
pattern_head(fused)  → 5 类纹样概率
defect_head(fused)   → 有/无疵点概率
```

在实际应用（GUI/Android）中，遵循“**先鉴真伪，真品才展示纹样与疵点**”的推理策略：若 `auth` 判定为伪作，则跳过纹样与疵点输出；若为真品，则进一步输出纹样类别与疵点状态。

---

## 8. 难点-模块对应总表

| 难点 | 负责模块 | 解决机制 |
|---|---|---|
| (6) 纹理细微、缺陷尺度小 | ResNet50 + DINOv2 patch tokens + fused defect head | ResNet 保留 7×7 局部特征；DINOv2 14×14 patch 提供细观块特征；融合后在 256 维空间中判别局部异常 |
| (7) 同类差异大、异类共享 motif | DINOv2-LoRA cls + Patch Transformer + ArcFace pattern head | 全局语义把握 motif 结构；Patch Transformer 建模空间组合；ArcFace 拉开易混淆类别边界 |
| (8) 缺陷与正常手工噪声难区分 | 统一 fused 特征 + defect head | 纹样分类头提供“正常纹样”先验，缺陷头在共享空间中学习“偏离正常”的模式 |
| (9) 真伪边界模糊 | fused 特征 + ArcFace auth head | 角度 margin 强制真/伪样本在特征空间中分离，放大轻度退化带来的细微差异 |
| (10) 三任务目标不同但特征可共享 | `auth_defect_use_fused=True` + 动态多任务损失 | auth/defect/pattern 共享 fused 表示；Kendall 损失自动平衡不同任务梯度 |

---

## 9. 模型规模与性能

`combined_fused_correct` 在未剪枝状态下的关键指标如下（在苗绣自建测试集上）：

| 指标 | 未剪枝 | 30% L1 非结构化剪枝后 |
|---|---|---|
| 非零参数量 | 49.5 M | 34.9 M |
| auth_acc | 0.969 | 0.933 |
| auth_auc | 0.990 | — |
| pat_acc | 0.996 | 0.957 |
| def_acc | 0.931 | 0.875 |
| def_auc | 0.948 | 0.917 |

**解读**：

- 高 auth_auc（0.990）说明 fused 特征 + ArcFace 对真伪这一细粒度二分类具有极强的判别能力；
- pat_acc 接近 0.996 说明三分支融合 + ArcFace 对 5 类纹样分类非常有效；
- 即使剪枝 30% 后，各项指标下降有限，验证了该架构存在较大的冗余压缩空间，适合后续 PC/Android 端部署。

---

## 10. 与单任务/其他架构的区别（讲好“故事”）

**故事线**：苗绣不是普通图像分类问题，而是“在巨大类内变化、共享 motif、手工噪声、细粒度真伪退化”下的**多任务细粒度视觉理解**问题。传统单任务 CNN 只能看到局部纹理，无法回答“这幅绣品整体属于哪类纹样”“哪里偏离了正常手工特征”；纯 DINOv2 虽有全局语义，却缺乏对苗绣特有针脚异常和局部 motif 组合的显式建模；ArcFace 若单独使用也无法利用纹样先验来辅助真伪鉴别。

`combined_fused_correct` 的**核心贡献**在于：

1. **互补三分支**：CNN 看细节、DINOv2 看语义、Transformer 看结构，三者取长补短；
2. **统一融合 + 全任务共享**：通过 `auth_defect_use_fused=True`，让真伪鉴别和疵点检测也能“看懂”纹样语义，实现“以正常纹样先验判异常”的类人推理；
3. **ArcFace 细粒度约束**：把纹样分类和真伪鉴别从普通 softmax 提升为角度度量学习，显著改善边界模糊问题；
4. **动态多任务协同**：Kendall 不确定性加权让三个任务在同一网络中各取所需，避免单一任务主导梯度。

因此，该网络不仅是一个“三个头拼在一起”的多任务模型，而是一个**围绕苗绣图像本质难点设计的、任务间相互增强的细粒度识别系统**。

---

## 参考文献（本节新引）

[16] HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778. DOI: 10.1109/CVPR.2016.90.

[17] DOSOVITSKIY A, BEYER L, KOLESNIKOV A, et al. An image is worth 16×16 words: Transformers for image recognition at scale[C]//International Conference on Learning Representations. 2021.

[18] OQUAB M, DARCET T, MOUTAKANNI T, et al. DINOv2: Learning robust visual features without supervision[J]. arXiv preprint arXiv:2304.07193, 2023.

[19] DENG J, GUO J, XUE N, et al. ArcFace: Additive angular margin loss for deep face recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 4690-4699. DOI: 10.1109/CVPR.2019.00482.

[20] ZHANG Y, YANG Q. A survey on multi-task learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2022, 34(12): 5586-5609. DOI: 10.1109/TKDE.2021.3079203.

[21] KENDALL A, GAL Y, CIPOLLA R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 7482-7491. DOI: 10.1109/CVPR.2018.00781.
