# 本文方法

## 1.1 网络结构阐述

鉴于苗绣图像存在纹理细微、同类纹样差异大、疵点与手工噪声易混淆、真伪边界模糊以及三个任务特征可共享等难点，本文提出了一种面向苗绣多任务识别的融合网络 **EmbroideryNetCombined**，其整体结构如图 X 所示。该网络摒弃了单一 CNN 或单一 Transformer 的建模方式，采用“三分支 + 统一融合 + 多任务头”的架构设计：

- **局部纹理分支（Local Texture Branch）**：以 ResNet50 为骨干，负责提取针脚、丝线走向等局部纹理特征；
- **全局语义分支（Global Semantic Branch）**：以 DINOv2-LoRA 为骨干，负责学习纹样的全局语义与高层风格表征；
- **结构关系分支（Structural Relation Branch）**：以 Patch Transformer 为序列建模器，负责刻画 patch 之间的空间布局关系。

输入图像 \(x \in \mathbb{R}^{3 \times 224 \times 224}\) 首先并行送入三个分支。各分支输出的特征维度与物理意义如表 1 所示。

**表 1  EmbroideryNetCombined 三分支输出特征**

| 分支 | 模块 | 输出特征 | 维度 | 物理意义 |
|---|---|---|---|---|
| 局部纹理分支 | ResNet50Backbone | \(F_{\text{tex}}\) | \((B, 2048)\) | 针脚纹理、丝线走向、绣面光泽等局部/中层特征 |
| 全局语义分支 | DINOv2-LoRA | \(F_{\text{sem}}\) | \((B, 384)\) | 纹样整体语义、风格、颜色分布等全局特征 |
| 全局语义分支 | DINOv2-LoRA | \(P\) | \((B, 256, 384)\) | 16×16 空间网格上的局部语义 token |
| 结构关系分支 | PatchTransformer | \(F_{\text{str}}\) | \((B, 384)\) | patch 之间的空间布局与上下文关系 |

三个分支的输出经过拼接后送入融合 MLP。设三个分支特征分别为 \(F_{\text{tex}}\)、\(F_{\text{sem}}\) 与 \(F_{\text{str}}\)，则融合过程可表示为

$$
F_{\text{fused}} = \text{MLP}_{\text{fusion}}\bigl([F_{\text{tex}}; F_{\text{sem}}; F_{\text{str}}]\bigr),
$$

其中 \([\cdot; \cdot]\) 表示通道拼接操作。融合 MLP 的具体结构为

$$
F_{\text{fused}} = \text{Dropout}\bigl(\text{ReLU}(W_2 \cdot \text{Dropout}(\text{ReLU}(W_1 [F_{\text{tex}}; F_{\text{sem}}; F_{\text{str}}] + b_1)) + b_2)\bigr),
$$

其中 \(W_1 \in \mathbb{R}^{512 \times 2816}\)、\(b_1 \in \mathbb{R}^{512}\)、\(W_2 \in \mathbb{R}^{256 \times 512}\)、\(b_2 \in \mathbb{R}^{256}\)。最终 \(F_{\text{fused}} \in \mathbb{R}^{256}\) 作为统一表征，并行送入三个任务头：

- `auth_head`：基于 ArcFace 的真伪鉴别头，输出 \(z_{\text{auth}} \in \mathbb{R}^{2}\)；
- `pattern_head`：基于 ArcFace 的纹样分类头，输出 \(z_{\text{pattern}} \in \mathbb{R}^{5}\)；
- `defect_head`：基于全连接层的疵点检测头，输出 \(z_{\text{defect}} \in \mathbb{R}^{2}\)（`cls` 模式）。

---

## 1.2 局部纹理分支

局部纹理分支以 **ResNet50**（He 等, 2016）为骨干网络。ResNet50 的基本组成单元为 Bottleneck 残差块，每个残差块包含三层卷积：\(1\times 1\) 降维卷积、\(3\times 3\) 空间卷积和 \(1\times 1\) 升维卷积，并在卷积之间插入批量归一化（Batch Normalization）与 ReLU 激活函数。残差块通过跳跃连接将输入直接加到卷积输出上，有效缓解了深层网络的梯度消失问题，其前向传播可表示为

$$
y = \mathcal{F}(x, \{W_i\}) + x,
$$

其中 \(x\) 为残差块输入，\(\mathcal{F}(\cdot)\) 表示由 \(1\times 1\)、\(3\times 3\)、\(1\times 1\) 卷积与 BN-ReLU 组成的非线性映射，\(y\) 为残差块输出。

本文使用的 ResNet50Backbone 由以下层次组成，各层输出特征图的尺寸与通道数如表 2 所示。

**表 2  ResNet50Backbone 各层输出**

| 层级 | 操作 | 输出尺寸 | 输出通道 | 提取的特征 |
|---|---|---|---|---|
| conv1 | \(7\times 7\) 卷积，stride=2，padding=3 | \(112 \times 112\) | 64 | 低层边缘、颜色、亮度过渡 |
| bn1+relu+maxpool | \(3\times 3\) 最大池化，stride=2 | \(56 \times 56\) | 64 | 初步下采样特征 |
| layer1 | 3 个 Bottleneck，stride=1 | \(56 \times 56\) | 256 | 短程纹理、局部绣面结构 |
| layer2 | 4 个 Bottleneck，stride=2 | \(28 \times 28\) | 512 | 中层纹理模式、针法基元 |
| layer3 | 6 个 Bottleneck，stride=2 | \(14 \times 14\) | 1024 | 较大范围的纹样部件、区域语义 |
| layer4 | 3 个 Bottleneck，stride=2 | \(7 \times 7\) | 2048 | 高层语义 + 保留的局部空间信息 |
| global average pooling | 全局平均池化 | \(1 \times 1\) | 2048 | 全局纹理统计向量 |

具体而言，输入图像 \(x\) 首先经过 `conv1` 和 `maxpool` 完成初步下采样，得到 \(56\times 56\) 的特征图；随后通过 `layer1` 至 `layer4` 四个残差阶段逐步提取高层语义。每个阶段的第一个残差块会通过 \(1\times 1\) 卷积（stride=2）对跳跃连接进行下采样，从而将空间分辨率减半、通道数加倍。

在苗绣识别任务中，针脚的疏密、丝线的走向、绣面的光泽变化以及机绣与手绣在纹理规则性上的差异，均属于局部纹理信息。`layer3` 与 `layer4` 输出的特征图空间分辨率分别为 \(14\times 14\) 和 \(7\times 7\)，既能覆盖较大感受野以感知纹样部件，又能保留足够的局部结构以区分细微纹理差异。最终通过全局平均池化得到 2048 维的局部纹理向量 \(F_{\text{tex}}\)，参与后续融合。

---

## 1.3 全局语义分支

全局语义分支以 **DINOv2-LoRA** 为核心。DINOv2（Oquab 等, 2023）是一种在大规模自然图像上通过自监督方式预训练的 Vision Transformer，其强大的全局语义表征能力对纹样分类尤为有利。本文采用 `dinov2_vits14` 作为基座，其默认配置为 12 层 Transformer、384 维嵌入、6 个注意力头、MLP 隐藏层维度为 1536。

### 1.3.1 Patch 嵌入与 token 生成

输入图像 \(x \in \mathbb{R}^{3 \times 224 \times 224}\) 首先经过 Patch Embedding 层。该层等价于一个卷积核为 \(14\times 14\)、步长为 14 的二维卷积，将图像划分为 \(16\times 16 = 256\) 个互不重叠的 patch，每个 patch 被映射为一个 384 维向量。因此，patch embedding 的输出为

$$
P_0 \in \mathbb{R}^{256 \times 384}.
$$

随后，DINOv2 在序列最前端拼接一个可学习的 **CLS token** \(c \in \mathbb{R}^{1 \times 384}\)，用于聚合全局信息；同时拼接若干 **register tokens**（默认为 4 个），用于稳定深层特征训练。最终输入 Transformer 的序列为

$$
T = [c; r_1; \dots; r_R; P_0],
$$

其中 \(R\) 为 register token 数量。

### 1.3.2 Transformer 编码与自注意力

DINOv2 由 12 个 Transformer block 组成，每个 block 包含多头自注意力（Multi-Head Self-Attention, MHSA）、前馈网络（Feed-Forward Network, FFN）与层归一化（LayerNorm）。自注意力的计算方式可表示为

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V,
$$

其中 \(Q, K, V \in \mathbb{R}^{N \times d}\) 分别由输入序列经三个不同的线性投影得到，\(d_k = d / h = 384 / 6 = 64\) 为每个注意力头的键向量维度，\(N\) 为序列长度。多头自注意力将 \(h\) 个头的输出拼接后再经线性投影，使模型能够同时关注不同子空间中的依赖关系。

前馈网络由两个线性层夹一个 GELU 激活函数组成，其隐藏层维度为 1536：

$$
\text{FFN}(X) = W_2 \cdot \text{GELU}(W_1 X + b_1) + b_2,
$$

其中 \(W_1 \in \mathbb{R}^{1536 \times 384}\)、\(W_2 \in \mathbb{R}^{384 \times 1536}\)。每个 Transformer block 的输出可表示为

$$
T^{(l+1)} = \text{FFN}\bigl(\text{MHSA}(\text{LN}(T^{(l)}))\bigr) + T^{(l)}.
$$

经过 12 层 Transformer 编码后，序列再经过一次 LayerNorm，得到归一化后的 token 序列。

### 1.3.3 LoRA 适配机制

为了在小样本苗绣数据上高效迁移 DINOv2 的先验知识，本文引入 **LoRA（Low-Rank Adaptation）**（Hu 等, 2021）。LoRA 冻结 DINOv2 的原始参数，仅在每个 Transformer 块的 `qkv` 投影旁注入低秩适配矩阵 \(A\) 与 \(B\)。`qkv` 投影的原始权重 \(W_0 \in \mathbb{R}^{1152 \times 384}\) 被冻结，其输出更新为

$$
W' = W_0 + \frac{\alpha}{r} BA,
$$

其中 \(B \in \mathbb{R}^{1152 \times r}\)、\(A \in \mathbb{R}^{r \times 384}\) 为可训练参数，秩 \(r \ll 384\)。本文设置 \(r=8\)、\(\alpha=16\)，因此等效缩放系数为 \(\alpha / r = 2\)。每个 `qkv` 投影的可训练参数量从 \(1152 \times 384 = 442,368\) 降至 \(1152 \times 8 + 8 \times 384 = 12,288\)，约为原始参数的 2.8%。

### 1.3.4 输出特征

DINOv2-LoRA 最终输出两类特征：

- **CLS token** \(F_{\text{sem}} \in \mathbb{R}^{384}\)：代表整幅图像的全局语义，用于纹样分类与整体风格判断；
- **Patch tokens** \(P \in \mathbb{R}^{256 \times 384}\)：保留空间对应关系的局部语义单元，作为结构关系分支的输入。

全局语义分支使网络能够在高层语义空间中理解“这是哪种苗绣纹样”，并对光照、拍摄角度、背景等无关因素具有较强的鲁棒性。

---

## 1.4 结构关系分支

结构关系分支以 **Patch Transformer** 对 DINOv2 输出的 patch token 序列 \(P\) 进行进一步建模。DINOv2 的 cls token 虽然语义丰富，但在全局池化过程中会损失空间布局信息；而 patch token 之间的相对位置与相互关系对于区分相似纹样、定位疵点以及判断真伪至关重要。

### 1.4.1 位置编码

Patch Transformer 首先为每个 patch token 添加可学习的位置编码，以保留原始图像中的空间位置信息：

$$
Z^{(0)} = P + E_{\text{pos}}[:, :N, :],
$$

其中 \(E_{\text{pos}} \in \mathbb{R}^{1 \times 512 \times 384}\) 为可学习位置编码矩阵，\(N=256\) 为当前 patch 数量。位置编码使模型能够区分“左上角的龙纹”与“右下角的花卉”，从而学习纹样的空间布局。

### 1.4.2 Transformer 编码器

\(Z^{(0)}\) 经过 2 层 8 头 TransformerEncoderLayer，每层均采用 `norm_first=True` 与 GELU 激活函数。每层 TransformerEncoderLayer 的内部计算分为两个子层：

1. **多头自注意力子层**：

$$
Z' = \text{MHSA}(\text{LN}(Z^{(l)})) + Z^{(l)},
$$

其中注意力头数 \(h=8\)，每个头维度 \(d_k = 384 / 8 = 48\)。通过自注意力，每个 patch 都能与全部 256 个 patch 交互，从而捕捉长程空间依赖。

2. **前馈网络子层**：

$$
Z^{(l+1)} = \text{FFN}(\text{LN}(Z')) + Z',
$$

其中 FFN 的结构为 `Linear(384, 512) → GELU → Dropout(0.1) → Linear(512, 384)`。与 DINOv2 不同，Patch Transformer 的 FFN 隐藏层维度仅为 512，参数量更小，专注于苗绣数据集上的布局关系学习。

### 1.4.3 全局平均池化

经过 2 层 Transformer 编码后，输出序列 \(Z^{(2)} \in \mathbb{R}^{256 \times 384}\)。为了得到固定维度的结构关系向量，采用全局平均池化：

$$
F_{\text{str}} = \frac{1}{N}\sum_{i=1}^{N} Z_i^{(2)},
$$

其中 \(N=256\)。最终 \(F_{\text{str}} \in \mathbb{R}^{384}\) 编码了苗绣纹样中局部元素之间的空间配置关系。

### 1.4.4 提取的特征类型

结构关系分支使网络能够：

- **区分同类不同样的纹样**：相同 motif 的不同排列方式会被不同的注意力模式区分；
- **识别疵点**：疵点区域的 patch 与周围正常 patch 在语义上不协调，自注意力会放大这种异常；
- **判断真伪**：机绣作品往往具有更规则的重复模式，而手绣的布局更随机、自然，结构关系分支能够捕捉这种统计差异。

---

## 1.5 ArcFace 任务头

为了增强网络在细粒度纹样分类与真伪鉴别中的判别能力，本文在 `auth_head` 与 `pattern_head` 中引入 **ArcFace（Additive Angular Margin Loss）**（Deng 等, 2019）。ArcFace 通过在特征与分类权重之间的夹角上加入加性间隔，使得同类样本更加紧凑、异类样本更加分散。

### 1.5.1 归一化与角度计算

设融合特征 \(F_{\text{fused}} \in \mathbb{R}^{256}\) 为任务头的输入。对于 `auth_head`，类别数为 2，权重矩阵 \(W_{\text{auth}} \in \mathbb{R}^{2 \times 256}\)；对于 `pattern_head`，类别数为 5，权重矩阵 \(W_{\text{pattern}} \in \mathbb{R}^{5 \times 256}\)。

ArcFace 首先对输入特征与分类权重进行 L2 归一化：

$$
\hat{x} = \frac{x}{\|x\|_2}, \quad \hat{W}_j = \frac{W_j}{\|W_j\|_2}.
$$

样本与第 \(j\) 类的夹角余弦为

$$
\cos \theta_j = \hat{W}_j^{\top} \hat{x}.
$$

### 1.5.2 加性角度间隔

对于目标类别 \(y_i\)，ArcFace 将夹角增加间隔 \(m\)，即

$$
\cos(\theta_{y_i} + m) = \cos \theta_{y_i} \cos m - \sin \theta_{y_i} \sin m.
$$

最终经过尺度因子 \(s\) 缩放后的 logit 为

$$
z_j = \begin{cases}
s \cdot \cos(\theta_{y_i} + m), & j = y_i, \\[6pt]
s \cdot \cos \theta_j, & j \neq y_i.
\end{cases}
$$

本文中设置 \(s=30\)、\(m=0.30\)。尺度因子 \(s\) 放大了特征空间的差异，便于后续 softmax 分类；角度间隔 \(m\) 则强制增大类间夹角，使网络学到的特征在超球面上更具判别性。

### 1.5.3 ArcFace 损失

ArcFace 损失为标准交叉熵损失：

$$
\mathcal{L}_{\text{arc}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{e^{z_{y_i}}}{\sum_{j=1}^{C} e^{z_j}},
$$

其中 \(B\) 为批量大小，\(C\) 为类别数。ArcFace 将分类问题转化为角度度量学习，对纹样边界模糊、真伪样本难以区分的苗绣任务具有显著的促进作用。

---

## 1.6 动态多任务损失函数

由于真伪鉴别、纹样分类与疵点检测三个任务的损失量级与学习难度不同，简单的固定权重求和往往难以取得最优平衡。本文采用 **Kendall 等人（2018）提出的基于同方差不确定性的动态多任务损失** 进行自适应加权。

### 1.6.1 各任务损失定义

设网络输出的 logits 为 \(z_{\text{auth}}\)、\(z_{\text{pattern}}\) 与 \(z_{\text{defect}}\)，对应真实标签为 \(y_{\text{auth}}\)、\(y_{\text{pattern}}\) 与 \(y_{\text{defect}}\)。三个任务均采用交叉熵损失：

$$
L_{\text{auth}} = -\sum_{i} \log \frac{e^{z_{\text{auth}}[y_{\text{auth}}^{(i)}]}}{\sum_{j=0}^{1} e^{z_{\text{auth}}[j]}},
$$

$$
L_{\text{pattern}} = -\sum_{i} \log \frac{e^{z_{\text{pattern}}[y_{\text{pattern}}^{(i)}]}}{\sum_{j=0}^{4} e^{z_{\text{pattern}}[j]}},
$$

$$
L_{\text{defect}} = -\sum_{i} \log \frac{e^{z_{\text{defect}}[y_{\text{defect}}^{(i)}]}}{\sum_{j=0}^{1} e^{z_{\text{defect}}[j]}}.
$$

在 `seg` 模式下，疵点检测可扩展为像素级二分类，使用 `BCEWithLogitsLoss`。

### 1.6.2 不确定性加权

网络为每个任务学习一个可优化的不确定性参数 \(\sigma_i\)。Kendall 等人证明，最优的多任务加权策略可表示为

$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{3} \left( \frac{1}{2\sigma_i^2} L_i + \log \sigma_i \right),
$$

其中 \(i \in \{\text{auth}, \text{pattern}, \text{defect}\}\)。为了避免数值不稳定，代码中对 \(\sigma_i^2\) 进行参数化变换 \(v_i = \log \sigma_i^2\)，并令 \(1/\sigma_i^2 = e^{-v_i}\)，则总损失可改写为

$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{3} \left( e^{-v_i} L_i + v_i \right).
$$

\(v_i\) 初始化为 0，此时三个任务的权重相等。训练过程中，网络通过优化 \(v_i\) 自动调整各任务的相对重要性：

- 当某任务损失 \(L_i\) 较大时，\(e^{-v_i} L_i\) 项对 \(v_i\) 的梯度为负，\(v_i\) 增大，对应权重 \(e^{-v_i}\) 减小；
- 当某任务损失 \(L_i\) 较小时，\(v_i\) 减小，对应权重增大。

这种自适应机制使得网络能够根据三个任务的学习状态动态分配梯度资源，避免简单固定权重下某个任务主导训练或某个任务被忽略的问题。
