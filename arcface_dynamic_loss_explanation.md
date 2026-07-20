# ArcFace 任务头与动态多任务损失函数

## 1. ArcFace 任务头

为了增强细粒度纹样分类与真伪鉴别的判别能力，本文在 `auth_head` 与 `pattern_head` 中引入 **ArcFace（Additive Angular Margin Loss）**。ArcFace 通过对特征与分类权重之间的夹角加入加性间隔，使同类样本在特征空间中更紧凑、异类样本更分散。

设融合特征为 $\boldsymbol{x} \in \mathbb{R}^{256}$，即融合 MLP 输出的任务共享表征；$\boldsymbol{W}$ 为分类头权重矩阵。首先对二者做 L2 归一化，使所有特征与权重都位于单位超球面上：

$$
\hat{\boldsymbol{x}} = \frac{\boldsymbol{x}}{\|\boldsymbol{x}\|_2}, \quad
\hat{\boldsymbol{W}}_j = \frac{\boldsymbol{W}_j}{\|\boldsymbol{W}_j\|_2}
$$

其中：

- $\boldsymbol{x} \in \mathbb{R}^{256}$：融合 MLP 输出的 256 维任务共享表征；
- $\boldsymbol{W}_j$：第 $j$ 类的分类权重向量；
- $\hat{\boldsymbol{x}}$：归一化后的样本特征；
- $\hat{\boldsymbol{W}}_j$：第 $j$ 类的归一化权重向量；
- $j$：类别索引。

样本与第 $j$ 类的夹角余弦为：

$$
\cos \theta_j = \hat{\boldsymbol{W}}_j^{\top} \hat{\boldsymbol{x}}
$$

其中 $\theta_j$ 为特征 $\hat{\boldsymbol{x}}$ 与权重 $\hat{\boldsymbol{W}}_j$ 之间的夹角。

对于目标类别 $y_i$（第 $i$ 个样本的真实标签），ArcFace 将夹角增加间隔 $m$，再经尺度因子 $s$ 缩放后得到改造后的 logit：

$$
z_j =
\begin{cases}
s \cdot \cos(\theta_{y_i} + m), & j = y_i, \\[6pt]
s \cdot \cos \theta_j, & j \neq y_i.
\end{cases}
$$

其中：

- $y_i$：第 $i$ 个样本的目标类别；
- $m$：加性角度边距，用于拉大目标类别与其他类别的夹角；
- $s$：尺度因子，用于放大 logit 数值以便 softmax 学习；
- $z_j$：改造后的第 $j$ 类 logit。

本文设置 $s = 30$、$m = 0.30$。ArcFace 损失即对改造后的 logits 应用标准交叉熵：

$$
\mathcal{L}_{\text{arc}} = -\log \frac{e^{z_{y_i}}}{\sum_{j=1}^{C} e^{z_j}}
$$

其中：

- $C$：类别总数（`auth_head` 中 $C = 2$，`pattern_head` 中 $C = 5$）；
- $z_{y_i}$：目标类别对应的改造后 logit；
- $z_j$：第 $j$ 类的改造后 logit。

ArcFace 将分类问题转化为角度度量学习，对纹样边界模糊、真伪样本难分的苗绣任务具有显著促进作用。

---

## 2. 动态多任务损失函数

由于真伪鉴别、纹样分类与疵点检测三个任务的损失量级与学习难度不同，本文采用 **Kendall 等人提出的基于同方差不确定性的动态多任务损失** 进行自适应加权。

设三个任务的输出 logits 分别为 $z_{\text{auth}}$、$z_{\text{pattern}}$、$z_{\text{defect}}$，对应真实标签为 $y_{\text{auth}}$、$y_{\text{pattern}}$、$y_{\text{defect}}$。三个任务均采用交叉熵损失：

$$
\begin{aligned}
\mathcal{L}_{\text{auth}} &= \text{CE}(z_{\text{auth}}, y_{\text{auth}}), \\
\mathcal{L}_{\text{pattern}} &= \text{CE}(z_{\text{pattern}}, y_{\text{pattern}}), \\
\mathcal{L}_{\text{defect}} &= \text{CE}(z_{\text{defect}}, y_{\text{defect}}).
\end{aligned}
$$

其中：

- $z_{\text{auth}}$、$z_{\text{pattern}}$、$z_{\text{defect}}$：三个任务输出的 logits；
- $y_{\text{auth}}$、$y_{\text{pattern}}$、$y_{\text{defect}}$：三个任务的真实标签；
- $\text{CE}(\cdot, \cdot)$：交叉熵损失。

网络为每个任务学习一个可优化的不确定性参数 $\sigma_i$。总损失定义为：

$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{3} \left( \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i \right)
$$

其中：

- $i \in \{\text{auth}, \text{pattern}, \text{defect}\}$：三个任务索引；
- $\mathcal{L}_i$：第 $i$ 个任务的单任务交叉熵损失；
- $\sigma_i$：第 $i$ 个任务的不确定性。

为了避免数值不稳定，代码中对 $\sigma_i^2$ 进行参数化变换 $v_i = \log \sigma_i^2$，并令 $1/\sigma_i^2 = e^{-v_i}$，则总损失可改写为：

$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{3} \left( e^{-v_i} \mathcal{L}_i + v_i \right)
$$

其中：

- $v_i = \log \sigma_i^2$：可学习的参数化变量；
- $e^{-v_i} = 1 / \sigma_i^2$：对应任务的自动权重。

训练过程中，不确定性高的任务权重 $e^{-v_i}$ 自动减小，不确定性低的任务权重增大，从而避免某个任务主导训练或被忽略。
