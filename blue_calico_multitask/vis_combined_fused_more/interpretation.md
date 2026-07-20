# 新增 10 个样本可视化结果逐图解读

> 说明：以下解读中的“模型认为”“模型关注”均指基于 Grad-CAM / DINOv2 attention 的可解释性分析，不等同于人工标注的缺陷位置。

---

## sample_000_test_00000_a0（auth=0, pattern=0, defect=0）

**原图**：深色底、中心有一朵大型对称星形花的绣片，整体构图非常规整。

### Grad-CAM 解读
- **auth（class=0，仿品）**：热力图几乎覆盖整朵中心花，模型从主体花朵的规整对称性中判断其为仿品。
- **pattern（class=0）**：高亮集中在中心花右侧花瓣，说明右侧花瓣形态是 pattern=0 的关键判别依据。
- **defect（class=0，无缺陷）**：同样关注中心花右侧区域，说明模型在该区域未发现异常。

### DINOv2 attention 解读
- CLS token 分散关注图像上、右、下等多个边缘点，中心花仅被弱关注。
- 这说明 Transformer 对这幅图的整体布局/边缘信息更敏感，与 CNN 高度关注中心花形成互补。

---

## sample_081_test_00081_a0（auth=1, pattern=2, defect=1）

**原图**：红色调传统绣片，纹样密集，包含上下两条横向装饰带和中间主体图案。

### Grad-CAM 解读
- **auth（class=1，真品）**：热力图覆盖中间偏左和下方主体纹样区域，模型综合这些区域判断为真品。
- **pattern（class=2）**：关注中间偏下的一处主体纹样以及底部小图案，说明这两处是 pattern=2 的代表性特征。
- **defect（class=1，有缺陷）**：高亮集中在右侧中间区域，与 auth/pattern 的关注点明显不同，模型认为该处可能存在异常。

### DINOv2 attention 解读
- CLS token 关注中间主体纹样团块以及右下侧局部区域。
- Overlay 显示 Transformer 的注意点与 Grad-CAM 的 defect 区域有一定重合，可能共同指向异常位置。

---

## sample_163_test_00163_a0（auth=0, pattern=1, defect=0）

**原图**：绿色边框刺绣，中央是蓝色菱形镂空，四周有复杂卷草纹边框。

### Grad-CAM 解读
- **auth（class=0，仿品）**：热力图集中在顶部中心和右侧边框，模型从这些区域的规则性判断为仿品。
- **pattern（class=1）**：关注右侧竖边框和底部中心纹样，说明边框装饰是 pattern=1 的识别关键。
- **defect（class=0，无缺陷）**：关注顶部中心区域，与 auth 关注点相近，说明模型在该区域未发现瑕疵。

### DINOv2 attention 解读
- CLS token 强烈关注中央蓝色菱形/星形区域。
- Transformer 把核心语义放在中心几何结构上，而 CNN 更依赖边框特征，二者视角不同。

---

## sample_245_test_00245_a0（auth=1, pattern=4, defect=1）

**原图**：红色底、中央有黑色圆点、上下对称分布花卉/蝴蝶状纹样的绣片。

### Grad-CAM 解读
- **auth（class=1，真品）**：热力图较弱，主要分布在顶部和右下角，真品判断依据较分散。
- **pattern（class=4）**：高度关注中央上方对称花卉（类似蝴蝶形），说明该纹样是 pattern=4 的核心特征。
- **defect（class=1，有缺陷）**：同样集中在中央上方对称区域，但范围更聚焦，模型认为该对称结构处存在异常。

### DINOv2 attention 解读
- CLS token 关注中央上方花卉以及右下角区域。
- 与 Grad-CAM 的 pattern/defect 关注点高度一致，说明 Transformer 也把它视为关键语义区域。

---

## sample_326_test_00326_a0（auth=1, pattern=0, defect=1）

**原图**：黑色底、两条对角线分布的粉色花枝，构图简洁。

### Grad-CAM 解读
- **auth（class=1，真品）**：热力图分散在右上角和右下角边缘，模型可能更多依赖背景或边缘纹理判断真品。
- **pattern（class=0）**：集中关注中央花朵簇，是 pattern=0 的代表性依据。
- **defect（class=1，有缺陷）**：高亮出现在上方花枝和右下角边缘，模型认为这些区域存在异常。

### DINOv2 attention 解读
- CLS token 强烈关注中央花朵簇，并稍带关注上方花枝。
- Transformer 与 CNN 的 pattern 关注点高度一致，同时上方花枝也受到关注，与 defect 判断相呼应。

---

## sample_408_test_00408_a0（auth=1, pattern=3, defect=1）

**原图**：装裱在木质框架中的绣品，主体是一只鸟/鸡、两条鱼和若干蔬菜。

### Grad-CAM 解读
- **auth（class=1，真品）**：热力图集中在底部木雕框架和左下角，模型从装裱框架细节判断为真品。
- **pattern（class=3）**：高度关注画面右侧的鸟/鸡身体，说明该动物形象是 pattern=3 的核心特征。
- **defect（class=1，有缺陷）**：高亮集中在右侧边缘和底部框架，模型认为这些位置存在异常。

### DINOv2 attention 解读
- CLS token 关注鸟/鸡身体和画面中的鱼。
- Transformer 更关注绣品主体内容，而 CNN 的真伪/缺陷判断部分依赖框架边缘，体现不同分支的互补性。

---

## sample_490_test_00490_a0（auth=1, pattern=4, defect=0）

**原图**：圆形绣片，中央是龙/兽纹团花，外圈环绕卷草纹。

### Grad-CAM 解读
- **auth（class=1，真品）**：热力图集中在中央团花和左上角装饰，说明模型从主体团花判断真品。
- **pattern（class=4）**：高度关注左上角外圈装饰纹样，是 pattern=4 的判别依据。
- **defect（class=0，无缺陷）**：热力图集中在中央团花，说明模型在该区域未发现瑕疵。

### DINOv2 attention 解读
- CLS token 关注中央团花偏下位置以及若干分散点。
- Transformer 的注意点相对稀疏，落在中央团花区域，与 CNN 的 auth/defect 关注点一致。

---

## sample_571_test_00571_a0（auth=1, pattern=1, defect=0）

**原图**：红色底、两个人物形象居中、四周有菱形边框和花卉装饰的绣片。

### Grad-CAM 解读
- **auth（class=1，真品）**：热力图覆盖右侧人物头部/身体以及底部中心装饰，综合判断为真品。
- **pattern（class=1）**：高度关注右侧人物头部和上半身，是 pattern=1 的核心特征。
- **defect（class=0，无缺陷）**：关注底部中心装饰区域，说明模型在该处未发现异常。

### DINOv2 attention 解读
- CLS token 关注右侧人物头部和顶部中心区域。
- 与 Grad-CAM 的 pattern 关注点一致，均把右侧人物作为核心语义对象。

---

## sample_653_test_00653_a0（auth=1, pattern=1, defect=1）

**原图**：一件刺绣头饰/面具，纹样由大量螺旋和圆形图案组成，右下角带有局部放大 inset。

### Grad-CAM 解读
- **auth（class=1，真品）**：热力图集中在左侧螺旋纹样区域，模型从这里判断为真品。
- **pattern（class=1）**：同样关注左侧纹样，说明左侧装饰是 pattern=1 的关键。
- **defect（class=1，有缺陷）**：高亮集中在左上角和右下角 inset 区域，模型认为这些局部区域存在异常。

### DINOv2 attention 解读
- CLS token 关注左下角和底部几处螺旋纹样。
- Transformer 的注意点与 CNN 的 auth/pattern 区域同属左侧，但具体位置略有差异，可能捕捉到不同尺度的纹理信息。

---

## sample_735_test_00735_a0（auth=0, pattern=2, defect=0）

**原图**：红色底、左右对称的两个椭圆形纹样（类似蝴蝶或鱼形），顶部有装饰图案。

### Grad-CAM 解读
- **auth（class=0，仿品）**：热力图集中在顶部中心装饰，模型从顶部装饰的规律性判断为仿品。
- **pattern（class=2）**：关注右侧椭圆形纹样，说明右侧纹样是 pattern=2 的判别依据。
- **defect（class=0，无缺陷）**：同样关注右侧椭圆形纹样，说明模型在该区域未发现异常。

### DINOv2 attention 解读
- CLS token 分散关注左右两个椭圆形纹样以及底部多个点。
- Transformer 对两侧对称结构均有注意，而 CNN 的 pattern/defect 更偏向右侧，显示二者提取了互补的左右对称信息。

---

## 总体观察

1. **Auth 任务**：真品样本多关注主体纹样或装裱细节；仿品样本则倾向于关注规整对称区域或边缘，可能对应机器仿制品的纹理规律。
2. **Pattern 任务**：热力图明显集中在各类别最具代表性的主体图案上，如人物、动物、花卉、边框等。
3. **Defect 任务**：有缺陷样本常出现局部高亮区域；无缺陷样本则更多关注主体图案本身。
4. **DINOv2 vs Grad-CAM**：Transformer attention 通常更稀疏、聚焦，有时落在 Grad-CAM 未覆盖的次要区域，二者形成互补。
