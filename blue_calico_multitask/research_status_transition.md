# 贵州苗绣背景段后的“国内外研究现状”转接稿

> 说明：本稿用于承接“贵州苗绣背景/概述”段落，自然过渡到国内外研究现状综述。文内上标编号为本文档临时编号，插入正式论文时请按整体参考文献顺序重新编号。

---

## 一、可直接插入论文的转接段落


随着深度学习在计算机视觉领域的突破，非物质文化遗产的数字化保护与智能识别逐渐成为研究热点。在苗绣及刺绣相关领域，学者们已从纹样分类、分割、检索、修复以及疵点检测等角度展开探索。国际上，ZHANG 等<sup>[1]</sup>较早将 Inception-v4 卷积神经网络引入黔东南苗绣分类，在自建五类数据集上取得了 98.88% 的识别精度；GUAN 等<sup>[2]</sup>提出基于纹理合成的自动刺绣图案生成方法，用于服装设计与在线展示；ZHONG 等<sup>[3]</sup>则结合生成对抗网络与 U-Net，对破损苗绣纹样进行修复。国内方面，陈世婕等<sup>[4]</sup>针对苗绣绣片提出多尺度语义分割网络，提升了纹样提取与数字化存档能力；张银建等<sup>[5]</sup>利用 Stable Diffusion 数据增强与改进 ResNet-18 实现苗绣图像分类；朱佳俊等<sup>[6]</sup>将 MobileNet V1 迁移学习用于沈绣识别，准确率达 98.45%；卓鑫震等<sup>[7]</sup>提出融合增强 CNN 与 Blend-Transformer 的刺绣图像检索方法；陈安琦等<sup>[8]</sup>以 Xception 网络对羌绣图案进行边缘检测与矢量化。上述研究验证了 CNN、Transformer 及生成模型在刺绣图像分析中的有效性。

然而，现有工作大多针对单一任务，鲜有将“真伪鉴别—纹样识别—疵点检测”纳入统一框架的研究。在真伪鉴别方面，人脸活体检测与指纹活体检测等 Presentation Attack Detection 研究<sup>[9,10]</sup>为“真/伪”二分类提供了可借鉴的深度监督范式，但尚未见用于苗绣真伪判别的系统报道；在疵点检测方面，Mobile-Unet<sup>[11]</sup>等轻量网络已在织物疵点检测中取得优异性能，陶显等<sup>[12]</sup>、罗东亮等<sup>[13]</sup>对基于深度学习的表面缺陷检测方法进行了系统综述，为刺绣疵点检测提供了方法基础。综上，如何在保护苗绣本体特征的同时，构建多任务、高精度且可部署的苗绣智能识别模型，仍是亟待解决的课题。基于此，本文提出一种融合 ResNet-50、DINOv2 与 Patch Transformer 的端到端多任务网络，实现苗绣真伪鉴别、纹样分类与疵点检测的联合推理，并通过模型剪枝与跨平台封装探索其实用化部署。

---

## 二、对应参考文献（GB/T 7714—2015 格式）

[1] ZHANG C N, WU S, CHEN J H. Identification of Miao embroidery in southeast Guizhou province of China based on convolution neural network[J]. Autex Research Journal, 2021, 21(2): 198-206. DOI: 10.2478/aut-2020-0063.

[2] GUAN X Y, LUO L K, LI H L, et al. Automatic embroidery texture synthesis for garment design and online display[J]. The Visual Computer, 2021, 37(9-11): 2553-2565. DOI: 10.1007/s00371-021-02216-0.

[3] ZHONG C, YU X M, XIA H, et al. Restoring intricate Miao embroidery patterns: a GAN-based U-Net with spatial-channel attention[J]. The Visual Computer, 2025, 41(10): 7521-7533. DOI: 10.1007/s00371-025-03821-z.

[4] 陈世婕, 王卫星, 彭莉. 基于多尺度网络的苗绣绣片纹样分割算法研究[J]. 计算机技术与发展, 2023, 33(11): 149-155.

[5] 张银建, 杨邦勤, 陈研, 等. 基于稳定扩散模型的苗绣图像分类研究[J]. 信息技术与信息化, 2024(12): 5-12. DOI: 10.3969/j.issn.1672-9528.2024.12.001.

[6] ZHU J J, ZHU C Y. Research on the innovative application of Shen embroidery cultural heritage based on convolutional neural network[J]. Scientific Reports, 2024, 14(1): 9574. DOI: 10.1038/s41598-024-60121-7.

[7] ZHUO X Z, HUANG D H, LIN Y, et al. Combined query embroidery image retrieval based on enhanced CNN and blend transformer[J]. Scientific Reports, 2024, 14(1): 27518. DOI: 10.1038/s41598-024-79012-y.

[8] CHEN A Q, PENG Y C, LI M, et al. Generate vector graphics of fine-grained pattern based on the Xception edge detection[J]. PLoS One, 2025, 20(6): e0318930. DOI: 10.1371/journal.pone.0318930.

[9] ATOUN Y, LIU Y, JOURABLOO A, et al. Face anti-spoofing using patch and depth-based CNNs[C]//2017 IEEE International Joint Conference on Biometrics (IJCB). IEEE, 2017: 319-328. DOI: 10.1109/BTAS.2017.8272713.

[10] ÖZKİPER Z İ, TURGUT Z, ATMACA T, et al. Fingerprint liveness detection using deep learning[C]//2022 9th International Conference on Future Internet of Things and Cloud (FiCloud). IEEE, 2022: 129-135. DOI: 10.1109/FiCloud57274.2022.00025.

[11] JING J, WANG Z, RÄTSCH M, et al. Mobile-Unet: An efficient convolutional neural network for fabric defect detection[J]. Textile Research Journal, 2022, 92(1-2): 30-42. DOI: 10.1177/0040517520928604.

[12] 陶显, 侯伟, 徐德. 基于深度学习的表面缺陷检测方法综述[J]. 自动化学报, 2021, 47(5): 1017-1034.

[13] 罗东亮, 蔡雨萱, 杨子豪, 等. 工业缺陷检测深度学习方法综述[J]. 中国科学: 信息科学, 2022, 52(6): 1002-1039. DOI: 10.1360/SSI-2021-0231.

---

## 三、建议补充/扩展参考文献（按主题）

以下文献未全部出现在上述转接段落中，但可根据论文后续章节（研究方法、实验分析、轻量化部署等）需要选择性引用。

### 3.1 刺绣/纺织品计算机视觉早期经典

[14] KUO C F J, HSU C T M, SHIH C Y. Automatic pattern recognition and color separation of embroidery fabrics[J]. Textile Research Journal, 2011, 81(11): 1145-1157. DOI: 10.1177/0040517511399963.

[15] KUO C F J, SHIH C Y, HSU C T M. Pattern-making simulation on embroidery using probabilistic neural network and texture fitting method[J]. Textile Research Journal, 2011, 81(20): 2082-2094. DOI: 10.1177/0040517511414980.

### 3.2 深度学习基础模型

[16] HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778. DOI: 10.1109/CVPR.2016.90.

[17] DOSOVITSKIY A, BEYER L, KOLESNIKOV A, et al. An image is worth 16×16 words: Transformers for image recognition at scale[C]//International Conference on Learning Representations. 2021.

[18] OQUAB M, DARCET T, MOUTAKANNI T, et al. DINOv2: Learning robust visual features without supervision[J]. arXiv preprint arXiv:2304.07193, 2023.

[19] DENG J, GUO J, XUE N, et al. ArcFace: Additive angular margin loss for deep face recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 4690-4699. DOI: 10.1109/CVPR.2019.00482.

### 3.3 多任务学习/轻量化部署

[20] ZHANG Y, YANG Q. A survey on multi-task learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2022, 34(12): 5586-5609. DOI: 10.1109/TKDE.2021.3079203.

---

## 四、使用建议

1. **编号调整**：将“二、对应参考文献”中的 [1]–[13] 按论文已有文献顺序重新编号，并同步修改转接段落中的上标。  
2. **精简/增删**：若论文侧重“真伪鉴别”，可保留 [9][10] 并补充相关生物特征反欺骗综述；若侧重“纹样分类”，可强化 [1][4][5][6] 的论述。  
3. **英文写作**：如需英文论文的 Literature Review，可直接将上述段落译为英文，参考文献格式转换为 APA/IEEE 即可。  
4. **方法章节引用**：扩展文献 [16]–[20] 建议在“研究方法”或“实验设置”中引用，以支撑 ResNet-50、DINOv2、ArcFace、多任务联合训练等技术选型。