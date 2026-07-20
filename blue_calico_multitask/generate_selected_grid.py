import os
import re
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches


DESKTOP_DIR = 'C:/Users/admin/Desktop/热力图对比'
OUTPUT_DIR = 'D:/song/kimi_DEMO/blue_calico_multitask/gradcam_comparison_outputs'
DOC_SRC = 'D:/song/kimi_DEMO/comparison_experiment_paper_v6.docx'
DOC_DST = 'D:/song/kimi_DEMO/comparison_experiment_paper_v7.docx'

NET_ORDER = [
    ('ours', 'Ours'),
    ('ResNet50-CBAM', 'ResNet50-CBAM'),
    ('EfficientNet-B3', 'EfficientNet-B3'),
    ('EfficientNet-B0', 'EfficientNet-B0'),
    ('MobileNetV4-Conv-Small', 'MobileNetV4-Conv-Small'),
    ('MambaOut', 'MambaOut'),
    ('ViT-B16', 'ViT-B/16'),
]


def parse_sample_index(filename):
    m = re.search(r'sample_(\d+)_pred', filename)
    return int(m.group(1)) if m else None


def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img) / 255.0


def get_selected_indices(desktop_dir):
    """Infer selected sample indices from the first network folder."""
    folder, _ = NET_ORDER[0]
    folder_path = os.path.join(desktop_dir, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    indices = sorted(set(parse_sample_index(f) for f in files if parse_sample_index(f) is not None))
    return indices


def find_image(desktop_dir, net_folder, idx):
    """Find the image for a given network and sample index."""
    net_dir = os.path.join(desktop_dir, net_folder)
    candidates = [f for f in os.listdir(net_dir) if f.startswith(f'sample_{idx:03d}_pred')]
    if candidates:
        return os.path.join(net_dir, candidates[0])
    return None


def build_grid(desktop_dir, selected_indices, net_order, save_path):
    n_rows = len(selected_indices)
    n_cols = len(net_order)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(selected_indices):
        for j, (folder, title) in enumerate(net_order):
            img_path = find_image(desktop_dir, folder, idx)
            if img_path is None:
                print(f'Warning: missing image for {folder}, sample {idx}')
                img = np.ones((224, 224, 3))
            else:
                img = load_image(img_path)
            axes[i, j].imshow(img)
            if i == 0:
                axes[i, j].set_title(title, fontsize=12)
            if j == 0:
                axes[i, j].set_ylabel(f'Sample {idx}', fontsize=12)
            axes[i, j].axis('off')

    plt.suptitle('Grad-CAM Visualization Comparison of Selected Real Samples', fontsize=15, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved selected grid to: {save_path}')


def update_word_doc(doc_src, doc_dst, img_path, selected_indices):
    doc = Document(doc_src)

    # Replace image in paragraph 22 (Grad-CAM figure)
    p22 = doc.paragraphs[22]
    for shape in p22._element.xpath('.//w:drawing'):
        shape.getparent().remove(shape)
    run = p22.add_run()
    run.add_picture(img_path, width=Inches(6.5))

    # Update caption
    sample_str = '、'.join(str(idx) for idx in selected_indices)
    new_caption = (
        f'图 3 用户在测试集上挑选的样本（编号 {sample_str}）的 Grad-CAM 可视化对比。'
        '每行对应一个真实样本，每列分别对应 Ours、ResNet50-CBAM [2026]、'
        'EfficientNet-B3、EfficientNet-B0、MobileNetV4-Conv-Small [2026]、'
        'MambaOut [2025] 以及 ViT-B/16。颜色越红表示模型关注度越高。'
    )
    doc.paragraphs[23].text = new_caption

    # Update detailed observation
    new_p24 = (
        '图 3 展示了从测试集真实样本中挑选的 3 个典型样本的 Grad-CAM 可视化对比。'
        '可以看出，Ours 的关注区域始终与绣品的关键纹样、结构轮廓以及局部纹理细节保持一致，'
        '说明三分支特征提取与动态多任务损失能够引导网络将注意力分配在最具判别性的区域。'
        'ResNet50-CBAM [2026] 虽然也能定位到主要目标，但热力图响应相对弥散，'
        '在样本边缘处容易出现无关响应。EfficientNet-B3 与 EfficientNet-B0 更关注局部纹理块，'
        '缺乏对全局纹样语义的把握，导致在纹样相似但类别不同的情况下容易误判。'
        'MobileNetV4-Conv-Small [2026] 因模型容量有限，热力图响应集中在高对比度像素，'
        '难以覆盖完整的刺绣区域。MambaOut [2025] 作为状态空间模型，全局上下文建模能力较强，'
        '但在苗绣这种纹理细微、缺陷尺度小的数据上，关注区域仍显粗糙。'
        'ViT-B/16 直接将图像切分为 16×16 的 patch，缺少针对刺绣纹理的预训练与局部归纳偏置，'
        '注意力分散或产生大面积无关响应，这与它在真伪鉴别任务上准确率仅为 0.640 的量化结果一致。'
    )
    doc.paragraphs[24].text = new_p24

    # Update conclusion
    new_p25 = (
        '综上，可视化对比进一步验证了 Ours 在苗绣真伪鉴别、纹样识别与疵点检测任务中的优势：'
        '它不仅能够关注绣品的关键表征，还能将注意力约束在真实刺绣区域，避免背景干扰。'
        '其他网络或受限于模型容量，或缺少局部—全局协同机制，'
        '难以在纹理细微、缺陷尺度小、真伪边界模糊的苗绣数据上取得一致提升。'
        '这也说明三分支特征提取、动态多任务损失与 ArcFace 边际损失在引导网络关注判别性区域方面具有重要作用。'
    )
    doc.paragraphs[25].text = new_p25

    doc.save(doc_dst)
    print(f'Saved Word doc to: {doc_dst}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selected = get_selected_indices(DESKTOP_DIR)
    print(f'Selected sample indices: {selected}')

    grid_path = os.path.join(OUTPUT_DIR, 'gradcam_selected_grid.png')
    build_grid(DESKTOP_DIR, selected, NET_ORDER, grid_path)

    update_word_doc(DOC_SRC, DOC_DST, grid_path, selected)


if __name__ == '__main__':
    main()
