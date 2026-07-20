# -*- coding: utf-8 -*-
import os
from docx import Document

entries = os.listdir('D:/song/kimi_DEMO')
docx_files = [e for e in entries if e.endswith('_modified.docx') and not e.startswith('~')]
if not docx_files:
    print('No modified docx found')
    exit(1)

src = os.path.join('D:/song/kimi_DEMO', docx_files[0])
doc = Document(src)
print(f'Loaded: {src}')
print(f'Paragraphs before: {len(doc.paragraphs)}')

# Add citations in the comparison experiment paragraph
for p in doc.paragraphs:
    text = p.text
    if 'EfficientNet-B0' in text and 'ViT-B/16' in text and 'ResNet50-CBAM' in text:
        # Add citations to the method list
        text = text.replace(
            'EfficientNet-B0 与 EfficientNet-B3',
            'EfficientNet-B0 与 EfficientNet-B3 [17]'
        )
        text = text.replace(
            'ViT-B/16',
            'ViT-B/16 [18]'
        )
        text = text.replace(
            'ResNet50-CBAM',
            'ResNet50-CBAM [19,20]'
        )
        text = text.replace(
            'MobileNetV4-Conv-Small',
            'MobileNetV4-Conv-Small [21]'
        )
        text = text.replace(
            'MambaOut',
            'MambaOut [22]'
        )
        p.text = text
        print('Updated comparison methods paragraph')
        break

# Add references for the comparison methods
new_refs = '''
[17] TAN M, LE Q. EfficientNet: Rethinking model scaling for convolutional neural networks[C]//International Conference on Machine Learning. PMLR, 2019: 6105-6114.
[18] DOSOVITSKIY A, BEYER L, KOLESNIKOV A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[C]//International Conference on Learning Representations. 2021.
[19] HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2016: 770-778.
[20] WOO S, PARK J, LEE J Y, et al. CBAM: Convolutional block attention module[C]//Proceedings of the European Conference on Computer Vision. 2018: 3-19.
[21] BURGER B, CHUANG W, YANG Y, et al. MobileNetV4: Universal models for the mobile ecosystem[J]. arXiv preprint arXiv:2404.10518, 2024.
[22] YU W, SI C, ZHOU P, et al. MambaOut: Do we really need Mamba for vision?[J]. arXiv preprint arXiv:2405.07992, 2024.
'''.strip()

# Find the last paragraph and append new references after it
for para in new_refs.split('\n'):
    if para.strip():
        doc.add_paragraph(para.strip())

# Save
# Get a new filename to avoid overwriting temp lock issues
dst = src.rsplit('.', 1)[0] + '_v2.docx'
doc.save(dst)
print(f'Saved to: {dst}')
print(f'Paragraphs after: {len(doc.paragraphs)}')
