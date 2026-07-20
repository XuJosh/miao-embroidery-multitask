from docx import Document
doc = Document('papers/Zhao_ResNet50_CBAM_中文译稿_v2.docx')
with open('docx_preview_v2.txt', 'w', encoding='utf-8') as f:
    for i, para in enumerate(doc.paragraphs[:50]):
        f.write(f'{i+1}. {para.text}\n')
