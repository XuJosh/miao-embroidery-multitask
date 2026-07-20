import fitz, sys
pdf_path = 'papers/Zhao_ResNet50_CBAM_npj_Heritage_Science_2026.pdf'
doc = fitz.open(pdf_path)
with open('extracted_pymupdf.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(doc, start=1):
        f.write(f'--- Page {i} ---\n')
        f.write(page.get_text())
        f.write('\n\n')
