"""
Translate an English PDF to Chinese sentence-by-sentence using Helsinki-NLP/opus-mt-en-zh,
with PyMuPDF text extraction and cleaning.

Usage:
    python translate_pdf_to_docx_v3.py \
        --input papers/Zhao_ResNet50_CBAM_npj_Heritage_Science_2026.pdf \
        --output papers/Zhao_ResNet50_CBAM_中文译稿_v3.docx
"""
import argparse
import os
import re
import sys

import fitz
import torch
from docx import Document
from transformers import pipeline


DROP_PATTERNS = [
    re.compile(r'^npj\s*\|\s*heritage science', re.I),
    re.compile(r'^Article$', re.I),
    re.compile(r'^Check for updates$', re.I),
    re.compile(r'^https?://doi\.org/', re.I),
    re.compile(r'^npj Heritage Science\s*\|', re.I),
    re.compile(r'^\d+$'),
    re.compile(r'^\d{10,}\(\):,;$'),
]


def should_drop(line):
    stripped = line.strip()
    if not stripped:
        return True
    for pat in DROP_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def split_into_sentences(text):
    # Keep the delimiter with each sentence.
    sents = re.findall(r'[^.!?\n]+[.!?]*', text)
    return [s.strip() for s in sents if s.strip()]


def sanitize_for_xml(text):
    """Remove control characters that are invalid in XML."""
    return ''.join(c for c in text if c == '\n' or c == '\t' or (ord(c) >= 32 and ord(c) < 0xD800) or ord(c) > 0xDFFF)


def is_translatable(text):
    """Skip fragments that are mostly numbers/symbols or too short."""
    t = text.strip()
    if len(t) < 3:
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 3:
        return False
    # Drop lines that look like table rows with many numbers/symbols.
    if letters / len(t) < 0.3:
        return False
    return True


def translate_sentences(sentences, translator, batch_size=16):
    """Translate a list of sentences in batches."""
    translated = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        outputs = translator(batch, max_length=128, batch_size=len(batch))
        translated.extend([o['translation_text'] for o in outputs])
    return translated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Input file not found: {args.input}')
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else -1
    model_name = 'Helsinki-NLP/opus-mt-en-zh'
    print(f'Loading translation model: {model_name} (device={device})')
    translator = pipeline('translation', model=model_name, tokenizer=model_name, device=device)

    print(f'Extracting text from: {args.input}')
    doc = Document()
    doc.add_heading('ResNet50-CBAM 亚洲民族刺绣纹样分类（中文机译稿 v3）', level=0)
    doc.add_paragraph('说明：本文档由 Helsinki-NLP/opus-mt-en-zh 模型对英文 PDF 逐句机器翻译，并经过简单的页眉页脚清理，仅供参考。')

    pdf = fitz.open(args.input)
    for page_idx, page in enumerate(pdf, start=1):
        raw_lines = page.get_text().splitlines()
        cleaned_lines = [ln for ln in raw_lines if not should_drop(ln)]
        text = '\n'.join(cleaned_lines).strip()
        if not text:
            continue

        doc.add_heading(f'第 {page_idx} 页', level=1)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        for para in paragraphs:
            sents = split_into_sentences(para)
            translatable = [s for s in sents if is_translatable(s)]
            if not translatable:
                # Keep original for tables/equations if anything remains.
                if any(len(s.strip()) > 0 for s in sents):
                    doc.add_paragraph(sanitize_for_xml('[原文片段未翻译]\n' + '\n'.join(sents)))
                continue

            translated = translate_sentences(translatable, translator, batch_size=args.batch_size)
            # Rebuild paragraph by mapping translated sentences in order.
            # For skipped non-translatable sentences, insert a placeholder.
            result_parts = []
            trans_idx = 0
            for s in sents:
                if is_translatable(s):
                    result_parts.append(translated[trans_idx])
                    trans_idx += 1
                else:
                    result_parts.append(s.strip())
            doc.add_paragraph(sanitize_for_xml(''.join(result_parts)))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    doc.save(args.output)
    print(f'Saved translated document to: {args.output}')


if __name__ == '__main__':
    main()
