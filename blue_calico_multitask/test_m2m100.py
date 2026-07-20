from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
translator = pipeline('translation', model='facebook/m2m100_418M', device=device)

texts = [
    "We propose a deep learning method for embroidery pattern classification.",
    "The results show that ResNet50-CBAM exhibits best performance across fold, pattern and network performance.",
]
with open('test_m2m100_out.txt', 'w', encoding='utf-8') as f:
    for t in texts:
        out = translator(t, max_length=512, src_lang='en', tgt_lang='zh')
        f.write(out[0]['translation_text'] + '\n')
