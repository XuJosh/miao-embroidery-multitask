import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
translator = pipeline('translation', model='facebook/nllb-200-distilled-600M', device=device, torch_dtype='auto')

texts = [
    "We propose a deep learning method for embroidery pattern classification.",
    "The results show that ResNet50-CBAM exhibits best performance across fold, pattern and network performance.",
]
with open('test_nllb_out.txt', 'w', encoding='utf-8') as f:
    for t in texts:
        out = translator(t, max_length=512, src_lang='eng_Latn', tgt_lang='zho_Hans')
        f.write(out[0]['translation_text'] + '\n')
