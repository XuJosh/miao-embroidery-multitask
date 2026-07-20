from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
translator = pipeline('translation', model='Helsinki-NLP/opus-mt-tc-big-en-zh', device=device)

texts = [
    "We propose a deep learning method for embroidery pattern classification.",
    "The results show that ResNet50-CBAM exhibits best performance across fold, pattern and network performance.",
]
with open('test_opus_big_out.txt', 'w', encoding='utf-8') as f:
    for t in texts:
        out = translator(t, max_length=512)
        f.write(out[0]['translation_text'] + '\n')
