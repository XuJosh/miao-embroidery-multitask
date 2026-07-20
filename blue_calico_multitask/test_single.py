from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-zh', device=device)

text = "Despite the growing application of deep learning methods to embroidery pattern classification, existing studies are largely limited to single ethnic groups or datasets, and struggle to handle the abstract, hybrid, and visually overlapping patterns commonly found across ethnic embroidery patterns in Asia."
out = translator(text, max_length=512)
with open('test_single_out.txt', 'w', encoding='utf-8') as f:
    f.write(out[0]['translation_text'] + '\n')
