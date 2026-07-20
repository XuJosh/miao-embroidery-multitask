import argostranslate.translate
text = "We propose a deep learning method for embroidery pattern classification."
translated = argostranslate.translate.translate(text, 'en', 'zh')
with open('test_argos_out.txt', 'w', encoding='utf-8') as f:
    f.write(translated + '\n')
