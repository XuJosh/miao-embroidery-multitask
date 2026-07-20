from deep_translator import GoogleTranslator

text = 'We propose a deep learning method for embroidery pattern classification.'
try:
    print('Google:', GoogleTranslator(source='auto', target='zh-CN').translate(text))
except Exception as e:
    print('Google error:', e)
