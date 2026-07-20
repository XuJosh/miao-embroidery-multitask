import sys
print('python:', sys.version.split()[0])

try:
    import torch
    print('torch:', torch.__version__)
    print('cuda:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('cuda_version:', torch.version.cuda)
except Exception as e:
    print('torch error:', e)

try:
    import torchvision
    print('torchvision:', torchvision.__version__)
except Exception as e:
    print('torchvision error:', e)

try:
    import peft
    print('peft:', peft.__version__)
except Exception as e:
    print('peft error:', e)

try:
    import scipy
    print('scipy:', scipy.__version__)
except Exception as e:
    print('scipy error:', e)

try:
    import matplotlib
    print('matplotlib:', matplotlib.__version__)
except Exception as e:
    print('matplotlib error:', e)

try:
    import PIL
    print('PIL:', PIL.__version__)
except Exception as e:
    print('PIL error:', e)

try:
    import docx
    print('python-docx: ok')
except Exception as e:
    print('docx error:', e)

try:
    import numpy
    print('numpy:', numpy.__version__)
except Exception as e:
    print('numpy error:', e)

try:
    import mambapy
    print('mambapy: ok')
except Exception as e:
    print('mambapy error:', e)
