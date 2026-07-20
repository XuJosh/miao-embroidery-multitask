import torch
from model_external import MultiTaskBackbone

print('start test')
for bb in ['resnet50_cbam', 'mobilenetv4_conv_small.e1200_r224_in1k']:
    print('Testing', bb)
    m = MultiTaskBackbone(backbone=bb, num_patterns=5, defect_mode='cls', dropout=0.3)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print({k: v.shape for k, v in out.items()}, 'feature_dim', m.feature_dim)
print('done')
