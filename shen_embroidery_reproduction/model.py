"""
Paper: "Research on the innovative application of Shen Embroidery cultural
       heritage based on convolutional neural network"
Network reproduction in PyTorch.

- MobileNet V1 backbone (built from the paper's Table 1)
- Spatial Pyramid Pooling (SPP) replacing the final avg pool
- Optional transfer learning via timm's pretrained MobileNet-V1 weights
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPPLayer(nn.Module):
    """
    Spatial Pyramid Pooling.
    Uses adaptive max-pooling to produce fixed-size feature maps of the
    requested levels, flattens them and concatenates.

    For an input feature map of shape (B, C, H, W) and levels=(1, 3, 5):
        output_dim = C * (1^2 + 3^2 + 5^2)
    """

    def __init__(self, levels=(1, 3, 5), pool_type='max'):
        super().__init__()
        self.levels = levels
        self.pool_type = pool_type.lower()
        if self.pool_type not in ('max', 'avg'):
            raise ValueError("pool_type must be 'max' or 'avg'")

    def forward(self, x):
        B, C, H, W = x.size()
        feats = []
        for level in self.levels:
            if self.pool_type == 'max':
                y = F.adaptive_max_pool2d(x, (level, level))
            else:
                y = F.adaptive_avg_pool2d(x, (level, level))
            feats.append(y.view(B, -1))
        return torch.cat(feats, dim=1)

    def output_dim(self, channels):
        return channels * sum(l * l for l in self.levels)


def _conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=None):
    if padding is None:
        padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


def _dsc(in_ch, out_ch, stride):
    """Depthwise separable convolution: depthwise + pointwise."""
    return nn.Sequential(
        # depthwise
        nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU6(inplace=True),
        # pointwise
        nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1Backbone(nn.Module):
    """
    MobileNet V1 feature extractor following the paper's Table 1.
    Input:  (B, 3, 224, 224)
    Output: (B, 1024, 7, 7)
    """

    def __init__(self):
        super().__init__()
        layers = [
            # standard conv
            _conv_bn_relu(3, 32, kernel_size=3, stride=2),  # -> 112
            # separable conv blocks
            _dsc(32, 64, 1),    # -> 112
            _dsc(64, 128, 2),   # -> 56
            _dsc(128, 128, 1),  # -> 56
            _dsc(128, 256, 2),  # -> 28
            _dsc(256, 256, 1),  # -> 28
            _dsc(256, 512, 2),  # -> 14
            # 5 repeated blocks
            _dsc(512, 512, 1),
            _dsc(512, 512, 1),
            _dsc(512, 512, 1),
            _dsc(512, 512, 1),
            _dsc(512, 512, 1),
            _dsc(512, 1024, 2), # -> 7
            _dsc(1024, 1024, 1),# -> 7
        ]
        self.features = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.features(x)


def _make_timm_mobilenetv1_backbone(pretrained=True):
    """Try to load a pretrained MobileNet-V1 backbone from timm."""
    import timm
    # num_classes=0 removes the classifier; global_pool='' removes global pooling
    backbone = timm.create_model(
        'mobilenetv1_100.ra4_e3600_r224_in1k',
        pretrained=pretrained,
        num_classes=0,
        global_pool='',
    )
    return backbone


class MobileNetV1Classifier(nn.Module):
    """
    Baseline MobileNet V1 classifier (global avg pool + FC).
    This reproduces the "original MobileNet V1" baseline in the paper.
    """

    def __init__(self, num_classes=2, pretrained=True, backbone=None):
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = self._create_backbone(pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

    def _create_backbone(self, pretrained):
        if pretrained:
            try:
                return _make_timm_mobilenetv1_backbone(pretrained=True)
            except Exception as e:
                print(f"[WARN] Could not load pretrained timm MobileNet-V1: {e}")
                print("[WARN] Falling back to randomly initialized built-in backbone.")
        return MobileNetV1Backbone()

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ImprovedMobileNetV1(nn.Module):
    """
    Improved MobileNet V1 from the paper:
    - MobileNet V1 backbone
    - SPP module replacing the final avg pool
    - Fully-connected classifier
    """

    def __init__(self, num_classes=2, pretrained=True, spp_levels=(1, 3, 5),
                 dropout=0.0, backbone=None):
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = self._create_backbone(pretrained)

        self.spp = SPPLayer(levels=spp_levels, pool_type='max')
        # compute the classifier input dimension from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(dummy)
            in_dim = self.spp.output_dim(feat.size(1))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_dim, num_classes)

    def _create_backbone(self, pretrained):
        if pretrained:
            try:
                return _make_timm_mobilenetv1_backbone(pretrained=True)
            except Exception as e:
                print(f"[WARN] Could not load pretrained timm MobileNet-V1: {e}")
                print("[WARN] Falling back to randomly initialized built-in backbone.")
        return MobileNetV1Backbone()

    def forward(self, x):
        x = self.backbone(x)
        x = self.spp(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def create_model(model_name='improved_mobilenet_v1', num_classes=2,
                 pretrained=True, spp_levels=(1, 3, 5), dropout=0.0):
    """
    Factory for the models used in the paper.

    model_name options:
        - 'improved_mobilenet_v1'  (MobileNet V1 + SPP)
        - 'mobilenet_v1'           (baseline MobileNet V1)
        - 'alexnet'
        - 'vgg16'
        - 'resnet50'
        - 'inception_v3'
    """
    model_name = model_name.lower().replace('-', '_')

    if model_name == 'improved_mobilenet_v1':
        return ImprovedMobileNetV1(
            num_classes=num_classes, pretrained=pretrained,
            spp_levels=spp_levels, dropout=dropout)

    if model_name == 'mobilenet_v1':
        return MobileNetV1Classifier(
            num_classes=num_classes, pretrained=pretrained)

    # Comparison models from torchvision
    import torchvision.models as models

    weights_arg = 'DEFAULT' if pretrained else None

    if model_name == 'alexnet':
        m = models.alexnet(weights=weights_arg)
        m.classifier[-1] = nn.Linear(4096, num_classes)
    elif model_name == 'vgg16':
        m = models.vgg16_bn(weights=weights_arg)
        m.classifier[-1] = nn.Linear(4096, num_classes)
    elif model_name == 'resnet50':
        m = models.resnet50(weights=weights_arg)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'inception_v3':
        m = models.inception_v3(weights=weights_arg)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        m.aux_logits = False  # disable auxiliary logits for simplicity
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return m


if __name__ == '__main__':
    # quick smoke test
    x = torch.randn(2, 3, 224, 224)
    for name in ['mobilenet_v1', 'improved_mobilenet_v1']:
        net = create_model(name, num_classes=2, pretrained=False)
        y = net(x)
        print(f"{name}: input {x.shape} -> output {y.shape}")
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  parameters: {n_params / 1e6:.2f}M")
