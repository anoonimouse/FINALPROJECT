import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block, scaled down for MCU use."""

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        # need FloatFunctional so the skip-add works after quantization
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

        layers = []
        if expand_ratio != 1:
            # 1x1 expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])

        layers.extend([
            # 3x3 depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 1x1 projection back down
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class TinyMCUNet(nn.Module):
    """
    Lightweight MobileNetV2-inspired classifier for CIFAR-10.
    Designed to fit MCU-level resource budgets with width_mult scaling.
    bias=True on convolutions for onednn quantization compatibility.
    """

    def __init__(self, num_classes=10, width_mult=1.0):
        super(TinyMCUNet, self).__init__()
        input_channel = int(16 * width_mult)
        last_channel = int(1280 * width_mult)

        # t=expand ratio, c=output channels, n=repeat, s=stride
        block_settings = [
            [1, int(16 * width_mult), 1, 1],
            [6, int(24 * width_mult), 2, 1],   # stride 1 because cifar is only 32x32
            [6, int(32 * width_mult), 2, 2],
            [6, int(64 * width_mult), 3, 2],
            [6, int(96 * width_mult), 3, 1],
            [6, int(160 * width_mult), 2, 2],
            [6, int(320 * width_mult), 1, 1],
        ]

        features = [
            nn.Conv2d(3, input_channel, 3, 1, 1, bias=True),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        ]

        for t, c, n, s in block_settings:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, c, stride, expand_ratio=t))
                input_channel = c

        features.extend([
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=True),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(inplace=True)
        ])

        self.features = nn.Sequential(*features)

        # quant/dequant stubs for post-training quantization
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.dequant(x)
        x = x.mean([2, 3])  # global avg pool
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    dummy = torch.randn(1, 3, 32, 32)
    net = TinyMCUNet(width_mult=0.5)
    out = net(dummy)
    print(f"Output shape: {out.shape}")
