import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet18 import ResNet18, ResNet18Pre


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RFBModified(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch0 = ConvBNReLU(in_channels, out_channels, 1)

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 1),
            ConvBNReLU(out_channels, out_channels, (1, 3), padding=(0, 1)),
            ConvBNReLU(out_channels, out_channels, (3, 1), padding=(1, 0)),
            ConvBNReLU(out_channels, out_channels, 3, padding=3, dilation=3),
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 1),
            ConvBNReLU(out_channels, out_channels, (1, 5), padding=(0, 2)),
            ConvBNReLU(out_channels, out_channels, (5, 1), padding=(2, 0)),
            ConvBNReLU(out_channels, out_channels, 3, padding=5, dilation=5),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 1),
            ConvBNReLU(out_channels, out_channels, (1, 7), padding=(0, 3)),
            ConvBNReLU(out_channels, out_channels, (7, 1), padding=(3, 0)),
            ConvBNReLU(out_channels, out_channels, 3, padding=7, dilation=7),
        )

        self.conv_cat = ConvBNReLU(out_channels * 4, out_channels, 3, padding=1)
        self.conv_res = ConvBNReLU(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat([x0, x1, x2, x3], dim=1))
        return self.relu(x_cat + self.conv_res(x))


class Aggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_upsample1 = ConvBNReLU(channels, channels, 3, padding=1)
        self.conv_upsample2 = ConvBNReLU(channels, channels, 3, padding=1)
        self.conv_upsample3 = ConvBNReLU(channels, channels, 3, padding=1)
        self.conv_upsample4 = ConvBNReLU(channels, channels, 3, padding=1)
        self.conv_upsample5 = ConvBNReLU(channels * 2, channels * 2, 3, padding=1)

        self.conv_concat2 = ConvBNReLU(channels * 2, channels * 2, 3, padding=1)
        self.conv_concat3 = ConvBNReLU(channels * 3, channels * 3, 3, padding=1)
        self.conv4 = ConvBNReLU(channels * 3, channels * 3, 3, padding=1)
        self.conv5 = nn.Conv2d(channels * 3, 1, 1)

    def forward(self, x1, x2, x3):
        x1_up = self.upsample(x1)
        x2_1 = self.conv_upsample1(x1_up) * x2

        x1_up_twice = self.upsample(x1_up)
        x2_up = self.upsample(x2)
        x3_1 = self.conv_upsample2(x1_up_twice) * self.conv_upsample3(x2_up) * x3

        x2_2 = torch.cat([x2_1, self.conv_upsample4(x1_up)], dim=1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat([x3_1, self.conv_upsample5(self.upsample(x2_2))], dim=1)
        x3_2 = self.conv_concat3(x3_2)

        return self.conv5(self.conv4(x3_2))


class HolisticAttention(nn.Module):
    def __init__(self, kernel_size=32, sigma=4.0):
        super().__init__()
        kernel = self._build_gaussian_kernel(kernel_size, sigma)
        self.kernel = nn.Parameter(kernel)

    @staticmethod
    def _build_gaussian_kernel(kernel_size, sigma):
        center = (kernel_size - 1) / 2.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - center
        yy, xx = torch.meshgrid(coords, coords)
        kernel = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    @staticmethod
    def _normalize(x, eps=1e-8):
        x_flat = x.flatten(2)
        x_min = x_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        x_max = x_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        return (x - x_min) / (x_max - x_min + eps)

    def forward(self, attention_map, features):
        kernel_size = self.kernel.shape[-1]
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size // 2
        padded = F.pad(
            attention_map,
            (pad_left, pad_right, pad_left, pad_right),
            mode="replicate",
        )

        smooth_attention = F.conv2d(padded, self.kernel)
        smooth_attention = self._normalize(smooth_attention)
        holistic_attention = torch.max(smooth_attention, attention_map)
        return features * holistic_attention


class B2ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = ResNet18Pre() if pretrained else ResNet18()

        self.stem   = base.stem
        self.layer1 = base.layer1   # 64ch
        self.layer2 = base.layer2   # 128ch

        self.layer3_1 = base.layer3                   # 256ch
        self.layer4_1 = base.layer4                   # 512ch
        self.layer3_2 = copy.deepcopy(base.layer3)
        self.layer4_2 = copy.deepcopy(base.layer4)

    def forward(self, x):
        x  = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3_1 = self.layer3_1(x2)
        x4_1 = self.layer4_1(x3_1)
        return x1, x2, x3_1, x4_1


class CPDResNet(nn.Module):
    def __init__(self, channel=32, pretrained=True):
        super().__init__()
        self.backbone = B2ResNet(pretrained=pretrained)

        self.rfb2_1 = RFBModified(128,  channel)
        self.rfb3_1 = RFBModified(256,  channel)
        self.rfb4_1 = RFBModified(512,  channel)
        self.agg1 = Aggregation(channel)

        self.ha = HolisticAttention(kernel_size=32, sigma=4.0)

        self.rfb2_2 = RFBModified(128,  channel)
        self.rfb3_2 = RFBModified(256,  channel)
        self.rfb4_2 = RFBModified(512,  channel)
        self.agg2 = Aggregation(channel)

        self._init_custom_weights()

    def _init_custom_weights(self):
        for module in [
            self.rfb2_1, self.rfb3_1, self.rfb4_1, self.agg1,
            self.rfb2_2, self.rfb3_2, self.rfb4_2, self.agg2,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        input_size = x.shape[2:]

        x1, x2, x3_1, x4_1 = self.backbone(x)

        x2_rfb_1 = self.rfb2_1(x2)
        x3_rfb_1 = self.rfb3_1(x3_1)
        x4_rfb_1 = self.rfb4_1(x4_1)
        attention_map = self.agg1(x4_rfb_1, x3_rfb_1, x2_rfb_1)

        x2_ha = self.ha(torch.sigmoid(attention_map), x2)
        x3_2  = self.backbone.layer3_2(x2_ha)
        x4_2  = self.backbone.layer4_2(x3_2)

        x2_rfb_2 = self.rfb2_2(x2_ha)
        x3_rfb_2 = self.rfb3_2(x3_2)
        x4_rfb_2 = self.rfb4_2(x4_2)
        detection_map = self.agg2(x4_rfb_2, x3_rfb_2, x2_rfb_2)

        detection_map = F.interpolate(detection_map, size=input_size, mode="bilinear", align_corners=True)
        attention_map = F.interpolate(attention_map, size=input_size, mode="bilinear", align_corners=True)

        return detection_map, attention_map
