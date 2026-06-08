import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class RFB(nn.Module):
    """RFB-like multi-scale feature adapter used by CPD."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        return self.relu(x_cat + self.conv_res(x))


class Aggregation(nn.Module):
    """Dense aggregation decoder over the last three ResNet stages."""

    def __init__(self, channels):
        super().__init__()
        self.conv_upsample1 = BasicConv2d(channels, channels, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channels, channels, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channels, channels, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channels, channels, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channels, 2 * channels, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channels, 2 * channels, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channels, 3 * channels, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channels, 3 * channels, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channels, 1, 1)

    @staticmethod
    def _resize(x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x_deep, x_mid, x_shallow):
        deep_to_mid = self._resize(x_deep, x_mid)
        deep_to_shallow = self._resize(x_deep, x_shallow)
        mid_to_shallow = self._resize(x_mid, x_shallow)

        x_mid_gated = self.conv_upsample1(deep_to_mid) * x_mid
        x_shallow_gated = (
            self.conv_upsample2(deep_to_shallow)
            * self.conv_upsample3(mid_to_shallow)
            * x_shallow
        )

        x_mid_cat = torch.cat(
            (x_mid_gated, self.conv_upsample4(deep_to_mid)),
            dim=1,
        )
        x_mid_cat = self.conv_concat2(x_mid_cat)

        x_shallow_cat = torch.cat(
            (x_shallow_gated, self.conv_upsample5(self._resize(x_mid_cat, x_shallow))),
            dim=1,
        )
        x_shallow_cat = self.conv_concat3(x_shallow_cat)
        return self.conv5(self.conv4(x_shallow_cat))


def _gaussian_kernel(kernel_size=31, sigma=4.0):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def _min_max_norm(x):
    max_value = x.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    min_value = x.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    return (x - min_value) / (max_value - min_value + 1e-8)


class HolisticAttention(nn.Module):
    """Holistic attention from CPD, implemented as a fixed Gaussian smoothing prior."""

    def __init__(self):
        super().__init__()
        self.register_buffer("gaussian_kernel", _gaussian_kernel(31, 4.0))

    def forward(self, attention_map, features):
        kernel = self.gaussian_kernel.to(device=attention_map.device, dtype=attention_map.dtype)
        soft_attention = F.conv2d(attention_map, kernel, padding=15)
        soft_attention = _min_max_norm(soft_attention)
        return features * torch.maximum(soft_attention, attention_map)


class B2ResNet50(nn.Module):
    """ResNet50 with shared shallow stages and two duplicated deep branches."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet50(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2

        self.layer3_1 = base.layer3
        self.layer4_1 = base.layer4
        self.layer3_2 = copy.deepcopy(base.layer3)
        self.layer4_2 = copy.deepcopy(base.layer4)

    def forward_to_layer2(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        return self.layer2(x)


class CPDResNet(nn.Module):
    """Paper-faithful CPD-ResNet50.

    The model returns ``(detection_logits, attention_logits)`` so the existing
    training/evaluation code uses the final detection branch as the primary map.
    """

    def __init__(self, channel=32, pretrained=True, backbone_name="resnet50"):
        super().__init__()
        self.backbone_name = "resnet50"
        self.loss_mode = "cpd_bce"
        self.backbone = B2ResNet50(pretrained=pretrained)

        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = Aggregation(channel)

        self.ha = HolisticAttention()

        self.rfb2_2 = RFB(512, channel)
        self.rfb3_2 = RFB(1024, channel)
        self.rfb4_2 = RFB(2048, channel)
        self.agg2 = Aggregation(channel)

    def forward(self, x):
        input_size = x.shape[-2:]
        x2 = self.backbone.forward_to_layer2(x)

        x3_1 = self.backbone.layer3_1(x2)
        x4_1 = self.backbone.layer4_1(x3_1)
        attention_logits = self.agg1(
            self.rfb4_1(x4_1),
            self.rfb3_1(x3_1),
            self.rfb2_1(x2),
        )

        x2_2 = self.ha(torch.sigmoid(attention_logits), x2)
        x3_2 = self.backbone.layer3_2(x2_2)
        x4_2 = self.backbone.layer4_2(x3_2)
        detection_logits = self.agg2(
            self.rfb4_2(x4_2),
            self.rfb3_2(x3_2),
            self.rfb2_2(x2_2),
        )

        detection_logits = F.interpolate(
            detection_logits,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )
        attention_logits = F.interpolate(
            attention_logits,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )
        return detection_logits, attention_logits
