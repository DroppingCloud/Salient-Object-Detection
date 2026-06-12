import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class B2Backbone(nn.Module):
    """Wraps any backbone (resnet18/34/50) into the dual-branch CPD structure.

    The backbone must expose `stem`, `layer1`, `layer2`, `layer3`, `layer4`.
    We share stem/layer1/layer2 and duplicate layer3/layer4 for the two branches.
    """

    def __init__(self, backbone_model):
        super().__init__()
        self.stem = backbone_model.stem
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2

        self.layer3_1 = backbone_model.layer3
        self.layer4_1 = backbone_model.layer4
        self.layer3_2 = copy.deepcopy(backbone_model.layer3)
        self.layer4_2 = copy.deepcopy(backbone_model.layer4)

    def forward_to_layer2(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        return self.layer2(x)


class CPDResNet(nn.Module):
    """CPD adapted to support resnet18 / resnet34 / resnet50 backbones.

    The backbone is selected via ``backbone_name`` (matches keys in
    ``BACKBONE_REGISTRY`` in config.py).  Channel widths are inferred
    automatically from the backbone's feature map sizes.

    Returns ``(detection_logits, attention_logits)`` so existing
    training/evaluation code uses the detection branch as the primary map.
    """

    # channel layout: [c1, c2, c3, c4]
    _CHANNELS = {
        "resnet18": [64, 128, 256, 512],
        "resnet34": [64, 128, 256, 512],
        "resnet50": [256, 512, 1024, 2048],
    }

    def __init__(self, channel=32, pretrained=True, backbone_name=None):
        super().__init__()

        # ── resolve backbone_name ──────────────────────────────────────────
        if backbone_name is None:
            # lazy import to avoid circular dependency at module load time
            try:
                from common.config import BACKBONE
                backbone_name = BACKBONE
            except Exception:
                backbone_name = "resnet50"

        self.backbone_name = backbone_name
        self.loss_mode = "cpd_bce"

        # ── build backbone ────────────────────────────────────────────────
        backbone_model = self._build_backbone(backbone_name, pretrained)
        self.backbone = B2Backbone(backbone_model)

        # ── channel widths for this backbone ──────────────────────────────
        ch = self._CHANNELS[backbone_name]
        c2, c3, c4 = ch[1], ch[2], ch[3]

        # ── attention branch (branch 1) ───────────────────────────────────
        self.rfb2_1 = RFB(c2, channel)
        self.rfb3_1 = RFB(c3, channel)
        self.rfb4_1 = RFB(c4, channel)
        self.agg1 = Aggregation(channel)

        self.ha = HolisticAttention()

        # ── detection branch (branch 2) ───────────────────────────────────
        self.rfb2_2 = RFB(c2, channel)
        self.rfb3_2 = RFB(c3, channel)
        self.rfb4_2 = RFB(c4, channel)
        self.agg2 = Aggregation(channel)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_backbone(backbone_name: str, pretrained: bool):
        """Instantiate the right backbone class from resnet.py."""
        from model.resnet import (
            ResNet18, ResNet18Pre,
            ResNet34Pre,
            ResNet50Pre,
        )

        if backbone_name == "resnet18":
            return ResNet18Pre() if pretrained else ResNet18()
        elif backbone_name == "resnet34":
            if not pretrained:
                raise ValueError("resnet34 scratch is not implemented in BACKBONE_REGISTRY")
            return ResNet34Pre()
        elif backbone_name == "resnet50":
            if not pretrained:
                raise ValueError("resnet50 scratch is not implemented in BACKBONE_REGISTRY")
            return ResNet50Pre()
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name!r}. "
                             "Choose from 'resnet18', 'resnet34', 'resnet50'.")

    # ------------------------------------------------------------------
    def forward(self, x):
        input_size = x.shape[-2:]
        x2 = self.backbone.forward_to_layer2(x)

        # --- attention branch ---
        x3_1 = self.backbone.layer3_1(x2)
        x4_1 = self.backbone.layer4_1(x3_1)
        attention_logits = self.agg1(
            self.rfb4_1(x4_1),
            self.rfb3_1(x3_1),
            self.rfb2_1(x2),
        )

        # --- holistic attention gate ---
        attention_map = torch.sigmoid(
            F.interpolate(attention_logits, size=x2.shape[-2:], mode="bilinear", align_corners=True)
        )
        x2_attended = self.ha(attention_map, x2)

        # --- detection branch ---
        x3_2 = self.backbone.layer3_2(x2_attended)
        x4_2 = self.backbone.layer4_2(x3_2)
        detection_logits = self.agg2(
            self.rfb4_2(x4_2),
            self.rfb3_2(x3_2),
            self.rfb2_2(x2_attended),
        )

        detection_logits = F.interpolate(
            detection_logits, size=input_size, mode="bilinear", align_corners=True
        )
        attention_logits = F.interpolate(
            attention_logits, size=input_size, mode="bilinear", align_corners=True
        )
        if self.training:
            return {'main': detection_logits, 'aux_sal': [attention_logits]}
        return detection_logits
