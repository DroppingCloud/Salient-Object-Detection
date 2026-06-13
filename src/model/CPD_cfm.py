import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpd import Aggregation, B2ResNet50, HolisticAttention, RFB


class CFM(nn.Module):
    """Cross-feature module for adjacent CPD decoder features."""

    def __init__(self, channels):
        super().__init__()

        def cbr():
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

        self.deep_proj = cbr()
        self.skip_proj = cbr()
        self.deep_refine = cbr()
        self.skip_refine = cbr()
        self.info_proj = cbr()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _resize(x, target):
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, deep, skip, info=None):
        deep = self._resize(deep, skip)
        if info is None:
            info = deep
        else:
            info = self._resize(info, skip)

        common = self.deep_proj(deep) * self.skip_proj(skip)
        deep_refined = deep + self.deep_refine(common)
        skip_refined = skip + self.skip_refine(common)
        fused = self.fuse(torch.cat((deep_refined, skip_refined), dim=1))

        base = skip + deep + info
        enhanced = self.info_proj(fused + info)
        return base + self.gamma * enhanced


class CPDCFM(nn.Module):
    """CPD-ResNet50 with CFM refinement before each partial decoder aggregation."""

    def __init__(self, channel=32, pretrained=True, backbone_name="resnet50"):
        super().__init__()
        self.backbone_name = "resnet50"
        self.loss_mode = "cpd_bce"
        self.backbone = B2ResNet50(pretrained=pretrained)

        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.cfm43_1 = CFM(channel)
        self.cfm32_1 = CFM(channel)
        self.agg1 = Aggregation(channel)

        self.ha = HolisticAttention()

        self.rfb2_2 = RFB(512, channel)
        self.rfb3_2 = RFB(1024, channel)
        self.rfb4_2 = RFB(2048, channel)
        self.cfm43_2 = CFM(channel)
        self.cfm32_2 = CFM(channel)
        self.agg2 = Aggregation(channel)

    def forward(self, x):
        input_size = x.shape[-2:]
        x2 = self.backbone.forward_to_layer2(x)

        x3_1 = self.backbone.layer3_1(x2)
        x4_1 = self.backbone.layer4_1(x3_1)
        x2_1 = self.rfb2_1(x2)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        x3_1 = self.cfm43_1(x4_1, x3_1, x4_1)
        x2_1 = self.cfm32_1(x3_1, x2_1, x4_1)
        attention_logits = self.agg1(x4_1, x3_1, x2_1)

        x2_2 = self.ha(torch.sigmoid(attention_logits), x2)
        x3_2 = self.backbone.layer3_2(x2_2)
        x4_2 = self.backbone.layer4_2(x3_2)
        x2_2 = self.rfb2_2(x2_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x3_2 = self.cfm43_2(x4_2, x3_2, x4_2)
        x2_2 = self.cfm32_2(x3_2, x2_2, x4_2)
        detection_logits = self.agg2(x4_2, x3_2, x2_2)

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
