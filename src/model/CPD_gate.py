import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpd import Aggregation, B2ResNet50, HolisticAttention, RFB


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.constant_(self.fc[-1].bias, 2.0)

    def forward(self, decoder_feat, skip_feat):
        weight = torch.sigmoid(self.fc(F.adaptive_avg_pool2d(decoder_feat, 1)))
        return skip_feat * weight


class SpatialGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.constant_(self.conv[-1].bias, 2.0)

    def forward(self, decoder_feat, skip_feat):
        attention = torch.sigmoid(self.conv(torch.cat((decoder_feat, skip_feat), dim=1)))
        return skip_feat * attention


class DualGateFusion(nn.Module):
    """Use deeper decoder context to filter the adjacent shallower feature."""

    def __init__(self, channels):
        super().__init__()
        self.channel_gate = ChannelGate(channels)
        self.spatial_gate = SpatialGate(channels)
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

    def forward(self, decoder_feat, skip_feat):
        decoder_feat = self._resize(decoder_feat, skip_feat)
        channel_out = self.channel_gate(decoder_feat, skip_feat)
        spatial_out = self.spatial_gate(decoder_feat, skip_feat)
        gated = self.fuse(torch.cat((channel_out, spatial_out), dim=1))
        return skip_feat + self.gamma * gated


class CPDGate(nn.Module):
    """CPD-ResNet50 with dual channel/spatial gates before aggregation."""

    def __init__(self, channel=32, pretrained=True, backbone_name="resnet50"):
        super().__init__()
        self.backbone_name = "resnet50"
        self.loss_mode = "cpd_bce"
        self.backbone = B2ResNet50(pretrained=pretrained)

        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.gate43_1 = DualGateFusion(channel)
        self.gate32_1 = DualGateFusion(channel)
        self.agg1 = Aggregation(channel)

        self.ha = HolisticAttention()

        self.rfb2_2 = RFB(512, channel)
        self.rfb3_2 = RFB(1024, channel)
        self.rfb4_2 = RFB(2048, channel)
        self.gate43_2 = DualGateFusion(channel)
        self.gate32_2 = DualGateFusion(channel)
        self.agg2 = Aggregation(channel)

    def forward(self, x):
        input_size = x.shape[-2:]
        x2 = self.backbone.forward_to_layer2(x)

        x3_1 = self.backbone.layer3_1(x2)
        x4_1 = self.backbone.layer4_1(x3_1)
        x2_1 = self.rfb2_1(x2)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        x3_1 = self.gate43_1(x4_1, x3_1)
        x2_1 = self.gate32_1(x3_1, x2_1)
        attention_logits = self.agg1(x4_1, x3_1, x2_1)

        x2_2 = self.ha(torch.sigmoid(attention_logits), x2)
        x3_2 = self.backbone.layer3_2(x2_2)
        x4_2 = self.backbone.layer4_2(x3_2)
        x2_2 = self.rfb2_2(x2_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x3_2 = self.gate43_2(x4_2, x3_2)
        x2_2 = self.gate32_2(x3_2, x2_2)
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
