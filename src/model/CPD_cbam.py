import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpd import Aggregation, B2ResNet50, HolisticAttention, RFB


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_attn = self.mlp(self.avg_pool(x))
        max_attn = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_attn + max_attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat((avg_map, max_map), dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResidualCBAM(nn.Module):
    """CBAM refinement with zero-initialized residual strength."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.cbam = CBAM(channels, reduction)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.alpha * self.cbam(x)


class CPDCBAM(nn.Module):
    """CPD-ResNet50 with CBAM refinement after each RFB feature adapter."""

    def __init__(self, channel=32, pretrained=True, backbone_name="resnet50"):
        super().__init__()
        self.backbone_name = "resnet50"
        self.loss_mode = "cpd_bce"
        self.backbone = B2ResNet50(pretrained=pretrained)

        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.cbam2_1 = ResidualCBAM(channel)
        self.cbam3_1 = ResidualCBAM(channel)
        self.cbam4_1 = ResidualCBAM(channel)
        self.agg1 = Aggregation(channel)

        self.ha = HolisticAttention()

        self.rfb2_2 = RFB(512, channel)
        self.rfb3_2 = RFB(1024, channel)
        self.rfb4_2 = RFB(2048, channel)
        self.cbam2_2 = ResidualCBAM(channel)
        self.cbam3_2 = ResidualCBAM(channel)
        self.cbam4_2 = ResidualCBAM(channel)
        self.agg2 = Aggregation(channel)

    def forward(self, x):
        input_size = x.shape[-2:]
        x2 = self.backbone.forward_to_layer2(x)

        x3_1 = self.backbone.layer3_1(x2)
        x4_1 = self.backbone.layer4_1(x3_1)
        x2_1 = self.cbam2_1(self.rfb2_1(x2))
        x3_1 = self.cbam3_1(self.rfb3_1(x3_1))
        x4_1 = self.cbam4_1(self.rfb4_1(x4_1))
        attention_logits = self.agg1(x4_1, x3_1, x2_1)

        x2_2 = self.ha(torch.sigmoid(attention_logits), x2)
        x3_2 = self.backbone.layer3_2(x2_2)
        x4_2 = self.backbone.layer4_2(x3_2)
        x2_2 = self.cbam2_2(self.rfb2_2(x2_2))
        x3_2 = self.cbam3_2(self.rfb3_2(x3_2))
        x4_2 = self.cbam4_2(self.rfb4_2(x4_2))
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
