import torch.nn as nn
import torch.nn.functional as F

from .cpd import Aggregation, B2ResNet50, HolisticAttention, RFB


class SideHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x, target_size):
        x = self.pred(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)


class CPDDS(nn.Module):
    """CPD-ResNet50 with extra deep supervision on detection-branch RFB features."""

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

        self.side2 = SideHead(channel)
        self.side3 = SideHead(channel)
        self.side4 = SideHead(channel)

    def forward(self, x):
        input_size = x.shape[-2:]
        x2 = self.backbone.forward_to_layer2(x)

        x3_1 = self.backbone.layer3_1(x2)
        x4_1 = self.backbone.layer4_1(x3_1)
        x2_1 = self.rfb2_1(x2)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        attention_logits = self.agg1(x4_1, x3_1, x2_1)

        x2_2 = self.ha(attention_logits.sigmoid(), x2)
        x3_2 = self.backbone.layer3_2(x2_2)
        x4_2 = self.backbone.layer4_2(x3_2)
        x2_2 = self.rfb2_2(x2_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
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
        side2_logits = self.side2(x2_2, input_size)
        side3_logits = self.side3(x3_2, input_size)
        side4_logits = self.side4(x4_2, input_size)

        return detection_logits, attention_logits, side2_logits, side3_logits, side4_logits
