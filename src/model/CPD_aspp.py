import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpd_new import Aggregation, B2ResNet50, BasicConv2d, HolisticAttention, RFB


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    """Lightweight ASPP for the reduced 32-channel CPD feature space."""

    def __init__(self, in_channels, out_channels, rates=(1, 3, 5, 7)):
        super().__init__()
        self.branch0 = ConvBNReLU(in_channels, out_channels, 1)
        self.branch1 = ConvBNReLU(
            in_channels,
            out_channels,
            3,
            padding=rates[1],
            dilation=rates[1],
        )
        self.branch2 = ConvBNReLU(
            in_channels,
            out_channels,
            3,
            padding=rates[2],
            dilation=rates[2],
        )
        self.branch3 = ConvBNReLU(
            in_channels,
            out_channels,
            3,
            padding=rates[3],
            dilation=rates[3],
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = BasicConv2d(out_channels * 5, out_channels, 1)
        self.residual = BasicConv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        global_context = F.interpolate(
            self.global_branch(x),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        out = torch.cat(
            (
                self.branch0(x),
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                global_context,
            ),
            dim=1,
        )
        return self.relu(self.project(out) + self.residual(x))


class CPDASPP(nn.Module):
    """CPD-ResNet50 with ASPP context enhancement on the deepest decoder feature."""

    def __init__(self, channel=32, pretrained=True, backbone_name="resnet50"):
        super().__init__()
        self.backbone_name = "resnet50"
        self.loss_mode = "cpd_bce"
        self.backbone = B2ResNet50(pretrained=pretrained)

        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.aspp4_1 = ASPP(channel, channel)
        self.agg1 = Aggregation(channel)

        self.ha = HolisticAttention()

        self.rfb2_2 = RFB(512, channel)
        self.rfb3_2 = RFB(1024, channel)
        self.rfb4_2 = RFB(2048, channel)
        self.aspp4_2 = ASPP(channel, channel)
        self.agg2 = Aggregation(channel)

    def forward(self, x):
        input_size = x.shape[-2:]
        x2 = self.backbone.forward_to_layer2(x)

        x3_1 = self.backbone.layer3_1(x2)
        x4_1 = self.backbone.layer4_1(x3_1)
        attention_logits = self.agg1(
            self.aspp4_1(self.rfb4_1(x4_1)),
            self.rfb3_1(x3_1),
            self.rfb2_1(x2),
        )

        x2_2 = self.ha(torch.sigmoid(attention_logits), x2)
        x3_2 = self.backbone.layer3_2(x2_2)
        x4_2 = self.backbone.layer4_2(x3_2)
        detection_logits = self.agg2(
            self.aspp4_2(self.rfb4_2(x4_2)),
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
