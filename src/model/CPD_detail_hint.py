import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpd import Aggregation, B2ResNet50, BasicConv2d, HolisticAttention, RFB


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetailHintFusion(nn.Module):
    """Inject a low-channel conv2_x detail hint into the shallow CPD decoder feature."""

    def __init__(self, low_channels=256, decoder_channels=32, hint_channels=16):
        super().__init__()
        self.detail_hint = nn.Sequential(
            ConvBNReLU(low_channels, hint_channels, kernel_size=1, padding=0),
            ConvBNReLU(hint_channels, hint_channels, kernel_size=3, padding=1),
        )
        self.hint_proj = BasicConv2d(hint_channels, decoder_channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(decoder_channels + hint_channels + 1, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _resize(x, target):
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, decoder_feat, low_feat, attention_logits):
        hint = self._resize(self.detail_hint(low_feat), decoder_feat)
        attention = self._resize(torch.sigmoid(attention_logits), decoder_feat)
        gate = self.gate(torch.cat((decoder_feat, hint, attention), dim=1))
        return decoder_feat + self.gamma * gate * self.hint_proj(hint)


class CPDDetailHint(nn.Module):
    """CPD-ResNet50 with an unsupervised lightweight detail hint branch."""

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
        self.detail_fusion = DetailHintFusion(
            low_channels=256,
            decoder_channels=channel,
            hint_channels=max(channel // 2, 8),
        )
        self.agg2 = Aggregation(channel)

    def forward(self, x):
        input_size = x.shape[-2:]
        x0 = self.backbone.stem(x)
        x1 = self.backbone.layer1(x0)
        x2 = self.backbone.layer2(x1)

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
        x2_2 = self.detail_fusion(
            self.rfb2_2(x2_2),
            x1,
            attention_logits,
        )
        detection_logits = self.agg2(
            self.rfb4_2(x4_2),
            self.rfb3_2(x3_2),
            x2_2,
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
