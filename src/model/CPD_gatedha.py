import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpd import Aggregation, B2ResNet50, RFB, _gaussian_kernel, _min_max_norm


class GatedHolisticAttention(nn.Module):
    """Holistic attention with a learned residual gate over the shared CPD feature."""

    def __init__(self, channels=512):
        super().__init__()
        self.register_buffer("gaussian_kernel", _gaussian_kernel(31, 4.0))
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, attention_map, features):
        kernel = self.gaussian_kernel.to(device=attention_map.device, dtype=attention_map.dtype)
        soft_attention = F.conv2d(attention_map, kernel, padding=15)
        soft_attention = _min_max_norm(soft_attention)
        attention = torch.maximum(soft_attention, attention_map)

        if attention.shape[-2:] != features.shape[-2:]:
            attention = F.interpolate(
                attention,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        gate = self.gate(torch.cat((features, attention), dim=1))
        return features * (1.0 + self.alpha * gate * attention)


class CPDGatedHA(nn.Module):
    """CPD-ResNet50 with gated residual holistic attention."""

    def __init__(self, channel=32, pretrained=True, backbone_name="resnet50"):
        super().__init__()
        self.backbone_name = "resnet50"
        self.loss_mode = "cpd_bce"
        self.backbone = B2ResNet50(pretrained=pretrained)

        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = Aggregation(channel)

        self.ha = GatedHolisticAttention(channels=512)

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

        x2_2 = self.ha(attention_logits.sigmoid(), x2)
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
