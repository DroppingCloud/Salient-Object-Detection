import torch
import torch.nn as nn
import torch.nn.functional as F

from .cpd import Aggregation, B2ResNet50, RFB, _gaussian_kernel, _min_max_norm


class ResidualHolisticAttention(nn.Module):
    """Holistic attention that preserves the original feature path."""

    def __init__(self, init_alpha=0.5):
        super().__init__()
        self.register_buffer("gaussian_kernel", _gaussian_kernel(31, 4.0))
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, attention_map, features):
        kernel = self.gaussian_kernel.to(device=attention_map.device, dtype=attention_map.dtype)
        soft_attention = F.conv2d(attention_map, kernel, padding=15)
        soft_attention = _min_max_norm(soft_attention)
        attention = torch.maximum(soft_attention, attention_map)
        return features * (1.0 + self.alpha * attention)


class CPDResHA(nn.Module):
    """CPD-ResNet50 with residual holistic attention in the detection branch."""

    def __init__(self, channel=32, pretrained=True, backbone_name="resnet50"):
        super().__init__()
        self.backbone_name = "resnet50"
        self.loss_mode = "cpd_bce"
        self.backbone = B2ResNet50(pretrained=pretrained)

        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)
        self.agg1 = Aggregation(channel)

        self.ha = ResidualHolisticAttention(init_alpha=0.5)

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
