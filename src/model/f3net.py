import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet18 import ResNet18Pre


def _cbr(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CFI(nn.Module):
    """
    Cross Feature Interaction (CFI) Module.

    Takes a high-level feature (coarse, semantically rich) and a low-level
    feature (spatially detailed), computes mutual attention gates, and returns
    a single cross-enhanced feature map at the low-level resolution.
    """

    def __init__(self, ch_hi, ch_lo, out_ch):
        super().__init__()
        self.proj_hi = _cbr(ch_hi, out_ch, k=1, p=0)
        self.proj_lo = _cbr(ch_lo, out_ch, k=1, p=0)

        # Two independent gates: each feature learns to attend to the other
        self.gate_hi = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 1), nn.Sigmoid())
        self.gate_lo = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 1), nn.Sigmoid())

        self.fuse = _cbr(out_ch * 2, out_ch)

    def forward(self, f_hi, f_lo):
        f_hi = F.interpolate(f_hi, f_lo.shape[2:], mode='bilinear', align_corners=True)
        h = self.proj_hi(f_hi)
        l = self.proj_lo(f_lo)

        cat = torch.cat([h, l], dim=1)
        h_out = h * self.gate_hi(cat) + l   # high-level enhanced by low-level context
        l_out = l * self.gate_lo(cat) + h   # low-level enhanced by high-level semantics

        return self.fuse(torch.cat([h_out, l_out], dim=1))


class FFM(nn.Module):
    """
    Feature Fusion Feedback Module (FFM).

    Fuses a feature from the current decoder path with a feedback signal from
    the other path. Channel attention (SE-style) re-weights the fused output.
    """

    def __init__(self, ch):
        super().__init__()
        self.fuse = _cbr(ch * 2, ch)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // 4, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // 4, 1), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, feedback):
        feedback = F.interpolate(feedback, x.shape[2:], mode='bilinear', align_corners=True)
        fused = self.fuse(torch.cat([x, feedback], dim=1))
        return fused * self.ca(fused) + fused   # residual keeps original info


class F3Net(nn.Module):
    """
    F3Net: Feature Fusion Feedback Network for Salient Object Detection.
    AAAI 2020  (https://arxiv.org/abs/2011.11167)

    Adapted for ResNet18Pre backbone.

    Forward returns (out_main, out_aux):
      - out_main : refined prediction from Path-2  (primary output)
      - out_aux  : coarse prediction from Path-1   (auxiliary output)

    The tuple is compatible with Trainer._compute_loss(), which applies
    weight 2.0 to out_main and 1.0 to out_aux.
    """

    def __init__(self, mid_ch=128):
        super().__init__()
        self.backbone = ResNet18Pre()
        # c1: 64ch H/4 | c2: 128ch H/8 | c3: 256ch H/16 | c4: 512ch H/32

        # ── Path 1: coarse top-down decoder ──────────────────────────────
        self.cfi1_43 = CFI(512, 256, mid_ch)   # c4 × c3  → H/16
        self.cfi1_32 = CFI(mid_ch, 128, mid_ch) # p1_3 × c2 → H/8
        self.cfi1_21 = CFI(mid_ch, 64, mid_ch)  # p1_2 × c1 → H/4

        # ── Path 2: refined decoder with feedback from Path 1 ────────────
        self.cfi2_43 = CFI(512, 256, mid_ch)
        self.ffm_3   = FFM(mid_ch)               # fuse cfi2_43 ← p1_3
        self.cfi2_32 = CFI(mid_ch, 128, mid_ch)
        self.ffm_2   = FFM(mid_ch)               # fuse cfi2_32 ← p1_2
        self.cfi2_21 = CFI(mid_ch, 64, mid_ch)
        self.ffm_1   = FFM(mid_ch)               # fuse cfi2_21 ← p1_1

        # ── Output heads ─────────────────────────────────────────────────
        self.head_main = nn.Conv2d(mid_ch, 1, 1)
        self.head_aux  = nn.Conv2d(mid_ch, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        H, W = x.shape[2:]

        c1, c2, c3, c4 = self.backbone(x)
        # c1: [B,  64, H/4,  W/4 ]
        # c2: [B, 128, H/8,  W/8 ]
        # c3: [B, 256, H/16, W/16]
        # c4: [B, 512, H/32, W/32]

        # ── Path 1 (coarse) ──────────────────────────────────────────────
        p1_3 = self.cfi1_43(c4, c3)        # [B, mid, H/16, W/16]
        p1_2 = self.cfi1_32(p1_3, c2)      # [B, mid, H/8,  W/8 ]
        p1_1 = self.cfi1_21(p1_2, c1)      # [B, mid, H/4,  W/4 ]

        # ── Path 2 (refined) ─────────────────────────────────────────────
        p2_3 = self.ffm_3(self.cfi2_43(c4, c3), p1_3)    # [B, mid, H/16, W/16]
        p2_2 = self.ffm_2(self.cfi2_32(p2_3, c2), p1_2)   # [B, mid, H/8,  W/8 ]
        p2_1 = self.ffm_1(self.cfi2_21(p2_2, c1), p1_1)   # [B, mid, H/4,  W/4 ]

        # ── Predictions ──────────────────────────────────────────────────
        out_main = F.interpolate(self.head_main(p2_1), (H, W), mode='bilinear', align_corners=True)
        out_aux  = F.interpolate(self.head_aux(p1_1),  (H, W), mode='bilinear', align_corners=True)

        return out_main, out_aux
