import torch
import torch.nn as nn
import torch.nn.functional as F

from .f3net import CFI, FFM, _cbr
from .poolnet import _get_backbone, _BACKBONE_TABLE


# ─────────────────────────────────────────────────────────────
# CBAM: Convolutional Block Attention Module
# 论文: CBAM: Convolutional Block Attention Module (ECCV 2018)
# ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    通道注意力：对每个通道学习一个标量权重。
    同时利用全局平均池化（捕捉整体统计）和全局最大池化（捕捉显著激活），
    通过共享 MLP 后相加再 sigmoid。
    """
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(attn)


class SpatialAttention(nn.Module):
    """
    空间注意力：对每个空间位置学习一个标量权重。
    沿通道维度做平均和最大，拼接后经 7×7 卷积产生空间注意力图。
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """
    CBAM = 通道注意力 → 空间注意力（串联）。
    先"选哪些通道有用"，再"选哪些空间位置有用"。
    """
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ─────────────────────────────────────────────────────────────
# F3Net-CBAM
# ─────────────────────────────────────────────────────────────

class F3NetCBAM(nn.Module):
    """
    F3Net-CBAM: 在 F3Net 的每个解码阶段输出后插入 CBAM 注意力模块。

    改进动机
    --------
    原始 F3Net 的 CFI 输出直接送入下一层 CFI 或 FFM，未对特征做显式筛选。
    CBAM 串联通道注意力与空间注意力，能够：
      1. 抑制无关通道的噪声干扰（通道注意力）；
      2. 精准定位显著区域，抑制背景响应（空间注意力）。
    在 Path-1 和 Path-2 的每个阶段输出后各插入一个 CBAM，
    使送入下一阶段的特征已被精炼，反馈信号更加准确。

    输出格式
    --------
    返回 (out_main, out_aux) 元组，与 Trainer._compute_loss() 兼容：
      - out_main：Path-2 精细化输出（权重 2.0）
      - out_aux ：Path-1 粗粒度输出（权重 1.0）
    """

    def __init__(self, mid_ch=128, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        self.backbone, channels = _get_backbone(backbone_name, pretrained)
        ch1, ch2, ch3, ch4 = channels

        # ── Path 1: 粗粒度解码器 ────────────────────────────────────
        self.cfi1_43 = CFI(ch4, ch3, mid_ch)
        self.cbam1_3 = CBAM(mid_ch)

        self.cfi1_32 = CFI(mid_ch, ch2, mid_ch)
        self.cbam1_2 = CBAM(mid_ch)

        self.cfi1_21 = CFI(mid_ch, ch1, mid_ch)
        self.cbam1_1 = CBAM(mid_ch)

        # ── Path 2: 精细化解码器（含来自 Path-1 的反馈） ────────────
        self.cfi2_43 = CFI(ch4, ch3, mid_ch)
        self.ffm_3   = FFM(mid_ch)
        self.cbam2_3 = CBAM(mid_ch)

        self.cfi2_32 = CFI(mid_ch, ch2, mid_ch)
        self.ffm_2   = FFM(mid_ch)
        self.cbam2_2 = CBAM(mid_ch)

        self.cfi2_21 = CFI(mid_ch, ch1, mid_ch)
        self.ffm_1   = FFM(mid_ch)
        self.cbam2_1 = CBAM(mid_ch)

        # ── 输出头 ────────────────────────────────────────────────
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

        # ── Path 1 (粗粒度，经 CBAM 精炼) ───────────────────────────
        p1_3 = self.cbam1_3(self.cfi1_43(c4, c3))     # [B, mid, H/16, W/16]
        p1_2 = self.cbam1_2(self.cfi1_32(p1_3, c2))   # [B, mid, H/8,  W/8 ]
        p1_1 = self.cbam1_1(self.cfi1_21(p1_2, c1))   # [B, mid, H/4,  W/4 ]

        # ── Path 2 (精细化，经 CBAM 精炼 + Path-1 反馈) ─────────────
        p2_3 = self.cbam2_3(self.ffm_3(self.cfi2_43(c4, c3), p1_3))
        p2_2 = self.cbam2_2(self.ffm_2(self.cfi2_32(p2_3, c2), p1_2))
        p2_1 = self.cbam2_1(self.ffm_1(self.cfi2_21(p2_2, c1), p1_1))

        # ── 预测头 ────────────────────────────────────────────────
        out_main = F.interpolate(self.head_main(p2_1), (H, W), mode='bilinear', align_corners=True)
        out_aux  = F.interpolate(self.head_aux(p1_1),  (H, W), mode='bilinear', align_corners=True)

        return out_main, out_aux
