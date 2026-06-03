import torch
import torch.nn as nn
import torch.nn.functional as F

from .f3net import FFM, _cbr
from .poolnet import _get_backbone, _BACKBONE_TABLE

# ─────────────────────────────────────────────────────────────
# ASPP: Atrous Spatial Pyramid Pooling
# 论文: DeepLab v3 (Chen et al., 2017)
# ─────────────────────────────────────────────────────────────

class ASPP(nn.Module):
    """
    轻量级 ASPP 模块：4 路并行空洞卷积 + 全局平均池化，捕捉多尺度上下文。

    分支构成（输入通道 → out_ch）：
      branch0: 1×1 conv           —— 保留原始局部信息
      branch1: 3×3 dil=2          —— 小感受野上下文
      branch2: 3×3 dil=4          —— 中感受野上下文
      branch3: 3×3 dil=6          —— 大感受野上下文
      branch4: Global Avg Pool    —— 全局语义

    五路拼接后经 1×1 conv 压缩回 out_ch 通道。
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b0 = _cbr(in_ch, out_ch, k=1, p=0)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 5 路拼接 → 压缩
        self.project = _cbr(out_ch * 5, out_ch, k=1, p=0)

    def forward(self, x):
        h, w = x.shape[2:]
        b0 = self.b0(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = F.interpolate(self.b4(x), (h, w), mode='bilinear', align_corners=True)
        return self.project(torch.cat([b0, b1, b2, b3, b4], dim=1))


# ─────────────────────────────────────────────────────────────
# CFI_ASPP：用 ASPP 增强高层特征后再做跨层交互
# ─────────────────────────────────────────────────────────────

class CFI_ASPP(nn.Module):
    """
    改进版 CFI：将高层特征的单一 1×1 投影替换为 ASPP 多尺度聚合，
    低层特征保持原始 1×1 投影。

    流程：
      f_hi → ASPP(ch_hi → out_ch) → h
      f_lo → 1×1 conv(ch_lo → out_ch) → l
      cross-attention(h, l) → fuse → output
    """

    def __init__(self, ch_hi, ch_lo, out_ch):
        super().__init__()
        self.aspp_hi = ASPP(ch_hi, out_ch)          # 多尺度增强高层特征
        self.proj_lo = _cbr(ch_lo, out_ch, k=1, p=0)

        self.gate_hi = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 1), nn.Sigmoid())
        self.gate_lo = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 1), nn.Sigmoid())

        self.fuse = _cbr(out_ch * 2, out_ch)

    def forward(self, f_hi, f_lo):
        f_hi = F.interpolate(f_hi, f_lo.shape[2:], mode='bilinear', align_corners=True)
        h = self.aspp_hi(f_hi)
        l = self.proj_lo(f_lo)

        cat = torch.cat([h, l], dim=1)
        h_out = h * self.gate_hi(cat) + l
        l_out = l * self.gate_lo(cat) + h

        return self.fuse(torch.cat([h_out, l_out], dim=1))


# ─────────────────────────────────────────────────────────────
# F3Net-ASPP
# ─────────────────────────────────────────────────────────────

class F3NetASPP(nn.Module):
    """
    F3Net-ASPP: 将 F3Net 中所有 CFI 模块内的高层特征投影替换为 ASPP，
    以获取更丰富的多尺度上下文信息。

    改进动机
    --------
    原始 F3Net 的 CFI 用 1×1 卷积对高层特征做通道投影，感受野固定，
    无法显式地建模不同尺度的上下文信息。
    本改进将 1×1 投影替换为 ASPP（含 dilation=2,4,6 及全局池化分支），
    使高层特征在与低层特征交互之前已经融合了多尺度语义，
    对不同大小的显著目标具有更强的适应能力。

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

        # ── Path 1: 粗粒度解码器（CFI → CFI_ASPP） ──────────────────
        self.cfi1_43 = CFI_ASPP(ch4, ch3, mid_ch)
        self.cfi1_32 = CFI_ASPP(mid_ch, ch2, mid_ch)
        self.cfi1_21 = CFI_ASPP(mid_ch, ch1, mid_ch)

        # ── Path 2: 精细化解码器（含来自 Path-1 的反馈） ────────────
        self.cfi2_43 = CFI_ASPP(ch4, ch3, mid_ch)
        self.ffm_3   = FFM(mid_ch)
        self.cfi2_32 = CFI_ASPP(mid_ch, ch2, mid_ch)
        self.ffm_2   = FFM(mid_ch)
        self.cfi2_21 = CFI_ASPP(mid_ch, ch1, mid_ch)
        self.ffm_1   = FFM(mid_ch)

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

        # ── Path 1 (粗粒度) ───────────────────────────────────────
        p1_3 = self.cfi1_43(c4, c3)        # [B, mid, H/16, W/16]
        p1_2 = self.cfi1_32(p1_3, c2)      # [B, mid, H/8,  W/8 ]
        p1_1 = self.cfi1_21(p1_2, c1)      # [B, mid, H/4,  W/4 ]

        # ── Path 2 (精细化) ───────────────────────────────────────
        p2_3 = self.ffm_3(self.cfi2_43(c4, c3), p1_3)
        p2_2 = self.ffm_2(self.cfi2_32(p2_3, c2), p1_2)
        p2_1 = self.ffm_1(self.cfi2_21(p2_2, c1), p1_1)

        # ── 预测头 ────────────────────────────────────────────────
        out_main = F.interpolate(self.head_main(p2_1), (H, W), mode='bilinear', align_corners=True)
        out_aux  = F.interpolate(self.head_aux(p1_1),  (H, W), mode='bilinear', align_corners=True)

        return out_main, out_aux
