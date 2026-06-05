import torch
import torch.nn as nn
import torch.nn.functional as F

from .f3net import CFI, _cbr
from .poolnet import _get_backbone
from .poolnet_cfm import CFM


class F3NetCFM(nn.Module):
    """
    F3Net-CFM: 跨模型组合 —— 用 PoolNetCFM 的 CFM 替换 F3Net 的 FFM。

    改进动机
    --------
    F3Net 原始的 FFM（Feature Fusion Feedback Module）通过 SE 通道注意力
    将 Path-2 特征与 Path-1 反馈做二元融合，但：
      1. FFM 只利用 Path-1 同层的反馈，缺乏深层语义锚定；
      2. FFM 使用全局 SE 加权，局部区分能力有限。

    PoolNetCFM 的 CFM（Cross-level Feature Module）具有以下优势：
      1. 三路输入：out（当前特征）× x_skip（同层反馈）× x_info（深层语义）；
      2. 互信息门控（deep_proj × skip_proj）提取两路特征的共同模式；
      3. 残差细化 + 可学习 γ 参数，训练初期不破坏原始特征分布。

    本模型将 CFM 嵌入 F3Net 的双路解码框架：
      - x_skip = Path-1 同层输出（局部反馈）
      - x_info = Path-1 最深层输出 p1_3（全局语义锚，类似 PoolNetCFM 的 PPM info）
      这样 CFM 既能接收同层细节反馈，又能感知全局语义信息。

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

        # ── Path 1: 粗粒度解码器（同 F3Net）────────────────────────
        self.cfi1_43 = CFI(ch4, ch3, mid_ch)
        self.cfi1_32 = CFI(mid_ch, ch2, mid_ch)
        self.cfi1_21 = CFI(mid_ch, ch1, mid_ch)

        # ── Path 2: 精细化解码器（CFM 替换 FFM）────────────────────
        self.cfi2_43 = CFI(ch4, ch3, mid_ch)
        self.cfm_3   = CFM(mid_ch)   # (p2_cfi, p1_3, p1_3) — 最深层作 skip 和 info
        self.cfi2_32 = CFI(mid_ch, ch2, mid_ch)
        self.cfm_2   = CFM(mid_ch)   # (p2_cfi, p1_2, p1_3) — 同层 skip + 深层 info
        self.cfi2_21 = CFI(mid_ch, ch1, mid_ch)
        self.cfm_1   = CFM(mid_ch)   # (p2_cfi, p1_1, p1_3) — 同层 skip + 深层 info

        # ── 输出头 ─────────────────────────────────────────────────
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
        # c1: [B,  ch1, H/4,  W/4 ]
        # c2: [B,  ch2, H/8,  W/8 ]
        # c3: [B,  ch3, H/16, W/16]
        # c4: [B,  ch4, H/32, W/32]

        # ── Path 1 (粗粒度) ─────────────────────────────────────────
        p1_3 = self.cfi1_43(c4, c3)        # [B, mid, H/16, W/16]
        p1_2 = self.cfi1_32(p1_3, c2)      # [B, mid, H/8,  W/8 ]
        p1_1 = self.cfi1_21(p1_2, c1)      # [B, mid, H/4,  W/4 ]

        # ── Path 2 (CFM 精细化) ─────────────────────────────────────
        # CFM(out, x_skip, x_info)
        #   out    : Path-2 当前 CFI 输出
        #   x_skip : Path-1 同层输出（局部细节反馈）
        #   x_info : Path-1 最深层 p1_3（全局语义锚，CFM 自动插值对齐）

        p2_3 = self.cfm_3(self.cfi2_43(c4, c3), p1_3, p1_3)   # [B, mid, H/16, W/16]
        p2_2 = self.cfm_2(self.cfi2_32(p2_3, c2), p1_2, p1_3) # [B, mid, H/8,  W/8 ]
        p2_1 = self.cfm_1(self.cfi2_21(p2_2, c1), p1_1, p1_3) # [B, mid, H/4,  W/4 ]

        # ── 预测 ───────────────────────────────────────────────────
        out_main = F.interpolate(self.head_main(p2_1), (H, W), mode='bilinear', align_corners=True)
        out_aux  = F.interpolate(self.head_aux(p1_1),  (H, W), mode='bilinear', align_corners=True)

        return out_main, out_aux
