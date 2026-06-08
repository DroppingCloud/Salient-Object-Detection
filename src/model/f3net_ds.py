import torch
import torch.nn as nn
import torch.nn.functional as F

from .f3net import CFI, FFM, _cbr
from .poolnet import _get_backbone


class F3NetDS(nn.Module):
    """
    F3Net-DS: F3Net with Deep Supervision.

    改进动机
    --------
    原始 F3Net 只在最终输出和 Path-1 末端各施加一次监督信号。
    深监督（Deep Supervision）在 Path-2 的中间解码阶段
    (H/16, H/8) 额外添加辅助预测头，给中间特征层直接施加梯度：
      1. 迫使中间特征层在早期就能感知显著目标，加快收敛；
      2. 提供更密集的监督梯度，缓解深层梯度消失；
      3. 相当于多尺度正则化，改善边界细节。

    输出格式 (ordered by Trainer loss weight)
    ------------------------------------------
    返回 4 元组，Trainer._compute_loss() 自动加权：
      out_main  (index 0, w=2.0) : Path-2 最终预测 (H×W)
      out_aux   (index 1, w=1.0) : Path-1 末端辅助 (H×W)
      out_ds2   (index 2, w=1.0) : Path-2 H/8 中间深监督 (H×W)
      out_ds3   (index 3, w=1.0) : Path-2 H/16 中间深监督 (H×W)
    """

    def __init__(self, mid_ch=128, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        self.backbone, channels = _get_backbone(backbone_name, pretrained)
        ch1, ch2, ch3, ch4 = channels

        # ── Path 1: 粗粒度解码器 ─────────────────────────────────────
        self.cfi1_43 = CFI(ch4, ch3, mid_ch)
        self.cfi1_32 = CFI(mid_ch, ch2, mid_ch)
        self.cfi1_21 = CFI(mid_ch, ch1, mid_ch)

        # ── Path 2: 精细化解码器（含 Path-1 反馈）───────────────────
        self.cfi2_43 = CFI(ch4, ch3, mid_ch)
        self.ffm_3   = FFM(mid_ch)
        self.cfi2_32 = CFI(mid_ch, ch2, mid_ch)
        self.ffm_2   = FFM(mid_ch)
        self.cfi2_21 = CFI(mid_ch, ch1, mid_ch)
        self.ffm_1   = FFM(mid_ch)

        # ── 输出头 ────────────────────────────────────────────────────
        self.head_main = nn.Conv2d(mid_ch, 1, 1)   # Path-2 最终
        self.head_aux  = nn.Conv2d(mid_ch, 1, 1)   # Path-1 末端辅助
        self.head_ds2  = nn.Conv2d(mid_ch, 1, 1)   # 深监督 H/8
        self.head_ds3  = nn.Conv2d(mid_ch, 1, 1)   # 深监督 H/16

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

        # ── Path 1 ────────────────────────────────────────────────────
        p1_3 = self.cfi1_43(c4, c3)        # [B, mid, H/16, W/16]
        p1_2 = self.cfi1_32(p1_3, c2)      # [B, mid, H/8,  W/8 ]
        p1_1 = self.cfi1_21(p1_2, c1)      # [B, mid, H/4,  W/4 ]

        # ── Path 2 ────────────────────────────────────────────────────
        p2_3 = self.ffm_3(self.cfi2_43(c4, c3), p1_3)    # [B, mid, H/16, W/16]
        p2_2 = self.ffm_2(self.cfi2_32(p2_3, c2), p1_2)   # [B, mid, H/8,  W/8 ]
        p2_1 = self.ffm_1(self.cfi2_21(p2_2, c1), p1_1)   # [B, mid, H/4,  W/4 ]

        # ── 输出（含中间深监督） ──────────────────────────────────────
        out_main = F.interpolate(self.head_main(p2_1), (H, W), mode='bilinear', align_corners=True)
        out_aux  = F.interpolate(self.head_aux(p1_1),  (H, W), mode='bilinear', align_corners=True)
        out_ds2  = F.interpolate(self.head_ds2(p2_2),  (H, W), mode='bilinear', align_corners=True)
        out_ds3  = F.interpolate(self.head_ds3(p2_3),  (H, W), mode='bilinear', align_corners=True)

        return out_main, out_aux, out_ds2, out_ds3
