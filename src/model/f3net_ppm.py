import torch
import torch.nn as nn
import torch.nn.functional as F

from .f3net import CFI, FFM, _cbr
from .poolnet import (
    ResNetLocate, _BACKBONE_TABLE, _get_decoder_cfg, _get_backbone
)


class F3NetPPM(nn.Module):
    """
    F3NetPPM: F3Net 双路解码器 + PoolNet PPM 全局上下文注入。

    改进动机
    --------
    之前 F3Net 系列在 ECSSD 上弱于 PoolNet 的根本原因：
    PoolNet 靠 PPM（Pyramid Pooling Module）在每一级解码时都注入了
    全局语义上下文（"整张图里哪里是目标"），而 F3Net 的 CFI/FFM
    仅做局部的跨层/跨路特征交互，缺乏这种全局引导信号。

    本模型将两者结合：
      - Encoder: 使用 ResNetLocate（backbone + PPM），
        在 H/16、H/8、H/4 三个尺度生成全局上下文 info3/info2/info1；
      - Decoder: 保留 F3Net 的双路 CFI+FFM 结构，
        每级 CFI 输出后**残差叠加**对应尺度的 PPM info，
        使每级解码都能感知全局目标位置。

    输出: (out_main, out_aux)，与 Trainer 兼容。
    """

    def __init__(self, mid_ch=128, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)
        channels = _BACKBONE_TABLE[backbone_name][2]
        ch1, ch2, ch3, ch4 = channels
        info_chs = cfg["dp_out"][:3]  # PPM info 在各尺度的通道数

        # ── Encoder: backbone + PPM ──────────────────────────────────
        self.base     = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        # ── PPM info 通道投影 → mid_ch ───────────────────────────────
        self.info_proj3 = nn.Conv2d(info_chs[0], mid_ch, 1, bias=False)
        self.info_proj2 = nn.Conv2d(info_chs[1], mid_ch, 1, bias=False)
        self.info_proj1 = nn.Conv2d(info_chs[2], mid_ch, 1, bias=False)

        # ── Path 1: 粗粒度解码 ───────────────────────────────────────
        self.cfi1_43 = CFI(ch4, ch3, mid_ch)
        self.cfi1_32 = CFI(mid_ch, ch2, mid_ch)
        self.cfi1_21 = CFI(mid_ch, ch1, mid_ch)

        # ── Path 2: 精细化解码（+ Path-1 反馈）──────────────────────
        self.cfi2_43 = CFI(ch4, ch3, mid_ch)
        self.ffm_3   = FFM(mid_ch)
        self.cfi2_32 = CFI(mid_ch, ch2, mid_ch)
        self.ffm_2   = FFM(mid_ch)
        self.cfi2_21 = CFI(mid_ch, ch1, mid_ch)
        self.ffm_1   = FFM(mid_ch)

        # ── PPM info 融合权重（初始为 0，训练逐渐激活）────────────────
        self.g3_alpha = nn.Parameter(torch.tensor(0.0))
        self.g2_alpha = nn.Parameter(torch.tensor(0.0))
        self.g1_alpha = nn.Parameter(torch.tensor(0.0))

        # ── 输出头 ────────────────────────────────────────────────────
        self.head_main = nn.Conv2d(mid_ch, 1, 1)
        self.head_aux  = nn.Conv2d(mid_ch, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('base') or name.startswith('backbone'):
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

        # backbone + PPM → feats=[c1,c2,c3,c4], infos=[info3,info2,info1]
        feats, infos = self.base(x)
        c1, c2, c3, c4 = feats

        # 投影 PPM info 到 mid_ch
        g3 = self.info_proj3(infos[0])   # [B, mid, H/16, W/16]
        g2 = self.info_proj2(infos[1])   # [B, mid, H/8,  W/8 ]
        g1 = self.info_proj1(infos[2])   # [B, mid, H/4,  W/4 ]

        # ── Path 1: CFI + PPM 全局上下文（可学习权重残差，初始为0）────
        p1_3 = self.cfi1_43(c4, c3) + self.g3_alpha * g3
        p1_2 = self.cfi1_32(p1_3, c2) + self.g2_alpha * g2
        p1_1 = self.cfi1_21(p1_2, c1) + self.g1_alpha * g1

        # ── Path 2: CFI + FFM 反馈 + PPM 全局上下文 ──────────────────
        p2_3 = self.ffm_3(self.cfi2_43(c4, c3), p1_3) + self.g3_alpha * g3
        p2_2 = self.ffm_2(self.cfi2_32(p2_3, c2), p1_2) + self.g2_alpha * g2
        p2_1 = self.ffm_1(self.cfi2_21(p2_2, c1), p1_1) + self.g1_alpha * g1

        out_main = F.interpolate(self.head_main(p2_1), (H, W), mode='bilinear', align_corners=True)
        out_aux  = F.interpolate(self.head_aux(p1_1),  (H, W), mode='bilinear', align_corners=True)

        return out_main, out_aux
