import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)


class PoolNetRA(nn.Module):
    """
    PoolNet-RA: PoolNet + Reverse Attention 自校正机制。

    改进动机
    --------
    PoolNet 单次前向解码时，网络会在高置信区域（明显前景）快速收敛，
    但对低对比度区域、小目标、边界附近的"难样本"往往预测不足。

    Reverse Attention（反向注意力）的思路：
      1. 第一次解码（Decoder-1）产生初始显著图 pred_1；
      2. 计算反向注意力图：RA = 1 - sigmoid(pred_1)
         — RA 在 pred_1 低置信的区域（遗漏/模糊区）取大值，
           在高置信前景区域取小值，即"聚焦未检出的区域"；
      3. 用 RA 重新加权最深层特征 c4，得到 c4_ra；
         c4_ra 中背景语义被压制，遗漏区域被放大；
      4. 第二次解码（Decoder-2，独立参数）以 c4_ra 为输入，
         产生精细化显著图 pred_2；
      5. 训练时返回 (pred_2, pred_1)，
         Trainer 以 2:1 权重监督，迫使两次解码均有效。

    输出格式
    --------
    训练: (pred_2, pred_1) — pred_2 为主输出（精细化），pred_1 为辅助
    推理: pred_2
    """

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        # ── 共享 Encoder ─────────────────────────────────────────────
        self.base     = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        # ── Decoder-1（初次预测）──────────────────────────────────────
        self.deep_pool1 = nn.ModuleList([
            DeepPoolLayer(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])
        self.score1 = ScoreLayer(cfg["dp_out"][-1])

        # ── Decoder-2（反向注意力精细化）─────────────────────────────
        # 输入 c4 已被 RA 重新加权，其余 skip 和 info 不变
        self.deep_pool2 = nn.ModuleList([
            DeepPoolLayer(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])
        self.score2 = ScoreLayer(cfg["dp_out"][-1])

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone') or name.startswith('base.backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _decode(self, deep_pool, score, feats_r, infos, input_size):
        merge = deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = deep_pool[-1](merge)
        return score(merge, input_size)

    def forward(self, x):
        input_size = x.shape

        # ── Encoder ───────────────────────────────────────────────────
        feats, infos = self.base(x)
        feats   = self.convert(feats)
        feats_r = feats[::-1]   # [c4, c3, c2, c1]

        # ── Decoder-1: 初次预测 ───────────────────────────────────────
        pred_1 = self._decode(self.deep_pool1, self.score1, feats_r, infos, input_size)

        # ── Reverse Attention ─────────────────────────────────────────
        # RA 聚焦于初次预测置信度低的区域（遗漏/模糊）
        ra = 1.0 - torch.sigmoid(pred_1.detach())         # (B, 1, H, W)
        ra_c4 = F.interpolate(                            # 下采样到 c4 分辨率
            ra, feats_r[0].shape[2:], mode='bilinear', align_corners=True
        )

        # 用 RA 重加权 c4，其余特征不变
        feats_r2    = list(feats_r)
        feats_r2[0] = feats_r[0] * ra_c4                 # 放大遗漏区，压制高置信区

        # ── Decoder-2: 精细化预测 ─────────────────────────────────────
        pred_2 = self._decode(self.deep_pool2, self.score2, feats_r2, infos, input_size)

        if self.training:
            return pred_2, pred_1   # (主输出×2.0, 辅助×1.0)
        return pred_2
