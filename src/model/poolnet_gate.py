"""
PoolNetGate: PoolNet + 解耦双分支门控 (Channel Gate ∥ Spatial Gate → Fuse)

改进动机:
  PoolNet 的 DeepPoolLayer 直接 x + x_skip + x_info，skip 中的噪声无差别混入。

  本模块对 skip 特征施加双分支并行门控:
  - Channel Branch: 通道注意力加权 skip → 得到"通道精炼"视角
  - Spatial Branch: 空间注意力加权 skip → 得到"空间精炼"视角
  - Fuse: 两路 concat → conv 融合为统一的 gated_skip
  - 最终: x + gated_skip + x_info → FAM

  解耦的好处:
  - 通道和空间各自独立学习，避免串联时梯度耦合导致一方主导另一方退化
  - 融合卷积可以学习两种视角的互补组合方式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)


# ─────────────────────────────────────────────────────────────
# Channel Gate: 通道注意力分支
# ─────────────────────────────────────────────────────────────

class ChannelGate(nn.Module):
    """
    通道注意力: 以 decoder 语义指导，对 skip 各通道加权。
    decoder GAP → FC → ReLU → FC → Sigmoid → (B, C, 1, 1)
    输出: skip * weight (通道精炼后的 skip)
    """

    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 16)
        self.fc1 = nn.Conv2d(ch, mid, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, ch, 1, bias=True)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 2.0)  # sigmoid(2)≈0.88

    def forward(self, decoder_feat, skip_feat):
        w = torch.sigmoid(self.fc2(self.relu(self.fc1(
            F.adaptive_avg_pool2d(decoder_feat, 1)
        ))))
        return skip_feat * w


# ─────────────────────────────────────────────────────────────
# Spatial Gate: 空间注意力分支
# ─────────────────────────────────────────────────────────────

class SpatialGate(nn.Module):
    """
    空间门控: 对 skip 各位置生成注意力权重。
    concat(decoder, skip) → 3×3 conv → BN → ReLU → 1×1 → Sigmoid → (B, 1, H, W)
    输出: skip * spatial_map (空间精炼后的 skip)
    """

    def __init__(self, decoder_ch, skip_ch):
        super().__init__()
        in_ch = decoder_ch + skip_ch
        mid = max(in_ch // 4, 16)
        self.conv1 = nn.Conv2d(in_ch, mid, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, 1, 1, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 2.0)  # sigmoid(2)≈0.88

    def forward(self, decoder_feat, skip_feat):
        feat = torch.cat([decoder_feat, skip_feat], dim=1)
        attn = torch.sigmoid(self.conv2(self.relu(self.bn(self.conv1(feat)))))
        return skip_feat * attn


# ─────────────────────────────────────────────────────────────
# Dual Gate Fusion: 双分支解耦 → 融合
# ─────────────────────────────────────────────────────────────

class DualGateFusion(nn.Module):
    """
    解耦双分支门控融合模块。

    skip_feat ──┬── ChannelGate(decoder, skip) → ch_out  ──┐
                │                                           ├─ concat → conv → gated_skip
                └── SpatialGate(decoder, skip) → sp_out  ──┘

    两路并行产生不同视角的精炼结果，再通过 3×3 conv 融合。
    """

    def __init__(self, ch):
        super().__init__()
        self.channel_gate = ChannelGate(ch)
        self.spatial_gate = SpatialGate(decoder_ch=ch, skip_ch=ch)

        # 融合两路: 2*ch → ch
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, decoder_feat, skip_feat):
        ch_out = self.channel_gate(decoder_feat, skip_feat)
        sp_out = self.spatial_gate(decoder_feat, skip_feat)
        return self.fuse(torch.cat([ch_out, sp_out], dim=1))


# ─────────────────────────────────────────────────────────────
# DeepPoolLayer with Dual Gate
# ─────────────────────────────────────────────────────────────

class DeepPoolLayerGated(nn.Module):
    """
    Decoder 单元:
      1. 上采样 + 通道调整
      2. DualGateFusion 对 skip 做双分支门控融合
      3. decoder + gated_skip + info
      4. FAM 多尺度增强
    """

    def __init__(self, k, k_out, need_x2, need_fuse):
        super().__init__()
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        self.pool_scales = [2, 4, 8]

        # 通道调整
        self.conv_in = nn.Sequential(
            nn.Conv2d(k, k_out, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 双分支门控融合 (仅融合节点)
        if need_fuse:
            self.dual_gate = DualGateFusion(k_out)

        # FAM
        self.pool_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(k_out, k_out, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for _ in self.pool_scales
        ])
        self.conv_out = nn.Sequential(
            nn.Conv2d(k_out, k_out, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def _adaptive_pool(self, x, scale):
        h, w = x.shape[2:]
        return F.adaptive_avg_pool2d(x, (max(1, h // scale), max(1, w // scale)))

    def _fam(self, x):
        """Feature Aggregate Module: 多尺度池化增强"""
        size = x.shape[2:]
        out = x
        for scale, conv in zip(self.pool_scales, self.pool_convs):
            pooled = conv(self._adaptive_pool(x, scale))
            pooled = F.interpolate(pooled, size, mode='bilinear', align_corners=True)
            out = out + pooled
        return self.conv_out(out)

    def forward(self, x, x_skip=None, x_info=None):
        if self.need_x2:
            x = F.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=True)

        x = self.conv_in(x)

        if self.need_fuse:
            gated_skip = self.dual_gate(x, x_skip)
            x = x + gated_skip + x_info

        x = self._fam(x)
        return x


# ─────────────────────────────────────────────────────────────
# PoolNetGate 主模型
# ─────────────────────────────────────────────────────────────

class PoolNetGate(nn.Module):
    """
    PoolNet + 解耦双分支门控 (Channel ∥ Spatial → Fuse)。

    Encoder → PPM → ConvertLayer → Decoder(DualGate + FAM) → Score
    """

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerGated(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        self.score = ScoreLayer(cfg["dp_out"][-1])
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone') or name.startswith('base.backbone'):
                continue
            if 'channel_gate' in name or 'spatial_gate' in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        input_size = x.shape

        feats, infos = self.base(x)
        feats = self.convert(feats)
        feats_r = feats[::-1]   # [c4, c3, c2, c1]

        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        return self.score(merge, input_size)
