"""
PoolNetGateCFM: PoolNet + 解耦双分支门控 + CFM 跨层交互

在 PoolNetGate (Channel Gate ∥ Spatial Gate → Fuse) 基础上，
引入 F3Net 风格的 CFM 机制: 用门控净化后的 skip 与 decoder 做
双分支乘法交互 + 残差回注，实现更深层次的跨层特征融合。

融合流程 (每个 decoder 节点):
  1. DualGateFusion: Channel Gate ∥ Spatial Gate → concat → conv → gated_skip
  2. CFM 交互: decoder 和 gated_skip 各经 2 层 conv 后做 element-wise product
  3. 交互特征残差回注两路，各经 2 层 conv 精炼
  4. 双路输出相加 + x_info
  5. FAM 多尺度增强

与 PoolNetGate 的区别:
  - PoolNetGate: x + gated_skip + info (加法融合)
  - PoolNetGateCFM: CFM(x, gated_skip) + info (乘法交互融合)
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
    以 decoder 语义为指导，对 skip 各通道加权。
    decoder GAP → FC → ReLU → FC → Sigmoid → (B, C, 1, 1)
    """

    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 16)
        self.fc1 = nn.Conv2d(ch, mid, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, ch, 1, bias=True)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 2.0)

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
        nn.init.constant_(self.conv2.bias, 2.0)

    def forward(self, decoder_feat, skip_feat):
        feat = torch.cat([decoder_feat, skip_feat], dim=1)
        attn = torch.sigmoid(self.conv2(self.relu(self.bn(self.conv1(feat)))))
        return skip_feat * attn


# ─────────────────────────────────────────────────────────────
# Dual Gate Fusion: 双分支解耦 → 融合
# ─────────────────────────────────────────────────────────────

class DualGateFusion(nn.Module):
    """
    解耦双分支门控: Channel Gate ∥ Spatial Gate → concat → conv → gated_skip
    """

    def __init__(self, ch):
        super().__init__()
        self.channel_gate = ChannelGate(ch)
        self.spatial_gate = SpatialGate(decoder_ch=ch, skip_ch=ch)

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
# CFM: F3Net 风格跨层特征交互 (作用在 gated_skip 与 decoder 之间)
# ─────────────────────────────────────────────────────────────

class CFM(nn.Module):
    """
    Cross-Feature Module (参考 F3Net 原版实现)。

    输入: decoder 特征 (down) 和 净化后的 skip 特征 (left)
    流程:
      down → conv-bn-relu → conv-bn-relu → out2d
      left → conv-bn-relu → conv-bn-relu → out2l
      fuse = out2d * out2l                         (乘法交互)
      out3d = conv-bn-relu(fuse) + out1d           (残差回注 decoder)
      out4d = conv-bn-relu(out3d)
      out3l = conv-bn-relu(fuse) + out1l           (残差回注 skip)
      out4l = conv-bn-relu(out3l)
    输出: out4d + out4l (双路增强结果相加)
    """

    def __init__(self, ch):
        super().__init__()
        # decoder (down) 分支
        self.conv1d = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1d = nn.BatchNorm2d(ch)
        self.conv2d = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2d = nn.BatchNorm2d(ch)
        self.conv3d = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm2d(ch)
        self.conv4d = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn4d = nn.BatchNorm2d(ch)

        # skip (left) 分支
        self.conv1l = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1l = nn.BatchNorm2d(ch)
        self.conv2l = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2l = nn.BatchNorm2d(ch)
        self.conv3l = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn3l = nn.BatchNorm2d(ch)
        self.conv4l = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn4l = nn.BatchNorm2d(ch)

    def forward(self, down, left):
        # 各自前 2 层特征提取
        out1d = F.relu(self.bn1d(self.conv1d(down)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out1d)), inplace=True)

        out1l = F.relu(self.bn1l(self.conv1l(left)), inplace=True)
        out2l = F.relu(self.bn2l(self.conv2l(out1l)), inplace=True)

        # 乘法交互: 找到两路共同激活的区域
        fuse = out2d * out2l

        # 残差回注 + 后 2 层精炼
        out3d = F.relu(self.bn3d(self.conv3d(fuse)), inplace=True) + out1d
        out4d = F.relu(self.bn4d(self.conv4d(out3d)), inplace=True)

        out3l = F.relu(self.bn3l(self.conv3l(fuse)), inplace=True) + out1l
        out4l = F.relu(self.bn4l(self.conv4l(out3l)), inplace=True)

        return out4d + out4l


# ─────────────────────────────────────────────────────────────
# DeepPoolLayer: DualGate → CFM → FAM
# ─────────────────────────────────────────────────────────────

class DeepPoolLayerGatedCFM(nn.Module):
    """
    Decoder 单元:
      1. 上采样 + 通道调整
      2. DualGateFusion: 对 skip 做双分支门控净化
      3. CFM: 净化后的 skip 与 decoder 做跨层乘法交互
      4. + x_info (全局上下文直接加入)
      5. FAM 多尺度增强
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

        # 门控 + CFM (仅融合节点)
        if need_fuse:
            self.dual_gate = DualGateFusion(k_out)
            self.cfm = CFM(k_out)

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
            # 双分支门控净化 skip
            gated_skip = self.dual_gate(x, x_skip)
            # CFM 跨层交互: decoder 与净化后的 skip 做乘法交互
            x = self.cfm(x, gated_skip) + x_info

        x = self._fam(x)
        return x


# ─────────────────────────────────────────────────────────────
# PoolNetGateCFM 主模型
# ─────────────────────────────────────────────────────────────

class PoolNetGateCFM(nn.Module):
    """
    PoolNet + 解耦双分支门控 + CFM。

    Encoder → PPM → ConvertLayer → Decoder(DualGate + CFM + FAM) → Score
    """

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerGatedCFM(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
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
