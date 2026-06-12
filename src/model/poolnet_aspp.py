import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)

# _DP_ASPP[i] = True 表示第 i 个 decoder 单元使用 ASPP，否则使用 FAM
# decoder 顺序：从深到浅 [0=最深, 1, 2, 3=最浅]
# 只在最深2层（0, 1）使用 ASPP
_DP_ASPP = [True, True, False, False]


# ─────────────────────────────────────────────────────────────
# ASPPBlock: 标准 ASPP，每个分支输出 k_out，concat 后投影到 k_out
# ─────────────────────────────────────────────────────────────

class ASPPBlock(nn.Module):
    """标准 ASPP 模块。

    分支：
      - 1×1 卷积（rate=1）
      - 3×3 空洞卷积 × 3（rate = 2, 4, 8）
      - 全局平均池化 → 1×1 卷积（无 BN）

    每个分支独立输出 k_out 通道，concat (5 × k_out) 后通过
    1×1 投影卷积降回 k_out。

    所有 Conv-BN-ReLU 采用 Kaiming 初始化。
    """

    def __init__(self, k_in, k_out, dilations=(2, 4, 8)):
        super().__init__()
        num_branches = 1 + len(dilations) + 1   # 1×1 + 空洞卷积 + GAP

        def cbr(cin, cout, k=1, p=0, d=1):
            m = nn.Sequential(
                nn.Conv2d(cin, cout, k, padding=p, dilation=d, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
            nn.init.kaiming_normal_(m[0].weight, mode='fan_out', nonlinearity='relu')
            return m

        # 1×1 卷积分支
        self.conv1x1 = cbr(k_in, k_out)

        # 空洞卷积分支
        self.atrous_convs = nn.ModuleList([
            cbr(k_in, k_out, k=3, p=d, d=d)
            for d in dilations
        ])

        # 全局平均池化分支（无 BN）
        gap_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        nn.init.kaiming_normal_(gap_conv.weight, mode='fan_out', nonlinearity='relu')
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            gap_conv,
            nn.ReLU(inplace=True),
        )

        # concat 后投影
        self.project = cbr(num_branches * k_out, k_out)

    def forward(self, x):
        size = x.shape[2:]
        branches = [self.conv1x1(x)]
        for conv in self.atrous_convs:
            branches.append(conv(x))
        gap_out = F.interpolate(self.gap(x), size=size, mode='bilinear', align_corners=True)
        branches.append(gap_out)
        return self.project(torch.cat(branches, dim=1))


# ─────────────────────────────────────────────────────────────
# DeepPoolLayerASPP: 上采样 → ASPP → (+skip +info)
# ─────────────────────────────────────────────────────────────

class DeepPoolLayerASPP(nn.Module):
    """Decoder 单元（深层）：上采样 → ASPP 增强 → 融合 skip+info"""

    def __init__(self, k, k_out, need_x2, need_fuse, dilations=(2, 4, 8)):
        super().__init__()
        self.need_x2   = need_x2
        self.need_fuse = need_fuse

        self.aspp = ASPPBlock(k, k_out, dilations=dilations)

    def forward(self, x, x_skip=None, x_info=None):
        if self.need_x2:
            x = F.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=True)

        x = self.aspp(x)

        if self.need_fuse:
            x = x + x_skip + x_info

        return x


# ─────────────────────────────────────────────────────────────
# PoolNetASPP 主模型
# ─────────────────────────────────────────────────────────────

class PoolNetASPP(nn.Module):
    """PoolNet + ASPP（最深 2 层）+ FAM（浅层 2 层）"""

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base     = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerASPP(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            if _DP_ASPP[i] else
            DeepPoolLayer(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        self.score = ScoreLayer(cfg["dp_out"][-1])

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone') or name.startswith('base.backbone'):
                continue
            # ASPP 内部已用 Kaiming 初始化，跳过避免覆盖
            if any(name.startswith(f'deep_pool.{i}.aspp') for i in range(4)):
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

        out = self.score(merge, input_size)
        return out
