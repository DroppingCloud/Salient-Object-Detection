import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)
from .f3net import CFI


class DeepPoolLayerCFI(nn.Module):
    """
    PoolNet DeepPoolLayer 的改进版：用 CFI 双向注意力替换原始的
    简单加法融合（out + skip + info → conv_fuse）。

    原始 DeepPoolLayer 融合:
        out = conv_fuse(out + x_skip + x_info)
        — 三路直接相加，无区分能力

    CFI 改进融合:
        out = CFI(out, x_skip) + x_info
        — CFI 对 out（高层）与 x_skip（低层）做双向注意力门控，
          使两路特征互相引导对方筛选有用信息；
          x_info（PPM全局上下文）随后以残差形式叠加，
          保留全局语义不被注意力错误抑制。
    """

    def __init__(self, k, k_out, need_x2, need_fuse):
        super().__init__()
        self.need_x2   = need_x2
        self.need_fuse = need_fuse

        dilations = [2, 4, 8]
        self.convs = nn.ModuleList([
            nn.Conv2d(k, k, 3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.relu     = nn.ReLU(inplace=True)
        self.conv_sum = nn.Conv2d(k, k_out, 3, padding=1, bias=False)

        if need_fuse:
            # CFI: 双向注意力融合 out(k_out) 与 skip(k_out)
            self.cfi = CFI(k_out, k_out, k_out)

    def forward(self, x, x_skip=None, x_info=None):
        out = x
        for conv in self.convs:
            out = out + conv(x)
        out = self.relu(out)

        if self.need_x2:
            out = F.interpolate(out, x_skip.shape[2:], mode='bilinear', align_corners=True)

        out = self.conv_sum(out)

        if self.need_fuse:
            # CFI 双向注意力融合高层(out)与低层(x_skip)
            out = self.cfi(out, x_skip)
            # PPM 全局上下文以残差方式叠加，避免被注意力错误抑制
            out = out + x_info

        return out


class PoolNetCFI(nn.Module):
    """
    PoolNet-CFI: 跨模型组合 —— 用 F3Net 的 CFI 双向注意力模块
    替换 PoolNet 解码器中的简单加法特征融合。

    改进动机
    --------
    PoolNet 原始解码器（DeepPoolLayer）在融合阶段直接对三路特征求和：
        out = conv_fuse(out + x_skip + x_info)
    这忽略了高层（语义）与低层（细节）特征之间的语义鸿沟，
    两路特征对彼此没有任何引导。

    F3Net 的 CFI 通过双向注意力门控：
      - gate_hi: 低层细节引导高层特征在哪些位置/通道保留语义
      - gate_lo: 高层语义引导低层特征在哪些位置/通道保留细节
    能有效减少特征噪声，提升跨层融合质量。

    本模型保留 PoolNet 的 PPM 全局上下文架构（整体框架不变），
    仅将 DeepPoolLayer 的 skip 融合替换为 CFI，
    PPM info 以残差方式保留。

    输出格式: 单张量 (B, 1, H, W)，与 Trainer 兼容。
    """

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base     = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCFI(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        self.score = ScoreLayer(cfg["dp_out"][-1])
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone') or name.startswith('base.backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
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
