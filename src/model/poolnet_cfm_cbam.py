import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)
from .poolnet_cfm import CFM
from .f3net_cbam import CBAM


class ResidualCBAM(nn.Module):
    """
    带可学习残差门控的 CBAM。

    标准 CBAM 初始时 sigmoid(0)=0.5，会把特征压缩到一半，
    与 PoolNetCFM 的 std=0.01 保守初始化冲突。

    解决方案：output = x + alpha * (cbam(x) - x)
      - alpha 初始为 0 → 完全等于 x（恒等映射，与 PoolNetCFM 一致）
      - 训练过程中 alpha 逐渐学习，决定 CBAM 的介入程度
    """
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.cbam  = CBAM(ch, reduction)
        self.alpha = nn.Parameter(torch.tensor(0.0))   # 初始为 0

    def forward(self, x):
        return x + self.alpha * (self.cbam(x) - x)


class DeepPoolLayerCFM_CBAM(nn.Module):
    """
    PoolNetCFM 解码层改进版：skip 特征经 ResidualCBAM 精炼后再进入 CFM。
    ResidualCBAM 以 alpha=0 初始化，保证与 PoolNetCFM 行为完全一致，
    训练中逐步激活注意力。
    """

    def __init__(self, k, k_out, need_x2, need_fuse):
        super().__init__()
        self.need_x2   = need_x2
        self.need_fuse = need_fuse
        self.conv_sum  = nn.Conv2d(k, k_out, 3, padding=1, bias=False)

        if need_fuse:
            self.cbam_skip = ResidualCBAM(k_out)   # 残差门控 CBAM，初始恒等
            self.cfm       = CFM(k_out)

    def forward(self, x, x_skip=None, x_info=None):
        if self.need_x2:
            x = F.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=True)

        out = self.conv_sum(x)

        if self.need_fuse:
            x_skip = self.cbam_skip(x_skip)    # 残差精炼：初始时等价于 x_skip
            out    = self.cfm(out, x_skip, x_info)

        return out


class PoolNetCFM_CBAM(nn.Module):
    """
    PoolNetCFM-CBAM: 在 PoolNetCFM 的每个 skip 特征上加 CBAM 精炼。

    改进动机
    --------
    PoolNetCFM 用 CFM 的互信息门控（deep_proj × skip_proj）替换了 PoolNet
    的简单加法融合，这已经是一个有效改进。但 CFM 的 skip_proj 接受的是
    ConvertLayer 输出的特征，其中仍混杂着大量背景响应：
      - 无关通道（如草地/天空纹理）拉低互信息门控的信噪比
      - 背景区域的高激活使门控值偏向错误位置

    CBAM 在 skip 进入 CFM 之前做双重筛选：
      1. 通道注意力：抑制无关特征通道
      2. 空间注意力：聚焦显著目标区域
    净化后的 skip 送入 CFM，互信息门控更准确，融合质量更高。

    输出: 单张量 (B, 1, H, W)。
    """

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base     = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCFM_CBAM(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        self.score = ScoreLayer(cfg["dp_out"][-1])
        self._init_weights()

    def _init_weights(self):
        # 与 PoolNetCFM 保持完全一致的 std=0.01 保守初始化
        # 这使 CFM 在训练初期近似于恒等映射（gamma=0），
        # 避免破坏 PoolNet backbone 已学习的特征分布
        for name, m in self.named_modules():
            if name.startswith('backbone') or name.startswith('base.backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        input_size = x.shape

        feats, infos = self.base(x)
        feats = self.convert(feats)
        feats_r = feats[::-1]

        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        return self.score(merge, input_size)
