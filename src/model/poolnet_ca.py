import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNet18Locate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _IN_CH, _OUT_CH, _DP_IN, _DP_OUT, _DP_X2, _DP_FUSE,
)

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)  # (B, C)
        return x * w.unsqueeze(-1).unsqueeze(-1)


class DeepPoolLayerCA(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse, reduction=16):
        super().__init__()
        self.need_x2 = need_x2
        self.need_fuse = need_fuse

        # 多尺度空洞卷积
        dilations = [2, 4, 8]
        self.convs = nn.ModuleList([
            nn.Conv2d(k, k, 3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.relu = nn.ReLU(inplace=True)

        # 通道注意力（插入点：空洞卷积后、conv_sum 前）
        self.ca = ChannelAttention(k, reduction=reduction)

        # 通道调整
        self.conv_sum = nn.Conv2d(k, k_out, 3, padding=1, bias=False)
        # 特征融合
        self.conv_fuse = nn.Conv2d(k_out, k_out, 3, padding=1, bias=False)

    def forward(self, x, x_skip=None, x_info=None):
        # 多尺度空洞卷积叠加
        out = x
        for conv in self.convs:
            out = out + conv(x)
        out = self.relu(out)

        # 通道注意力加权
        out = self.ca(out)

        # 对齐 skip 尺寸
        if self.need_x2:
            out = F.interpolate(
                out, x_skip.shape[2:],
                mode='bilinear', align_corners=True
            )

        # 对齐通道
        out = self.conv_sum(out)

        # 融合 out + skip + info
        if self.need_fuse:
            out = self.conv_fuse(out + x_skip + x_info)

        return out


class PoolNetCA(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.base = ResNet18Locate(pretrained=pretrained)
        self.backbone = self.base.backbone

        self.convert = ConvertLayer(_IN_CH, _OUT_CH)

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCA(_DP_IN[i], _DP_OUT[i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        self.score = ScoreLayer(128)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone') or name.startswith('base.backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        input_size = x.shape

        feats, infos = self.base(x)
        feats = self.convert(feats)
        feats_r = feats[::-1]

        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        out = self.score(merge, input_size)
        return out
