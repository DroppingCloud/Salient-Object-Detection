import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNet18Locate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _IN_CH, _OUT_CH, _DP_IN, _DP_OUT, _DP_X2, _DP_FUSE,
)

class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 16)

        self.mlp = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg) + self.mlp(mx)
        return self.sigmoid(attn)


class FBDA(nn.Module):

    def __init__(self, ch, reduction=16, spatial_mid=None):
        super().__init__()

        if spatial_mid is None:
            spatial_mid = max(ch // 4, 16)

        self.ca = ChannelAttention(ch, reduction=reduction)

        self.fg_attn = nn.Sequential(
            nn.Conv2d(ch, spatial_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_mid, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self.bg_attn = nn.Sequential(
            nn.Conv2d(ch, spatial_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_mid, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x_ca = x * self.ca(x)

        a_fg = self.fg_attn(x_ca)
        a_bg = self.bg_attn(x_ca)

        # 背景抑制不要太狠，加入 0.5 系数更稳
        x_att = x_ca * (1.0 + a_fg) * (1.0 - 0.5 * a_bg)
        x_att = self.refine(x_att)

        return x + self.gamma * x_att


class PoolNetFBDA(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.base     = ResNet18Locate(pretrained=pretrained)
        self.backbone = self.base.backbone

        self.convert = ConvertLayer(_IN_CH, _OUT_CH)

        self.deep_pool = nn.ModuleList([
            DeepPoolLayer(_DP_IN[i], _DP_OUT[i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        # FBDA 插入在 deep_pool[0/1/2] 之后（输出通道分别为 256, 256, 128）
        self.fbda0 = FBDA(_DP_OUT[0])   # 256ch
        self.fbda1 = FBDA(_DP_OUT[1])   # 256ch
        self.fbda2 = FBDA(_DP_OUT[2])   # 128ch

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

    def forward(self, x):
        input_size = x.shape

        feats, infos = self.base(x)
        feats = self.convert(feats)
        feats_r = feats[::-1]   # [c4(256), c3(256), c2(256), c1(128)]

        # deep_pool[0]: c4 → fuse(c3, info3) → FBDA   256ch H/16
        m0 = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        # deep_pool[1]: → fuse(c2, info2) → FBDA       256ch H/8
        m1 = self.fbda1(self.deep_pool[1](m0, feats_r[2], infos[1]))
        # deep_pool[2]: → fuse(c1, info1) → FBDA       128ch H/4
        m2 = self.fbda2(self.deep_pool[2](m1, feats_r[3], infos[2]))
        # deep_pool[3]: 最浅层，无 fuse，无 FBDA        128ch H/4
        m3 = self.deep_pool[3](m2)

        return self.score(m3, input_size)   # [B, 1, H, W]
