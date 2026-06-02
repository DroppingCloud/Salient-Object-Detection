import torch.nn as nn

from .poolnet import (
    ResNet18Locate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _IN_CH, _OUT_CH, _DP_IN, _DP_OUT, _DP_X2, _DP_FUSE,
)
from .poolnet_cfm import DeepPoolLayerCFM
from .poolnet_fbda import FBDA

class PoolNetCFMFBDA(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.base     = ResNet18Locate(pretrained=pretrained)
        self.backbone = self.base.backbone

        self.convert = ConvertLayer(_IN_CH, _OUT_CH)

        # CFM 解码层
        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCFM(_DP_IN[i], _DP_OUT[i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        # FBDA 插在 deep_pool[0/1/2] 之后（输出通道分别为 256, 256, 128）
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

        m0 = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])

        m1 = self.fbda1(self.deep_pool[1](m0, feats_r[2], infos[1]))

        m2 = self.fbda2(self.deep_pool[2](m1, feats_r[3], infos[2]))

        m3 = self.deep_pool[3](m2)

        return self.score(m3, input_size)   # [B, 1, H, W]
