import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)


class SideHead(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        mid = max(in_ch // 2, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, x, target_size):
        x = self.conv(x)
        return F.interpolate(x, target_size[2:], mode='bilinear', align_corners=True)


class PoolNetDS(nn.Module):

    loss_weights = (1.0, 0.4, 0.2)
    simple_aux_loss = True

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base     = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayer(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        self.score = ScoreLayer(cfg["dp_out"][-1])

        self.side1 = SideHead(cfg["dp_out"][1])   # deep_pool[1] 输出通道
        self.side2 = SideHead(cfg["dp_out"][2])   # deep_pool[2] 输出通道

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone') or name.startswith('base.backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        input_size = x.shape

        feats, infos = self.base(x)
        feats = self.convert(feats)
        feats_r = feats[::-1]

        m0 = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        m1 = self.deep_pool[1](m0, feats_r[2], infos[1])
        m2 = self.deep_pool[2](m1, feats_r[3], infos[2])
        m3 = self.deep_pool[3](m2)

        main    = self.score(m3, input_size)
        aux_dp2 = self.side2(m2, input_size)
        aux_dp1 = self.side1(m1, input_size)

        return main, aux_dp2, aux_dp1
