import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)

class CFM(nn.Module):
    def __init__(self, ch):
        super().__init__()

        def cbr(cin, cout, k=3, p=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.deep_proj = cbr(ch, ch)
        self.skip_proj = cbr(ch, ch)

        self.deep_refine = cbr(ch, ch)
        self.skip_refine = cbr(ch, ch)

        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        self.info_proj = cbr(ch, ch)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, out, x_skip, x_info):
        if x_info.shape[-2:] != out.shape[-2:]:
            x_info = F.interpolate(
                x_info,
                size=out.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        common = self.deep_proj(out) * self.skip_proj(x_skip)

        out_refined = out + self.deep_refine(common)
        skip_refined = x_skip + self.skip_refine(common)

        fused = self.fuse(torch.cat([out_refined, skip_refined], dim=1))

        base = out + x_skip + x_info
        enhanced = self.info_proj(fused + x_info)

        return base + self.gamma * enhanced


class DeepPoolLayerCFM(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse):
        super().__init__()
        self.need_x2   = need_x2
        self.need_fuse = need_fuse

        self.conv_sum = nn.Conv2d(k, k_out, 3, padding=1, bias=False)

        if need_fuse:
            self.cfm = CFM(k_out)

    def forward(self, x, x_skip=None, x_info=None):
        if self.need_x2:
            x = F.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=True)

        out = self.conv_sum(x)

        if self.need_fuse:
            out = self.cfm(out, x_skip, x_info)

        return out

class PoolNetCFM(nn.Module):

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base     = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCFM(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        self.score = ScoreLayer(cfg["dp_out"][-1])

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
        feats_r = feats[::-1]   # [c4, c3, c2, c1]

        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        out = self.score(merge, input_size)
        return out
