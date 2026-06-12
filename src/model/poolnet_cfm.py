import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)

class CFM(nn.Module):
    """F3Net 风格 CFM：两路各提取 2 层特征，乘法交互后残差回注，双路输出相加"""

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
        out1d = F.relu(self.bn1d(self.conv1d(down)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out1d)), inplace=True)

        out1l = F.relu(self.bn1l(self.conv1l(left)), inplace=True)
        out2l = F.relu(self.bn2l(self.conv2l(out1l)), inplace=True)

        # 乘法交互
        fuse = out2d * out2l

        # 残差回注 + 精炼
        out3d = F.relu(self.bn3d(self.conv3d(fuse)), inplace=True) + out1d
        out4d = F.relu(self.bn4d(self.conv4d(out3d)), inplace=True)

        out3l = F.relu(self.bn3l(self.conv3l(fuse)), inplace=True) + out1l
        out4l = F.relu(self.bn4l(self.conv4l(out3l)), inplace=True)

        return out4d + out4l


class DeepPoolLayerCFM(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse):
        super().__init__()
        self.need_x2   = need_x2
        self.need_fuse = need_fuse
        self.pool_scales = [2, 4, 8]

        self.conv_in = nn.Sequential(
            nn.Conv2d(k, k_out, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        if need_fuse:
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
            x = self.cfm(x, x_skip) + x_info

        return self._fam(x)

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
