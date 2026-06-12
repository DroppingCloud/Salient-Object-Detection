import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet18, ResNet18Pre, ResNet34Pre, ResNet50Pre

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
# 融合与上采样
_DP_X2   = [True,  True,  True,  False]
_DP_FUSE = [True,  True,  True,  False]

# backbone 名称 
_BACKBONE_TABLE = {
    "resnet18": (ResNet18Pre, ResNet18, [64, 128, 256, 512]),
    "resnet34": (ResNet34Pre, None,     [64, 128, 256, 512]),
    "resnet50": (ResNet50Pre, None,     [256, 512, 1024, 2048]),
}

# 各 backbone 对应的 decoder 通道配置
# out_ch      : ConvertLayer 输出通道 [c1, c2, c3, c4]
# dp_in/dp_out: DeepPoolLayer 输入/输出通道（从深到浅 4 级）
_DECODER_CFG = {
    "resnet18": {
        "out_ch": [128, 256, 256, 256],
        "dp_in":  [256, 256, 256, 128],
        "dp_out": [256, 256, 128, 128],
    },
    "resnet34": {
        "out_ch": [128, 256, 256, 256],
        "dp_in":  [256, 256, 256, 128],
        "dp_out": [256, 256, 128, 128],
    },
    "resnet50": {
        "out_ch": [256, 512, 512, 512],
        "dp_in":  [512, 512, 512, 256],
        "dp_out": [512, 512, 256, 256],
    },
}

def _get_backbone(name="resnet18", pretrained=True):
    """根据名称返回 backbone 实例和各 stage 通道列表"""
    pre_cls, scratch_cls, channels = _BACKBONE_TABLE[name]
    if pretrained:
        backbone = pre_cls()
    else:
        if scratch_cls is None:
            raise ValueError(f"{name} 没有随机初始化版本，请使用 pretrained=True")
        backbone = scratch_cls()
    return backbone, channels


def _get_decoder_cfg(name="resnet18"):
    """根据 backbone 名称返回 decoder 通道配置"""
    return _DECODER_CFG[name]

class ResNetLocate(nn.Module):
    """ Backbone + PPM """
    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        # 多尺度特征
        self.backbone, channels = _get_backbone(backbone_name, pretrained)
        cfg = _get_decoder_cfg(backbone_name)
        c4_ch = channels[3]

        # PPM 金字塔池化
        self.in_planes = cfg["dp_out"][0]
        self.ppms_pre = nn.Conv2d(c4_ch, self.in_planes, 1, bias=False)
        self.ppms = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(self.in_planes, self.in_planes, 1, bias=False),
                nn.ReLU(inplace=True),
            )
            for s in [1, 3, 5]
        ])
        # 卷积融合
        self.ppm_cat = nn.Sequential(
            nn.Conv2d(self.in_planes * 4, self.in_planes, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # PPM 特征插值到各 skip 尺寸 (对应 dp_out[0], dp_out[1], dp_out[2])
        self.out_planes = cfg["dp_out"][:3]
        self.infos = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_planes, ch, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for ch in self.out_planes
        ])

    def forward(self, x):
        # Backbone 多尺度特征
        c1, c2, c3, c4 = self.backbone(x)               # [64, 128, 256, 512]

        p = self.ppms_pre(c4)                         # (B, 256, H/32, W/32)
        # PPM 多尺度上下文
        xls = [p] + [
            F.interpolate(ppm(p), p.shape[2:], mode='bilinear', align_corners=True)
            for ppm in self.ppms
        ]
        xls = self.ppm_cat(torch.cat(xls, dim=1))

        # GGF: PPM 特征插值到 layer3/2/1 对应分辨率
        info3 = self.infos[0](
            F.interpolate(xls, c3.shape[2:], mode='bilinear', align_corners=True)
        )
        info2 = self.infos[1](
            F.interpolate(xls, c2.shape[2:], mode='bilinear', align_corners=True)
        )
        info1 = self.infos[2](
            F.interpolate(xls, c1.shape[2:], mode='bilinear', align_corners=True)
        )

        infos = [info3, info2, info1]

        return [c1, c2, c3, c4], infos

class ConvertLayer(nn.Module):
    """ 1×1 卷积调整各 skip 通道 """

    def __init__(self, in_list, out_list):
        super().__init__()
        self.converts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.ReLU(inplace=True))
            for i, o in zip(in_list, out_list)
        ])

    def forward(self, feats):
        return [conv(f) for conv, f in zip(self.converts, feats)]

class DeepPoolLayer(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse):
        super().__init__()

        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        self.pool_scales = [2, 4, 8]

        # 通道调整
        self.conv_in = nn.Sequential(
            nn.Conv2d(k, k_out, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # FAM 多尺度平均池化分支
        self.pool_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(k_out, k_out, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )
            for _ in self.pool_scales
        ])

        # 多尺度聚合融合卷积
        self.conv_out = nn.Sequential(
            nn.Conv2d(k_out, k_out, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def _adaptive_pool(self, x, scale):
        h, w = x.shape[2:]
        out_h = max(1, h // scale)
        out_w = max(1, w // scale)
        return F.adaptive_avg_pool2d(x, output_size=(out_h, out_w))

    def _fam(self, x):
        size = x.shape[2:]

        out = x
        for scale, conv in zip(self.pool_scales, self.pool_convs):
            pooled = self._adaptive_pool(x, scale)
            pooled = conv(pooled)
            pooled = F.interpolate(
                pooled,
                size=size,
                mode='bilinear',
                align_corners=True
            )
            out = out + pooled

        out = self.conv_out(out)
        return out

    def forward(self, x, x_skip=None, x_info=None):
        if self.need_x2:
            x = F.interpolate(
                x,
                size=x_skip.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        x = self.conv_in(x)

        if self.need_fuse:
            x = x + x_skip + x_info

        x = self._fam(x)

        return x
    
class ScoreLayer(nn.Module):
    """ 映射为单通道显著图 """

    def __init__(self, k):
        super().__init__()
        self.score = nn.Conv2d(k, 1, 1)

    def forward(self, x, target_size=None):
        x = self.score(x)
        if target_size is not None:
            x = F.interpolate(x, target_size[2:], mode='bilinear', align_corners=True)
        return x

class PoolNet(nn.Module):
    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayer(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        # 输出头
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

        # deep_pool[0..2]: 逐级上采样融合 skip+info，deep_pool[3]: 无 fuse
        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        out = self.score(merge, input_size)
        return out
