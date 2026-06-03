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

        # c4 特征压缩
        p = self.ppms_pre(c4)                         # (B, 256, H/32, W/32)
        # PPM 生成多尺度增强特征
        xls = [p] + [
            F.interpolate(ppm(p), p.shape[2:], mode='bilinear', align_corners=True)
            for ppm in self.ppms
        ]
        # 4 路特征拼接: 融合深层语义与全局上下文信息
        xls = self.ppm_cat(torch.cat(xls, dim=1))     # [B, (256 × 4), H/32, W/32]

        # PPM 特征插值到 layer3/2/1 对应分辨率并投影到对应通道
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
        self.need_x2 = need_x2              # 是否上采样
        self.need_fuse = need_fuse          # 是否融合

        # 用不同 dilation rate 的卷积捕获多尺度感受野，保持空间分辨率
        dilations = [2, 4, 8]
        self.convs = nn.ModuleList([
            nn.Conv2d(k, k, 3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])

        self.relu = nn.ReLU(inplace=True)

        self.conv_sum = nn.Conv2d(k, k_out, 3, padding=1, bias=False)       # 通道调整
        self.conv_fuse = nn.Conv2d(k_out, k_out, 3, padding=1, bias=False)  # 特征融合

    def forward(self, x, x_skip=None, x_info=None):
        # 多尺度空洞卷积叠加形成增强特征
        out = x

        for conv in self.convs:
            out = out + conv(x)

        out = self.relu(out)

        # 对齐 skip 尺寸
        if self.need_x2:
            out = F.interpolate(
                out,
                x_skip.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 对齐通道
        out = self.conv_sum(out)

        # 融合 out + skip + info
        if self.need_fuse:
            out = self.conv_fuse(out + x_skip + x_info)

        return out

class ScoreLayer(nn.Module):
    """ 映射为单通道显著图 """

    def __init__(self, k):
        super().__init__()
        self.score = nn.Conv2d(k, 1, 1)

    def forward(self, x, target_size=None):
        x = self.score(x)

        # 恢复分辨率
        if target_size is not None:
            x = F.interpolate(x, target_size[2:], mode='bilinear', align_corners=True)
        return x

class PoolNet(nn.Module):
    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        # Encoder 形成多尺度特征与全局上下文信息
        self.base = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        # [c1, c2, c3, c4] 通道调整
        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        # Decoder 融合不同特征
        self.deep_pool = nn.ModuleList([
            DeepPoolLayer(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        # 输出头（最浅层 dp_out[-1]）
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

        # ── Backbone + PPM ──
        # feats: [c1, c2, c3, c4]
        # infos: [info3, info2, info1]
        feats, infos = self.base(x)

        # ── 通道对齐 ──
        feats = self.convert(feats)   

        # ── 倒序 ──
        # feats_r: [c4(256), c3(256), c2(256), c1(128)]
        feats_r = feats[::-1]

        # ── Decode ──
        # deep_pool[0]: c4 → fuse(c3, info3)
        # deep_pool[1]: → fuse(c2, info2)
        # deep_pool[2]: → fuse(c1, info1)
        # deep_pool[3]: 最浅层，无 fuse
        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        # ── 输出显著图 ──
        out = self.score(merge, input_size)  # (B, 1, H, W)
        return out
