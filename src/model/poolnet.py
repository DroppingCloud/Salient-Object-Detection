import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet18 import ResNet18, ResNet18Pre

class ResNet18Locate(nn.Module):
    """ Backbone + PPM """
    def __init__(self):
        super().__init__()
        # 多尺度特征
        self.resnet = ResNet18Pre()

        # PPM 金字塔池化
        self.in_planes = 256
        self.ppms_pre = nn.Conv2d(512, self.in_planes, 1, bias=False)
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

        # PPM 特征插值到各 skip 尺寸
        self.out_planes = [256, 256, 128]
        self.infos = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_planes, ch, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for ch in self.out_planes
        ])

    def forward(self, x):
        # Backbone 多尺度特征
        c1, c2, c3, c4 = self.resnet(x)               # [64, 128, 256, 512]

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

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
# Convert Layer 
_IN_CH  = [64, 128, 256, 512]       # [c1, c2, c3, c4]
_OUT_CH = [128, 256, 256, 256]  

# deep_pool 配置
_DP_IN    = [256, 256, 256, 128]
_DP_OUT   = [256, 256, 128, 128]
_DP_X2    = [True,  True,  True,  False]
_DP_FUSE  = [True,  True,  True,  False]

class PoolNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder 形成多尺度特征与全局上下文信息
        self.base = ResNet18Locate()

        # [c1, c2, c3, c4] 通道调整
        self.convert = ConvertLayer(_IN_CH, _OUT_CH)

        # Decoder 融合不同特征
        self.deep_pool = nn.ModuleList([
            DeepPoolLayer(_DP_IN[i], _DP_OUT[i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        # 输出头
        self.score = ScoreLayer(128)

        self._init_weights()

    def _init_weights(self):
        modules_to_init = [
            self.base.ppms_pre,
            self.base.ppms,
            self.base.ppm_cat,
            self.base.infos,
            self.convert,
            self.deep_pool,
        ]

        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        for m in self.score.modules():
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
