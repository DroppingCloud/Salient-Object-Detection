import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNet18Locate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _IN_CH, _OUT_CH, _DP_IN, _DP_OUT, _DP_X2, _DP_FUSE,
)
from .poolnet_cfm import CFM
from .poolnet_ca import ChannelAttention
from .poolnet_rrm import ResidualRefinementModule


class DeepPoolLayerCFMCA(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse, reduction=16):
        super().__init__()
        self.need_x2 = need_x2
        self.need_fuse = need_fuse

        self.conv_sum = nn.Conv2d(k, k_out, 3, padding=1, bias=False)

        if need_fuse:
            self.cfm = CFM(k_out)
            self.ca = ChannelAttention(k_out, reduction=reduction)

    def forward(self, x, x_skip=None, x_info=None):
        if self.need_x2:
            x = F.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=True)

        out = self.conv_sum(x)

        if self.need_fuse:
            out = self.cfm(out, x_skip, x_info)
            out = self.ca(out)

        return out


class PoolNetCFMCARRM(nn.Module):

    loss_weights = (1.0, 0.4)
    simple_aux_loss = False

    def __init__(self, pretrained=True):
        super().__init__()
        # ── Encoder + PPM ──
        self.base = ResNet18Locate(pretrained=pretrained)
        self.backbone = self.base.backbone

        # ── 通道对齐 ──
        self.convert = ConvertLayer(_IN_CH, _OUT_CH)

        # ── CFM + CA 解码器 ──
        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCFMCA(_DP_IN[i], _DP_OUT[i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        # ── 粗预测头 ──
        self.score = ScoreLayer(128)

        # ── Residual Refinement Module ──
        self.rrm = ResidualRefinementModule(shallow_ch=64, mid_ch=64)

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        input_size = x.shape

        # ── Backbone + PPM ──
        feats, infos = self.base(x)

        # 保留浅层特征用于 RRM
        shallow_feat = feats[0]  # c1: (B, 64, H/4, W/4)

        # ── 通道对齐 ──
        feats = self.convert(feats)
        feats_r = feats[::-1]  # [c4, c3, c2, c1]

        # ── CFM + CA 解码 ──
        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        # ── 粗预测 ──
        coarse = self.score(merge, input_size)  # (B, 1, H, W)

        # ── 残差细化 ──
        refined = self.rrm(coarse, shallow_feat)

        return refined, coarse
