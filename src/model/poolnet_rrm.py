import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNet18Locate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _IN_CH, _OUT_CH, _DP_IN, _DP_OUT, _DP_X2, _DP_FUSE,
)

class ResidualRefinementModule(nn.Module):
    def __init__(self, shallow_ch=64, mid_ch=64):
        super().__init__()

        # shallow feature 投影到 mid_ch - 1
        # 再和 1 通道 coarse_prob 拼接，最终输入通道数为 mid_ch
        self.shallow_proj = nn.Sequential(
            nn.Conv2d(shallow_ch, mid_ch - 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch - 1),
            nn.ReLU(inplace=True),
        )

        in_ch = mid_ch  # 1 + (mid_ch - 1) = mid_ch

        # ── Encoder ──
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──
        self.dec3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(mid_ch, 1, 1)

        # 可学习残差缩放，防止 RRM 初期破坏 coarse prediction
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, coarse_pred, shallow_feat):
        input_size = coarse_pred.shape

        coarse_prob = torch.sigmoid(coarse_pred)
        coarse_prob = F.interpolate(
            coarse_prob,
            shallow_feat.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        shallow_feat = self.shallow_proj(shallow_feat)

        x = torch.cat([coarse_prob, shallow_feat], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d3 = self.dec3(e3)
        d3_up = F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=True)

        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))
        d2_up = F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=True)

        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))

        residual = self.out_conv(d1)
        residual = F.interpolate(
            residual,
            input_size[2:],
            mode='bilinear',
            align_corners=True
        )

        refined = coarse_pred + self.res_scale * residual

        return refined


class PoolNetRRM(nn.Module):

    loss_weights = (1.0, 0.4)
    simple_aux_loss = False  # 粗预测也用完整 BCE+SSIM+IoU

    def __init__(self, pretrained=True):
        super().__init__()
        # ── 主网络（与 PoolNet 相同）──
        self.base = ResNet18Locate(pretrained=pretrained)
        self.backbone = self.base.backbone

        self.convert = ConvertLayer(_IN_CH, _OUT_CH)

        self.deep_pool = nn.ModuleList([
            DeepPoolLayer(_DP_IN[i], _DP_OUT[i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

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

    def forward(self, x):
        input_size = x.shape

        # ── Backbone + PPM ──
        feats, infos = self.base(x)

        # 保留浅层特征用于 RRM
        shallow_feat = feats[0]  # c1: (B, 64, H/4, W/4)

        # ── 通道对齐 ──
        feats = self.convert(feats)

        # ── 倒序解码 ──
        feats_r = feats[::-1]

        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        # ── 粗预测 ──
        coarse = self.score(merge, input_size)  # (B, 1, H, W)

        # ── 残差细化 ──
        refined = self.rrm(coarse, shallow_feat)

        return refined, coarse
