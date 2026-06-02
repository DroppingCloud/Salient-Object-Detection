import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet18 import ResNet18, ResNet18Pre

from .poolnet import (
    ResNet18Locate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _IN_CH, _OUT_CH, _DP_IN, _DP_OUT, _DP_X2, _DP_FUSE,
)


class SpatialReductionAttention(nn.Module):

    def __init__(self, dim, num_heads=4, sr_ratio=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=False)

        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)

        # 空间降采样：步幅卷积
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )

        # 残差缩放，初始为 0 保证训练初期不破坏已有特征
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, H, W = x.shape

        # Q: 保持原始分辨率
        q = self.q(x)  # (B, C, H, W)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        # q: (B, heads, N, head_dim)

        # K, V: 空间降采样后生成
        if self.sr_ratio > 1:
            x_sr = self.sr(x)  # (B, C, H/r, W/r)
        else:
            x_sr = x

        kv = self.kv(x_sr)  # (B, 2C, H_sr, W_sr)
        _, _, H_sr, W_sr = kv.shape
        N_sr = H_sr * W_sr

        kv = kv.reshape(B, 2, self.num_heads, self.head_dim, N_sr).permute(0, 2, 1, 4, 3)
        # kv: (B, heads, 2, N_sr, head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        # k, v: (B, heads, N_sr, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N_sr)
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj(out)

        # LayerNorm (channel-last)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)

        return x + self.alpha * out


class CFM_GA(nn.Module):

    def __init__(self, ch, num_heads=4, sr_ratio=2):
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

        # 全局注意力：对融合特征做空间全局建模
        self.global_attn = SpatialReductionAttention(ch, num_heads=num_heads, sr_ratio=sr_ratio)

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

        # 全局注意力增强融合特征
        fused = self.global_attn(fused)

        base = out + x_skip + x_info
        enhanced = self.info_proj(fused + x_info)

        return base + self.gamma * enhanced


class DeepPoolLayerCFM_GA(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse, num_heads=4, sr_ratio=2):
        super().__init__()
        self.need_x2 = need_x2
        self.need_fuse = need_fuse

        self.conv_sum = nn.Conv2d(k, k_out, 3, padding=1, bias=False)

        if need_fuse:
            self.cfm = CFM_GA(k_out, num_heads=num_heads, sr_ratio=sr_ratio)

    def forward(self, x, x_skip=None, x_info=None):
        if self.need_x2:
            x = F.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=True)

        out = self.conv_sum(x)

        if self.need_fuse:
            out = self.cfm(out, x_skip, x_info)

        return out


class PoolNetCFMGA(nn.Module):

    def __init__(self, pretrained=True, num_heads=4, sr_ratio=2):
        super().__init__()
        self.base = ResNet18Locate(pretrained=pretrained)
        self.backbone = self.base.backbone

        self.convert = ConvertLayer(_IN_CH, _OUT_CH)

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCFM_GA(
                _DP_IN[i], _DP_OUT[i], _DP_X2[i], _DP_FUSE[i],
                num_heads=num_heads, sr_ratio=sr_ratio,
            )
            for i in range(4)
        ])

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x):
        input_size = x.shape

        feats, infos = self.base(x)
        feats = self.convert(feats)
        feats_r = feats[::-1]  # [c4, c3, c2, c1]

        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        out = self.score(merge, input_size)
        return out
