import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)

_DP_FAM = [True, True, True, False]

class SemanticPromptAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        hidden_ch = max(out_ch // 2, 32)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, prompt, target):
        prompt = self.proj(prompt)

        if prompt.shape[2:] != target.shape[2:]:
            prompt = F.interpolate(
                prompt,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        return target + alpha * prompt

class ForegroundGate(nn.Module):
    def __init__(self, guide_ch, hidden_ch=None, init_strength=0.4):
        super().__init__()

        if hidden_ch is None:
            hidden_ch = max(guide_ch // 4, 32)

        self.gate = nn.Sequential(
            nn.Conv2d(guide_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self.gate_strength = nn.Parameter(torch.tensor(float(init_strength)))

    def forward(self, x, guide, return_gate=False):
        gate = self.gate(guide)

        if gate.shape[2:] != x.shape[2:]:
            gate = F.interpolate(
                gate,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        strength = torch.clamp(self.gate_strength, 0.0, 1.0)
        factor = (1.0 - strength) + strength * gate
        out = x * factor

        if return_gate:
            return out, gate

        return out

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

    def __init__(self, k, k_out, need_x2, need_fuse, use_fam):
        super().__init__()
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        self.use_fam = use_fam
        self.pool_scales = [2, 4, 8]

        self.conv_in = nn.Sequential(
            nn.Conv2d(k, k_out, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        if need_fuse:
            self.cfm = CFM(k_out)

        if use_fam:
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
        else:
            self.refine = nn.Sequential(
                nn.Conv2d(k_out, k_out, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

    def _adaptive_pool(self, x, scale):
        h, w = x.shape[2:]
        return F.adaptive_avg_pool2d(
            x,
            (max(1, h // scale), max(1, w // scale)),
        )

    def _fam(self, x):
        size = x.shape[2:]
        out = x

        for scale, conv in zip(self.pool_scales, self.pool_convs):
            pooled = conv(self._adaptive_pool(x, scale))
            pooled = F.interpolate(
                pooled,
                size,
                mode="bilinear",
                align_corners=True,
            )
            out = out + pooled

        return self.conv_out(out)

    def forward(self, x, x_skip=None, x_info=None, prompt=None):
        if self.need_x2:
            x = F.interpolate(
                x,
                x_skip.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        x = self.conv_in(x)

        if self.need_fuse:
            x = self.cfm(x, x_skip) + x_info

        if prompt is not None:
            x = x + prompt

        if self.use_fam:
            return self._fam(x)

        return self.refine(x)

class PoolNetCFMEnhanced(nn.Module):

    def __init__(self, pretrained=True, backbone_name="resnet18", use_fg_gate=True, fg_gate_strength=0.5, use_prompt=False, prompt_alpha=0.1):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base = ResNetLocate(
            pretrained=pretrained,
            backbone_name=backbone_name,
        )
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCFM(
                cfg["dp_in"][i],
                cfg["dp_out"][i],
                _DP_X2[i],
                _DP_FUSE[i],
                _DP_FAM[i],
            )
            for i in range(4)
        ])

        self.use_prompt = use_prompt

        # deepest converted feature is feats_r[0]
        prompt_in_ch = cfg["dp_in"][0]

        if self.use_prompt:
            prompt_ch = cfg["dp_in"][0]
            self.prompt_adapters = nn.ModuleList([
                SemanticPromptAdapter(
                    in_ch=prompt_ch,
                    out_ch=cfg["dp_out"][i],
                )
                for i in range(4)
            ])

        self.use_fg_gate = use_fg_gate

        if use_fg_gate:
            self.fg_gate = ForegroundGate(
                guide_ch=cfg["dp_out"][0],
                init_strength=fg_gate_strength,
            )
        else:
            self.fg_gate = None

        self.score = ScoreLayer(cfg["dp_out"][-1])

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith("backbone") or name.startswith("base.backbone"):
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
        feats_r = feats[::-1]   # [c4, c3, c2, c1]

        # semantic_prompt = feats_r[0]
        # merge = None

        # for i, layer in enumerate(self.deep_pool):
        #     if i == 0:
        #         layer_input = feats_r[0]
        #     else:
        #         layer_input = merge

        #     # 先得到当前 stage 的 target 尺寸参考
        #     if i < len(feats_r) - 1:
        #         target_ref = feats_r[i + 1]
        #     else:
        #         target_ref = layer_input

        #     prompt = self.prompt_adapters[i](
        #         semantic_prompt,
        #         target_ref,
        #     )

        #     if i < len(feats_r) - 1:
        #         merge = layer(
        #             layer_input,
        #             feats_r[i + 1],
        #             infos[i],
        #             prompt=prompt,
        #         )
        #     else:
        #         merge = layer(
        #             layer_input,
        #             prompt=prompt,
        #         )

        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        fg_guide = merge

        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])

        merge = self.deep_pool[-1](merge)

        if self.use_fg_gate:
            merge = self.fg_gate(merge, fg_guide)

        out = self.score(merge, input_size)
        return out
