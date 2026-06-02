import torch.nn as nn

from .poolnet import ConvertLayer, ScoreLayer, ResNet18Locate, _IN_CH, _OUT_CH, _DP_IN, _DP_OUT, _DP_X2, _DP_FUSE
from .poolnet_cfm import DeepPoolLayerCFM
from .poolnet_rrm import ResidualRefinementModule

class PoolNetCFMRRM(nn.Module):

    loss_weights = (1.0, 0.4)
    simple_aux_loss = False

    def __init__(self, pretrained=True):
        super().__init__()
        # ── Encoder + PPM ──
        self.base = ResNet18Locate(pretrained=pretrained)
        self.backbone = self.base.backbone

        # ── 通道对齐 ──
        self.convert = ConvertLayer(_IN_CH, _OUT_CH)

        # ── CFM 解码器 ──
        self.deep_pool = nn.ModuleList([
            DeepPoolLayerCFM(_DP_IN[i], _DP_OUT[i], _DP_X2[i], _DP_FUSE[i])
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

    def forward(self, x):
        input_size = x.shape

        # ── Backbone + PPM ──
        feats, infos = self.base(x)

        # 保留浅层特征用于 RRM
        shallow_feat = feats[0]  # c1: (B, 64, H/4, W/4)

        # ── 通道对齐 ──
        feats = self.convert(feats)
        feats_r = feats[::-1]  # [c4, c3, c2, c1]

        # ── CFM 解码 ──
        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        # ── 粗预测 ──
        coarse = self.score(merge, input_size)  # (B, 1, H, W)

        # ── 残差细化 ──
        refined = self.rrm(coarse, shallow_feat)

        return refined, coarse
