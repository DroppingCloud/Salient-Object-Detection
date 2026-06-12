import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import (
    ResNetLocate, ConvertLayer, DeepPoolLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)


class EdgeAwareModule(nn.Module):
    """边缘感知模块：提取边缘特征增强显著性预测"""

    def __init__(self, in_ch):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 1, 1),
        )

    def forward(self, x):
        return self.edge_conv(x)


class MultiScaleHead(nn.Module):
    """多尺度预测头：特征精炼 + 显著性预测"""

    def __init__(self, in_ch):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, in_ch // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.pred = nn.Conv2d(in_ch // 4, 1, 1)

    def forward(self, x):
        x = self.refine(x)
        return self.pred(x)


class PoolNetDS(nn.Module):
    """PoolNet + 多尺度深度监督 + 边缘增强。

    深度监督策略：
        - 在 m1 (dp_outputs[2])、m2 (dp_outputs[1])、m3 (dp_outputs[0]) 三层输出显著性图
        - 每层都有独立的预测头和边缘增强模块
        - 损失权重递减：浅层（m1）权重大，深层（m3）权重小
        - 训练返回 (out_main, aux_sal_m1, aux_sal_m2, aux_sal_m3, edge_m1)，推理返回 out_main
    """

    use_distill = False  # 不使用特征蒸馏，使用显著性监督

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        self.base     = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        self.deep_pool = nn.ModuleList([
            DeepPoolLayer(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        self.score = ScoreLayer(cfg["dp_out"][-1])

        # 多尺度辅助预测头（m1, m2, m3）
        self.aux_head_m1 = MultiScaleHead(cfg["dp_out"][2])  # 浅层，大分辨率
        self.aux_head_m2 = MultiScaleHead(cfg["dp_out"][1])  # 中间层
        self.aux_head_m3 = MultiScaleHead(cfg["dp_out"][0])  # 深层，小分辨率

        # 边缘增强模块（仅在 m1 上使用，因为浅层细节丰富）
        self.edge_module = EdgeAwareModule(cfg["dp_out"][2])

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

        feats, infos = self.base(x)
        feats = self.convert(feats)
        feats_r = feats[::-1]   # [c4, c3, c2, c1]

        dp_outputs = []

        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        dp_outputs.append(merge)  # m3

        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
            dp_outputs.append(merge)  # m2, m1

        merge = self.deep_pool[-1](merge)
        dp_outputs.append(merge)  # final

        out_main = self.score(merge, input_size)

        if self.training:
            # 多尺度辅助输出
            aux_sal_m3 = self.aux_head_m3(dp_outputs[0])  # 深层
            aux_sal_m3 = F.interpolate(aux_sal_m3, size=input_size[2:],
                                       mode='bilinear', align_corners=True)

            aux_sal_m2 = self.aux_head_m2(dp_outputs[1])  # 中间层
            aux_sal_m2 = F.interpolate(aux_sal_m2, size=input_size[2:],
                                       mode='bilinear', align_corners=True)

            aux_sal_m1 = self.aux_head_m1(dp_outputs[2])  # 浅层
            aux_sal_m1 = F.interpolate(aux_sal_m1, size=input_size[2:],
                                       mode='bilinear', align_corners=True)

            # 边缘预测（仅 m1）
            edge_m1 = self.edge_module(dp_outputs[2])
            edge_m1 = F.interpolate(edge_m1, size=input_size[2:],
                                    mode='bilinear', align_corners=True)

            return {
                "main":    out_main,
                "aux_sal": [aux_sal_m1, aux_sal_m2, aux_sal_m3],
                "edge":    [edge_m1],
            }

        return out_main
