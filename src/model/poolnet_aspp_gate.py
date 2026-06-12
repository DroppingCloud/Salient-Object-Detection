import torch.nn as nn

from .poolnet import (
    ResNetLocate, ConvertLayer, ScoreLayer,
    _BACKBONE_TABLE, _get_decoder_cfg, _DP_X2, _DP_FUSE,
)
from .poolnet_aspp import DeepPoolLayerASPP
from .poolnet_gate import DeepPoolLayerGated

# 层模块选择配置：层 0,1 用 ASPP，层 2,3 用 Gate
_DP_ASPP = [True, True, False, False]


class PoolNetASPPGate(nn.Module):
    """PoolNet + ASPP（深层 0,1）+ Gate 注意力（浅层 2,3）

    架构：
    - 层 0,1: DeepPoolLayerASPP (ASPP 多尺度全局上下文)
    - 层 2,3: DeepPoolLayerGated (Gate 双分支注意力融合)
    """

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        cfg = _get_decoder_cfg(backbone_name)

        # Encoder
        self.base = ResNetLocate(pretrained=pretrained, backbone_name=backbone_name)
        self.backbone = self.base.backbone

        # Channel converter
        channels = _BACKBONE_TABLE[backbone_name][2]
        self.convert = ConvertLayer(channels, cfg["out_ch"])

        # Hybrid decoder: 根据 _DP_ASPP 选择模块类型
        self.deep_pool = nn.ModuleList([
            DeepPoolLayerASPP(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            if _DP_ASPP[i] else
            DeepPoolLayerGated(cfg["dp_in"][i], cfg["dp_out"][i], _DP_X2[i], _DP_FUSE[i])
            for i in range(4)
        ])

        # Output head
        self.score = ScoreLayer(cfg["dp_out"][-1])

        self._init_weights()

    def _init_weights(self):
        """初始化权重，跳过预训练 backbone 和已初始化的模块"""
        for name, m in self.named_modules():
            # 跳过预训练 backbone
            if name.startswith('backbone') or name.startswith('base.backbone'):
                continue

            # 跳过 ASPP 内部（已用 Kaiming 初始化）
            if any(name.startswith(f'deep_pool.{i}.aspp') for i in range(4)):
                continue

            # 跳过 Gate 内部（已有专门初始化）
            if 'channel_gate' in name or 'spatial_gate' in name:
                continue

            # 其他卷积层用标准初始化
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入图像 tensor (B, 3, H, W)

        Returns:
            输出显著图 tensor (B, 1, H, W)
        """
        input_size = x.shape

        # Encoder: 提取多尺度特征和边缘信息
        feats, infos = self.base(x)
        feats = self.convert(feats)
        feats_r = feats[::-1]  # [c4, c3, c2, c1] 从深到浅

        # Decoder: 逐层上采样和融合
        merge = self.deep_pool[0](feats_r[0], feats_r[1], infos[0])
        for k in range(1, len(feats_r) - 1):
            merge = self.deep_pool[k](merge, feats_r[k + 1], infos[k])
        merge = self.deep_pool[-1](merge)

        # Output: 上采样到输入分辨率
        return self.score(merge, input_size)
