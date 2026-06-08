import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import _get_backbone
from .gatenet import FoldConvASPP, GateModule


class GateNetDS(nn.Module):
    """
    GateNetDS: GateNet + 多级深监督。

    改进动机
    --------
    GateNet 原始训练时只有两路监督：
      output_fpn（FPN 单路）和 pre_sal（FPN+Parallel 融合）。
    中间解码层 D4、D3、D2 缺乏直接监督，导致：
      1. 中间层梯度稀疏，深层特征语义性弱；
      2. 全靠最终输出层向后传递梯度，收敛慢。

    深监督（Deep Supervision）在 D4（H/8）、D3（H/16 ↑ H/8）
    和 D2（H/4）三个中间解码阶段各加一个 1×1 预测头并施加监督，
    使每一层特征都能直接感知目标位置，梯度更充分，边界更清晰。

    输出格式（训练）
    ----------------
    (pre_sal, output_fpn, pred_d3, pred_d2)
    — Trainer 权重: 2.0 | 1.0 | 1.0 | 1.0
    推理时只返回 pre_sal。
    """

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        self.backbone, channels = _get_backbone(backbone_name, pretrained)
        ch1, ch2, ch3, ch4 = channels

        t4_ch = ch4 // 4
        t3_ch = ch3 // 4
        t2_ch = ch2 // 2
        t1_ch = ch1

        # ── DEM ───────────────────────────────────────────────────────
        self.dem4 = nn.Sequential(
            FoldConvASPP(ch4, t4_ch, win_size=2),
            nn.BatchNorm2d(t4_ch), nn.PReLU(),
        )
        self.dem3 = nn.Sequential(
            nn.Conv2d(ch3, t3_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(t3_ch), nn.PReLU(),
        )
        self.dem2 = nn.Sequential(
            nn.Conv2d(ch2, t2_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(t2_ch), nn.PReLU(),
        )
        self.dem1 = nn.Sequential(
            nn.Conv2d(ch1, t1_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(t1_ch), nn.PReLU(),
        )

        d4_ch = t3_ch
        d3_ch = t2_ch
        d2_ch = t1_ch

        # ── Gate Modules ──────────────────────────────────────────────
        self.gate4 = GateModule(ch4 + t4_ch)
        self.gate3 = GateModule(ch3 + d4_ch)
        self.gate2 = GateModule(ch2 + d3_ch)
        self.gate1 = GateModule(ch1 + d2_ch)

        # ── FPN Decoder ───────────────────────────────────────────────
        self.fpn4 = nn.Sequential(
            nn.Conv2d(t4_ch, d4_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(d4_ch), nn.PReLU(),
        )
        self.fpn3 = nn.Sequential(
            nn.Conv2d(t3_ch, d3_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(d3_ch), nn.PReLU(),
        )
        self.fpn2 = nn.Sequential(
            nn.Conv2d(t2_ch, d2_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(d2_ch), nn.PReLU(),
        )
        self.fpn1 = nn.Conv2d(t1_ch, 1, 3, padding=1)

        # ── Parallel Branch ───────────────────────────────────────────
        par_in = 1 + t4_ch + t3_ch + t2_ch + t1_ch
        par_mid = min(par_in // 2, 256)
        self.parallel = nn.Sequential(
            nn.Conv2d(par_in, par_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(par_mid), nn.PReLU(),
            nn.Conv2d(par_mid, par_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(par_mid), nn.PReLU(),
            nn.Conv2d(par_mid, 1, 3, padding=1),
        )

        # ── 深监督辅助预测头 ──────────────────────────────────────────
        self.head_d3 = nn.Conv2d(d3_ch, 1, 1)   # D3 (H/4)
        self.head_d2 = nn.Conv2d(d2_ch, 1, 1)   # D2 (H/4 after upsample)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        H, W = x.shape[2:]

        E1, E2, E3, E4 = self.backbone(x)

        T4 = self.dem4(E4)
        T3 = self.dem3(E3)
        T2 = self.dem2(E2)
        T1 = self.dem1(E1)

        # ── Gated FPN ─────────────────────────────────────────────────
        G4 = self.gate4(torch.cat([E4, T4], dim=1))
        D4 = self.fpn4(G4[:, 0:1] * T4)

        D4_up = F.interpolate(D4, E3.shape[2:], mode='bilinear', align_corners=True)
        G3 = self.gate3(torch.cat([E3, D4_up], dim=1))
        D3 = self.fpn3(D4_up + G3[:, 0:1] * T3)

        D3_up = F.interpolate(D3, E2.shape[2:], mode='bilinear', align_corners=True)
        G2 = self.gate2(torch.cat([E2, D3_up], dim=1))
        D2 = self.fpn2(D3_up + G2[:, 0:1] * T2)

        D2_up = F.interpolate(D2, E1.shape[2:], mode='bilinear', align_corners=True)
        G1 = self.gate1(torch.cat([E1, D2_up], dim=1))
        D1 = self.fpn1(D2_up + G1[:, 0:1] * T1)

        output_fpn = F.interpolate(D1, (H, W), mode='bilinear', align_corners=True)

        # ── Parallel Branch ───────────────────────────────────────────
        target_size = E1.shape[2:]
        par_feats = [
            D1,
            F.interpolate(G4[:, 1:2] * T4, target_size, mode='bilinear', align_corners=True),
            F.interpolate(G3[:, 1:2] * T3, target_size, mode='bilinear', align_corners=True),
            F.interpolate(G2[:, 1:2] * T2, target_size, mode='bilinear', align_corners=True),
            G1[:, 1:2] * T1,
        ]
        output_res = F.interpolate(
            self.parallel(torch.cat(par_feats, dim=1)), (H, W), mode='bilinear', align_corners=True
        )
        pre_sal = output_fpn + output_res

        if self.training:
            # 深监督：D3 和 D2 阶段各加一个辅助预测
            pred_d3 = F.interpolate(self.head_d3(D3), (H, W), mode='bilinear', align_corners=True)
            pred_d2 = F.interpolate(self.head_d2(D2), (H, W), mode='bilinear', align_corners=True)
            return pre_sal, output_fpn, pred_d3, pred_d2

        return pre_sal
