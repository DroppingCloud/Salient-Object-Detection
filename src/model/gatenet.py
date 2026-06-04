import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolnet import _get_backbone

class FoldConvASPP(nn.Module):
    """
    将特征图 unfold 成局部 patch，在 patch 空间做多尺度空洞卷积，
    再 fold 回原分辨率。比普通 ASPP 能捕获更精细的局部结构信息。

    Parameters
    ----------
    in_channel : int  — 输入通道数
    out_channel : int — 输出通道数（也是 unfold 前的中间通道）
    win_size : int    — unfold 窗口大小
    """

    def __init__(self, in_channel, out_channel, win_size=2):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(),
        )
        self.win_size = win_size
        self.unfold = nn.Unfold(
            kernel_size=win_size, dilation=1, padding=0, stride=win_size
        )
        fold_c = out_channel * win_size * win_size
        down_dim = fold_c // 2

        # 5 路并行: 1×1 + 3×3 dil=2 + 3×3 dil=4 + 3×3 dil=6 + global pool
        self.conv1 = nn.Sequential(
            nn.Conv2d(fold_c, down_dim, 1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fold_c, down_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fold_c, down_dim, 3, padding=4, dilation=4),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fold_c, down_dim, 3, padding=6, dilation=6),
            nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(fold_c, down_dim, 1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, fold_c, 1), nn.BatchNorm2d(fold_c), nn.PReLU()
        )
        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.down_conv(x)

        # unfold: (B, C*win*win, num_patches)
        x = self.unfold(x)
        x = x.view(x.size(0), x.size(1), H // self.win_size, W // self.win_size)

        # 多尺度空洞卷积
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        f4 = self.conv4(x)
        f5 = F.interpolate(
            self.conv5(F.adaptive_avg_pool2d(x, 1)),
            size=x.shape[2:], mode='bilinear', align_corners=True
        )
        x = self.fuse(torch.cat([f1, f2, f3, f4, f5], dim=1))

        # fold 回原始分辨率
        x = x.reshape(x.size(0), x.size(1), -1)
        x = F.fold(x, output_size=(H, W),
                   kernel_size=self.win_size, stride=self.win_size)
        x = self.up_conv(x)
        return x


# ─────────────────────────────────────────────────────────────
# Gate Module: 生成 2 通道 attention (FPN gate + Parallel gate)
# ─────────────────────────────────────────────────────────────

class GateModule(nn.Module):
    """
    门控模块：对拼接的 encoder 特征和上级 decoder 特征生成
    2 通道 sigmoid attention map (经 global avg pool 压缩为标量)。

    Parameters
    ----------
    in_ch : int — 输入拼接通道数 (encoder_ch + decoder_ch)
    """

    def __init__(self, in_ch):
        super().__init__()
        mid = max(in_ch // 4, 64)
        layers = [nn.Conv2d(in_ch, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.PReLU()]
        # 如果输入通道很大，多加一层压缩
        if in_ch > 512:
            mid2 = max(mid // 2, 64)
            layers += [nn.Conv2d(mid, mid2, 3, padding=1), nn.BatchNorm2d(mid2), nn.PReLU()]
            mid = mid2
        layers.append(nn.Conv2d(mid, 2, 3, padding=1))
        self.gate = nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns
        -------
        gate : Tensor (B, 2, 1, 1) — sigmoid attention weights
        """
        g = self.gate(x)
        g = F.adaptive_avg_pool2d(torch.sigmoid(g), 1)  # (B, 2, 1, 1)
        return g


# ─────────────────────────────────────────────────────────────
# GateNet 主网络
# ─────────────────────────────────────────────────────────────

class GateNet(nn.Module):

    def __init__(self, pretrained=True, backbone_name="resnet18"):
        super().__init__()
        self.backbone, channels = _get_backbone(backbone_name, pretrained)
        ch1, ch2, ch3, ch4 = channels
        # resnet18/34: [64, 128, 256, 512]
        # resnet50:    [256, 512, 1024, 2048]

        # ──────────────────────────────────────────────────────────
        # Transition Layer (DEM): 通道压缩
        # ──────────────────────────────────────────────────────────
        t4_ch = ch4 // 4   # 最深层: 512→128 / 2048→512
        t3_ch = ch3 // 4   # 256→64  / 1024→256
        t2_ch = ch2 // 2   # 128→64  / 512→256
        t1_ch = ch1        # 64→64   / 256→256

        # 最深层使用 FoldConvASPP 捕获多尺度上下文
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

        # ──────────────────────────────────────────────────────────
        # Gate Modules
        # 输入 = encoder 特征 + 上级 decoder 特征 (拼接通道)
        # ──────────────────────────────────────────────────────────
        # FPN 各级输出通道: 每级 FPN 输出需与下一级 T 通道匹配以便相加
        # fpn4 输出 → 与 T3 相加, fpn3 输出 → 与 T2 相加, fpn2 输出 → 与 T1 相加
        d4_ch = t3_ch   # fpn4 输出通道 = t3_ch
        d3_ch = t2_ch   # fpn3 输出通道 = t2_ch
        d2_ch = t1_ch   # fpn2 输出通道 = t1_ch

        self.gate4 = GateModule(ch4 + t4_ch)       # E4 + T4 (最深层)
        self.gate3 = GateModule(ch3 + d4_ch)       # E3 + D4_up
        self.gate2 = GateModule(ch2 + d3_ch)       # E2 + D3_up
        self.gate1 = GateModule(ch1 + d2_ch)       # E1 + D2_up

        # ──────────────────────────────────────────────────────────
        # FPN Decoder: 自顶向下逐级融合
        # fpn_N 输入通道 = t_N_ch (门控后的 T 与上级 D_up 相加)
        # fpn_N 输出通道 = d_N_ch (匹配下一级 T 通道)
        # ──────────────────────────────────────────────────────────
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
        self.fpn1 = nn.Conv2d(t1_ch, 1, 3, padding=1)  # 最浅 FPN → 单通道

        # ──────────────────────────────────────────────────────────
        # Parallel Branch: 聚合被抑制特征做残差修正
        # 输入 = D1(1ch) + gate[1]*T 各级
        # ──────────────────────────────────────────────────────────
        par_in = 1 + t4_ch + t3_ch + t2_ch + t1_ch
        par_mid = min(par_in // 2, 256)
        self.parallel = nn.Sequential(
            nn.Conv2d(par_in, par_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(par_mid), nn.PReLU(),
            nn.Conv2d(par_mid, par_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(par_mid), nn.PReLU(),
            nn.Conv2d(par_mid, 1, 3, padding=1),
        )

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

        # ── Encoder: 4-stage backbone ────────────────────────────
        E1, E2, E3, E4 = self.backbone(x)
        # E1: (B, ch1, H/4, W/4)
        # E2: (B, ch2, H/8, W/8)
        # E3: (B, ch3, H/16, W/16)
        # E4: (B, ch4, H/32, W/32)

        # ── Transition Layer ─────────────────────────────────────
        T4 = self.dem4(E4)    # 通道压缩 + 多尺度增强
        T3 = self.dem3(E3)
        T2 = self.dem2(E2)
        T1 = self.dem1(E1)

        # ── Gated FPN (自顶向下) ─────────────────────────────────
        # Level 4 (最深层)
        G4 = self.gate4(torch.cat([E4, T4], dim=1))          # (B, 2, 1, 1)
        D4 = self.fpn4(G4[:, 0:1] * T4)                     # gate[0] 控制前景

        # Level 3
        D4_up = F.interpolate(D4, E3.shape[2:], mode='bilinear', align_corners=True)
        G3 = self.gate3(torch.cat([E3, D4_up], dim=1))
        D3 = self.fpn3(D4_up + G3[:, 0:1] * T3)

        # Level 2
        D3_up = F.interpolate(D3, E2.shape[2:], mode='bilinear', align_corners=True)
        G2 = self.gate2(torch.cat([E2, D3_up], dim=1))
        D2 = self.fpn2(D3_up + G2[:, 0:1] * T2)

        # Level 1 (最浅层)
        D2_up = F.interpolate(D2, E1.shape[2:], mode='bilinear', align_corners=True)
        G1 = self.gate1(torch.cat([E1, D2_up], dim=1))
        D1 = self.fpn1(D2_up + G1[:, 0:1] * T1)             # (B, 1, H/4, W/4)

        # ── FPN 输出 ─────────────────────────────────────────────
        output_fpn = F.interpolate(D1, (H, W), mode='bilinear', align_corners=True)

        # ── Parallel Branch (残差修正) ───────────────────────────
        # 各层被抑制特征 gate[1]*T 上采样到 E1 尺寸
        target_size = E1.shape[2:]
        par_feats = [
            D1,
            F.interpolate(G4[:, 1:2] * T4, target_size, mode='bilinear', align_corners=True),
            F.interpolate(G3[:, 1:2] * T3, target_size, mode='bilinear', align_corners=True),
            F.interpolate(G2[:, 1:2] * T2, target_size, mode='bilinear', align_corners=True),
            G1[:, 1:2] * T1,   # 已经是 E1 尺寸
        ]
        output_res = self.parallel(torch.cat(par_feats, dim=1))
        output_res = F.interpolate(output_res, (H, W), mode='bilinear', align_corners=True)

        # ── 融合 ─────────────────────────────────────────────────
        pre_sal = output_fpn + output_res

        if self.training:
            return output_fpn, pre_sal
        return pre_sal
