import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out

class RefineUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        # 初始投影
        self.conv0 = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # 编码器
        self.enc1 = self._conv_bn_relu(base_ch, base_ch)
        self.enc2 = self._conv_bn_relu(base_ch, base_ch)
        self.enc3 = self._conv_bn_relu(base_ch, base_ch)
        self.enc4 = self._conv_bn_relu(base_ch, base_ch)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Bottleneck
        self.bottleneck = self._conv_bn_relu(base_ch, base_ch)

        # 解码器
        self.dec4 = self._conv_bn_relu(base_ch * 2, base_ch)
        self.dec3 = self._conv_bn_relu(base_ch * 2, base_ch)
        self.dec2 = self._conv_bn_relu(base_ch * 2, base_ch)
        self.dec1 = self._conv_bn_relu(base_ch * 2, base_ch)

        # 输出残差
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    @staticmethod
    def _conv_bn_relu(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        hx = self.conv0(x)

        # 编码: 保存 skip 特征
        e1 = self.enc1(hx);         hx = self.pool(e1)
        e2 = self.enc2(hx);         hx = self.pool(e2)
        e3 = self.enc3(hx);         hx = self.pool(e3)
        e4 = self.enc4(hx);         hx = self.pool(e4)

        hx = self.bottleneck(hx)

        # 解码: 逐层上采样并拼接 skip
        hx = self.dec4(torch.cat([self.up(hx), e4], dim=1))
        hx = self.dec3(torch.cat([self.up(hx), e3], dim=1))
        hx = self.dec2(torch.cat([self.up(hx), e2], dim=1))
        hx = self.dec1(torch.cat([self.up(hx), e1], dim=1))

        residual = self.out_conv(hx)
        return x + residual   

class BASNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.loss_mode = "basnet_bsi"

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Stem
        self.stem    = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder1 = resnet.layer1   # 256 -> 256, 64ch
        self.encoder2 = resnet.layer2   # 256 -> 128, 128ch
        self.encoder3 = resnet.layer3   # 128 -> 64,  256ch
        self.encoder4 = resnet.layer4   # 64  -> 32,  512ch

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 32 -> 16

        # Stage 5
        self.stage5 = nn.Sequential(
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )
        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 16 -> 8

        # Stage 6
        self.stage6 = nn.Sequential(
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )

        # ------------------------------------------------------------------
        # Bridge
        # ------------------------------------------------------------------
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # stage 6d: 512+512 -> 512
        self.dec6 = self._dec_block(1024, 512, 512, dilated=True)
        # stage 5d: 512+512 -> 512
        self.dec5 = self._dec_block(1024, 512, 512, dilated=False)
        # stage 4d: 512+512 -> 256
        self.dec4 = self._dec_block(1024, 512, 256, dilated=False)
        # stage 3d: 256+256 -> 128
        self.dec3 = self._dec_block(512,  256, 128, dilated=False)
        # stage 2d: 128+128 -> 64
        self.dec2 = self._dec_block(256,  128,  64, dilated=False)
        # stage 1d: 64+64 -> 64
        self.dec1 = self._dec_block(128,   64,  64, dilated=False)

        # ------------------------------------------------------------------
        # Upsample
        # ------------------------------------------------------------------
        self.up2  = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=False)
        self.up_b = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.up6  = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.up5  = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.up4  = nn.Upsample(scale_factor=8,  mode='bilinear', align_corners=False)
        self.up3  = nn.Upsample(scale_factor=4,  mode='bilinear', align_corners=False)
        self.up_d2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # ------------------------------------------------------------------
        # 各尺度侧输出
        # ------------------------------------------------------------------
        self.side_b  = nn.Conv2d(512, n_classes, 3, padding=1)
        self.side6   = nn.Conv2d(512, n_classes, 3, padding=1)
        self.side5   = nn.Conv2d(512, n_classes, 3, padding=1)
        self.side4   = nn.Conv2d(256, n_classes, 3, padding=1)
        self.side3   = nn.Conv2d(128, n_classes, 3, padding=1)
        self.side2   = nn.Conv2d(64,  n_classes, 3, padding=1)
        self.side1   = nn.Conv2d(64,  n_classes, 3, padding=1)

        # ------------------------------------------------------------------
        # Refine
        # ------------------------------------------------------------------
        self.refine = RefineUNet(in_ch=1, base_ch=64)

    @property
    def backbone(self):
        return nn.ModuleList([
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
        ])

    @staticmethod
    def _dec_block(in_ch, mid_ch, out_ch, dilated=False):
        d, p = (2, 2) if dilated else (1, 1)
        return nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, dilation=d, padding=p),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, dilation=d, padding=p),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_size = x.shape[-2:]

        # ---- 编码 ----
        h0 = self.stem(x)                   # 256, 64ch

        h1 = self.encoder1(h0)              # 256, 64ch
        h2 = self.encoder2(h1)              # 128, 128ch
        h3 = self.encoder3(h2)              # 64,  256ch
        h4 = self.encoder4(h3)              # 32,  512ch

        h5 = self.stage5(self.pool4(h4))    # 16,  512ch
        h6 = self.stage6(self.pool5(h5))    # 8,   512ch

        # ---- Bridge ----
        hb = self.bridge(h6)                # 8,   512ch

        # ---- 解码 ----
        hd6 = self.dec6(torch.cat([hb,              h6], dim=1))           # 8
        hd5 = self.dec5(torch.cat([self.up2(hd6),   h5], dim=1))           # 16
        hd4 = self.dec4(torch.cat([self.up2(hd5),   h4], dim=1))           # 32
        hd3 = self.dec3(torch.cat([self.up2(hd4),   h3], dim=1))           # 64
        hd2 = self.dec2(torch.cat([self.up2(hd3),   h2], dim=1))           # 128
        hd1 = self.dec1(torch.cat([self.up2(hd2),   h1], dim=1))           # 256

        # ---- 侧输出 ----
        db = F.interpolate(self.side_b(hb), size=out_size, mode="bilinear", align_corners=False)
        d6 = F.interpolate(self.side6(hd6), size=out_size, mode="bilinear", align_corners=False)
        d5 = F.interpolate(self.side5(hd5), size=out_size, mode="bilinear", align_corners=False)
        d4 = F.interpolate(self.side4(hd4), size=out_size, mode="bilinear", align_corners=False)
        d3 = F.interpolate(self.side3(hd3), size=out_size, mode="bilinear", align_corners=False)
        d2 = F.interpolate(self.side2(hd2), size=out_size, mode="bilinear", align_corners=False)
        d1 = self.side1(hd1)

        # ---- 精修 ----
        d_out = self.refine(d1)

        return (
            d_out,   # 最终输出 logits
            d1,      # 各尺度辅助输出 logits
            d2,
            d3,
            d4,
            d5,
            d6,
            db,
        )
