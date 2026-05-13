import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ ResNet-18 残差块 """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.relu(out)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        if skip is not None:
            # x 上采样到 skip 尺寸
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            # 深层特征图与编码器对应尺度浅层特征图在通道维度融合
            x = torch.cat([x, skip], dim=1)
        else:
            # scale_factor=2: 特征图宽/高放大 2 倍
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )

        return self.conv(x)


class ResNet18_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 Encoder
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.dec0 = DecoderBlock(64, 0, 32)

        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        e0 = self.relu(self.bn1(self.conv1(x)))   # /2,  64
        e1 = self.maxpool(e0)                     # /4
        e1 = self.layer1(e1)                      # /4,  64
        e2 = self.layer2(e1)                      # /8,  128
        e3 = self.layer3(e2)                      # /16, 256
        e4 = self.layer4(e3)                      # /32, 512

        # Decoder
        d = self.dec4(e4, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.dec0(d)

        out = self.head(d)

        # 确保尺寸对齐
        if out.shape[-2:] != input_size:
            out = F.interpolate(
                out,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

        return out