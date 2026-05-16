import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

class BasicBlock(nn.Module):
    """ ResNet 基本残差块 """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 当维度不匹配时用 1x1 卷积对齐
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)
            
        return self.relu(out + identity)

class ResNet18(nn.Module):
    def __init__(self, return_features=True):

        super().__init__()
        self.return_features = return_features
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 特征提取
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)   # 1/4,  64ch
        c2 = self.layer2(c1)  # 1/8,  128ch
        c3 = self.layer3(c2)  # 1/16, 256ch
        c4 = self.layer4(c3)  # 1/32, 512ch

        # 输出中间特征
        if self.return_features:
            return [c1, c2, c3, c4]

class ResNet18Pre(nn.Module):
    """ ImageNet 预训练 ResNet18 """
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool
        )
        self.layer1 = m.layer1   # 64, 1/4
        self.layer2 = m.layer2   # 128, 1/8
        self.layer3 = m.layer3   # 256, 1/16
        self.layer4 = m.layer4   # 512, 1/32

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4
