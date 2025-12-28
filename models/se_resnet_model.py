# -*- coding: utf-8 -*-
# models/se_resnet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseline_model import BasicBlock # Import the basic block from baseline (ResNet-18/34)

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Flatten pooled features
        y = self.fc(y).view(b, c, 1, 1) # Apply FC layers and reshape back
        return x * y.expand_as(x) # Apply the attention weights

# --- Modified Basic Block with SE ---
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction_ratio=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction_ratio)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Apply SE attention
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- ResNet with SE Blocks ---
class SEResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, reduction_ratio=16):
        super(SEResNet, self).__init__()
        self.in_planes = 64
        self.reduction_ratio = reduction_ratio

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.reduction_ratio))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# --- Renamed Helper Function for SE-ResNet-34 ---
def se_resnet34(num_classes=50, reduction_ratio=16):
    return SEResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, reduction_ratio=reduction_ratio) # SE-ResNet-34 config

# --- Kept Helper Function for SE-ResNet-18 (if needed for comparison) ---
def se_resnet18(num_classes=50, reduction_ratio=16):
    return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, reduction_ratio=reduction_ratio) # SE-ResNet-18 config
