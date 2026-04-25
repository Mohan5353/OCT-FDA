import torch
import torch.nn as nn
import torch.nn.functional as F

class ADBlock(nn.Module):
    """Anamorphic Depth Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.relu(out)

class AnamNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        # Encoder
        self.c1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.ad1 = ADBlock(32, 32)
        self.p1 = nn.MaxPool2d(2)

        self.ad2 = ADBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)

        self.ad3 = ADBlock(64, 128)
        self.p3 = nn.MaxPool2d(2)

        self.ad4 = ADBlock(128, 256)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.b = ADBlock(256, 512)

        # Decoder
        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dad3 = ADBlock(512, 256) # 256+256 skip

        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dad2 = ADBlock(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dad1 = ADBlock(128, 64)

        self.u0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dad0 = ADBlock(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def get_encoder_features(self, x):
        f1 = self.ad1(self.c1(x))
        f2 = self.ad2(self.p1(f1))
        f3 = self.ad3(self.p2(f2))
        f4 = self.ad4(self.p3(f3))
        bottleneck = self.b(self.p4(f4))
        return [f1, f2, f3, f4, bottleneck]

    def forward_from_features(self, feats):
        f1, f2, f3, f4, b = feats
        x = self.u3(b)
        x = self.dad3(torch.cat([x, f4], dim=1))
        x = self.u2(x)
        x = self.dad2(torch.cat([x, f3], dim=1))
        x = self.u1(x)
        x = self.dad1(torch.cat([x, f2], dim=1))
        x = self.u0(x)
        x = self.dad0(torch.cat([x, f1], dim=1))
        return self.final(x)

    def forward(self, x):
        feats = self.get_encoder_features(x)
        return self.forward_from_features(feats)
