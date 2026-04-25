import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class SegResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, init_filters=32):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, init_filters, kernel_size=3, padding=1)
        
        # Encoder
        self.e1 = nn.Sequential(ResBlock(init_filters))
        self.d1 = nn.Conv2d(init_filters, init_filters*2, kernel_size=3, stride=2, padding=1)
        
        self.e2 = nn.Sequential(ResBlock(init_filters*2), ResBlock(init_filters*2))
        self.d2 = nn.Conv2d(init_filters*2, init_filters*4, kernel_size=3, stride=2, padding=1)
        
        self.e3 = nn.Sequential(ResBlock(init_filters*4), ResBlock(init_filters*4))
        self.d3 = nn.Conv2d(init_filters*4, init_filters*8, kernel_size=3, stride=2, padding=1)
        
        self.e4 = nn.Sequential(ResBlock(init_filters*8), ResBlock(init_filters*8), ResBlock(init_filters*8), ResBlock(init_filters*8))
        
        # Decoder
        self.u3 = nn.ConvTranspose2d(init_filters*8, init_filters*4, kernel_size=2, stride=2)
        self.de3 = ResBlock(init_filters*4)
        
        self.u2 = nn.ConvTranspose2d(init_filters*4, init_filters*2, kernel_size=2, stride=2)
        self.de2 = ResBlock(init_filters*2)
        
        self.u1 = nn.ConvTranspose2d(init_filters*2, init_filters, kernel_size=2, stride=2)
        self.de1 = ResBlock(init_filters)
        
        self.final_conv = nn.Conv2d(init_filters, out_channels, kernel_size=1)

    def get_encoder_features(self, x):
        f1 = self.e1(self.init_conv(x))
        f2 = self.e2(self.d1(f1))
        f3 = self.e3(self.d2(f2))
        f4 = self.e4(self.d3(f3))
        return [f1, f2, f3, f4]

    def forward_from_features(self, feats):
        f1, f2, f3, f4 = feats
        x = self.de3(self.u3(f4) + f3)
        x = self.de2(self.u2(x) + f2)
        x = self.de1(self.u1(x) + f1)
        return self.final_conv(x)

    def forward(self, x):
        feats = self.get_encoder_features(x)
        return self.forward_from_features(feats)
