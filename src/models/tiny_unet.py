import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class TinyUnet(nn.Module):
    """
    Extremely lightweight Unet for mobile/edge.
    Initial filters: 16. Depth: 4 resolution levels.
    """
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.enc1 = TinyBlock(in_channels, 16)
        self.enc2 = TinyBlock(16, 32)
        self.enc3 = TinyBlock(32, 64)
        self.enc4 = TinyBlock(64, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = TinyBlock(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = TinyBlock(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = TinyBlock(32, 16)
        
        self.final = nn.Conv2d(16, out_channels, 1)

    def get_encoder_features(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(F.max_pool2d(f1, 2))
        f3 = self.enc3(F.max_pool2d(f2, 2))
        f4 = self.enc4(F.max_pool2d(f3, 2))
        return [f1, f2, f3, f4]

    def forward_from_features(self, feats):
        f1, f2, f3, f4 = feats
        x = self.up3(f4)
        x = self.dec3(torch.cat([x, f3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, f2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, f1], dim=1))
        return self.final(x)

    def forward(self, x):
        feats = self.get_encoder_features(x)
        return self.forward_from_features(feats)
