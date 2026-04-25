import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape for Attention
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Norm + Attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # Norm + MLP
        x_norm = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm)
        x_flat = x_flat + mlp_out
        
        # Reshape back
        return rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)

class MISSFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(2)
        
        self.enc3 = EnhancedTransformerBlock(128)
        self.p3 = nn.MaxPool2d(2)
        
        self.enc4 = EnhancedTransformerBlock(256) # We'll need a projection
        self.proj3_4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Decoder
        self.u3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        self.u2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.u1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def get_encoder_features(self, x):
        f1 = F.relu(self.enc1(x))
        f2 = F.relu(self.enc2(self.p1(f1)))
        f3 = self.enc3(self.p2(f2))
        f4 = self.enc4(self.proj3_4(self.p3(f3)))
        return [f1, f2, f3, f4]

    def forward_from_features(self, feats):
        f1, f2, f3, f4 = feats
        x = self.u3(f4)
        x = F.relu(self.dec3(torch.cat([x, f3], dim=1)))
        x = self.u2(x)
        x = F.relu(self.dec2(torch.cat([x, f2], dim=1)))
        x = self.u1(x)
        x = F.relu(self.dec1(x)) # Simplified skip for f1
        return self.final(x)

    def forward(self, x):
        feats = self.get_encoder_features(x)
        return self.forward_from_features(feats)
