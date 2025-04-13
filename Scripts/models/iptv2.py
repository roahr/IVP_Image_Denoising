import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim, eps=eps)

    def forward(self, x):
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        return self.ffn(x)

class SpatialMSA(nn.Module):
    def __init__(self, dim, patch_size=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x_patches = x.unfold(2, p, p).unfold(3, p, p).reshape(B, C, -1, p*p)
        x_patches = x_patches.permute(0, 2, 1, 3).reshape(B * (H // p) * (W // p), C, p*p)
        attn_out, _ = self.attn(x_patches, x_patches, x_patches)
        attn_out = attn_out.reshape(B, -1, C, p*p).permute(0, 2, 1, 3)
        return x  # simplified output (replace with actual reconstruction)

class ChannelMSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, -1).transpose(1, 2)
        out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        print(f"x_reshaped shape: {x_reshaped.shape}")
        return out.transpose(1, 2).view(B, C, H, W)

class SpatialChannelBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.spatial = SpatialMSA(dim)
        self.ffn1 = FeedForward(dim)

        self.norm2 = LayerNorm(dim)
        self.channel = ChannelMSA(dim)
        self.ffn2 = FeedForward(dim)

    def forward(self, x):
        x = x + self.ffn1(self.spatial(self.norm1(x)))
        x = x + self.ffn2(self.channel(self.norm2(x)))
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x):
        return self.up(x)

class IPTV2(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.input_proj = nn.Conv2d(3, dim, 3, padding=1)
        self.encoder1 = SpatialChannelBlock(dim)
        self.down1 = DownBlock(dim, dim*2)
        self.encoder2 = SpatialChannelBlock(dim*2)
        self.down2 = DownBlock(dim*2, dim*4)
        self.encoder3 = SpatialChannelBlock(dim*4)

        self.up2 = UpBlock(dim*4, dim*2)
        self.decoder2 = SpatialChannelBlock(dim*2)
        self.up1 = UpBlock(dim*2, dim)
        self.decoder1 = SpatialChannelBlock(dim)

        self.output_proj = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        x1 = self.input_proj(x)
        e1 = self.encoder1(x1)
        x2 = self.down1(e1)
        e2 = self.encoder2(x2)
        x3 = self.down2(e2)
        e3 = self.encoder3(x3)

        d2 = self.up2(e3) + e2
        d2 = self.decoder2(d2)
        d1 = self.up1(d2) + e1
        d1 = self.decoder1(d1)

        return self.output_proj(d1)
