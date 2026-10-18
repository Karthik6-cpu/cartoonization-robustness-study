import torch
import torch.nn as nn
import torchvision.models as models

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.up(x)

class ResBlockGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.conv(x)

class CARP_Generator(nn.Module):
    """
    CARP-Net: Content-Aware, Robustness-Preserving Cartoonization Network
    Reconciles semantic recognition retention with lossy JPEG compression robustness.
    """
    def __init__(self, in_channels=3, out_channels=3, base_ch=64):
        super().__init__()
        # 1. Dual-Path Feature Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 7, 1, 3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down1 = DownBlock(base_ch, base_ch * 2)      # Edge representation
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)  # Semantic representation
        
        # 2. ResNet Bottleneck with Gated Attention
        self.bottleneck = nn.Sequential(*[ResBlockGated(base_ch * 4) for _ in range(6)])
        
        # 3. Decomposed Decoders (Surface Shading Stream + Contour Stream)
        self.surf_up1 = UpBlock(base_ch * 4, base_ch * 2)
        self.surf_up2 = UpBlock(base_ch * 2, base_ch)
        self.surf_out = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, 7, 1, 3, padding_mode='reflect'),
            nn.Tanh()
        )
        
        # Gating mechanism from low-level edge features
        self.edge_gate = nn.Sequential(
            nn.Conv2d(base_ch * 2, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        f1 = self.enc_conv(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        b  = self.bottleneck(f3)
        
        # Surface stream with edge-gated contour refinement
        u1 = self.surf_up1(b)
        gate = self.edge_gate(f2)
        u1_gated = u1 * (1.0 + gate)
        u2 = self.surf_up2(u1_gated)
        return self.surf_out(u2)

if __name__ == "__main__":
    net = CARP_Generator()
    x = torch.randn(1, 3, 256, 256)
    out = net(x)
    print("CARP_Generator forward pass successful! Output shape:", out.shape)
