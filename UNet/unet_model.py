# License: https://github.com/milesial/Pytorch-UNet
""" Full assembly of the parts to form the complete network """

from torch import nn
from UNet.unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)     # H   x W
        x2 = self.down1(x1)  # H/2 x W/2
        x3 = self.down2(x2)  # H/4 x W/4
        x4 = self.down3(x3)  # H/8 x W/8
        x5 = self.down4(x4)  # H/16x W/16
        x = self.up1(x5, x4) # H/8 x W/8
        x = self.up2(x, x3)  # H/4 x W/4
        x = self.up3(x, x2)  # H/2 x W/2
        x = self.up4(x, x1)  # H   x W
        logits = self.outc(x)# H   x W

        return logits
