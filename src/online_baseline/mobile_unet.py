import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

# 修复后的 MobileUNet - 针对 512×512 输入优化
class MobileUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.enc1 = DepthwiseSeparableConv(n_channels, 32)
        self.enc2 = DepthwiseSeparableConv(32, 64)
        self.enc3 = DepthwiseSeparableConv(64, 128)
        self.enc4 = DepthwiseSeparableConv(128, 256)

        self.middle = DepthwiseSeparableConv(256, 512)

        # 上采样层
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DepthwiseSeparableConv(512, 256)  # 512 = 256(up1) + 256(enc4)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DepthwiseSeparableConv(256, 128)  # 256 = 128(up2) + 128(enc3)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DepthwiseSeparableConv(128, 64)  # 128 = 64(up3) + 64(enc2)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = DepthwiseSeparableConv(64, 32)  # 64 = 32(up4) + 32(enc1)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        e1 = self.enc1(x)  # 512×512
        e2 = self.enc2(F.max_pool2d(e1, 2))  # 256×256
        e3 = self.enc3(F.max_pool2d(e2, 2))  # 128×128
        e4 = self.enc4(F.max_pool2d(e3, 2))  # 64×64

        # 中间层
        mid = self.middle(F.max_pool2d(e4, 2))  # 32×32

        # 解码器路径
        d1 = self.up1(mid)  # 64×64
        # 确保尺寸匹配
        if e4.size()[2] != d1.size()[2] or e4.size()[3] != d1.size()[3]:
            d1 = F.interpolate(d1, size=e4.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)  # 128×128
        # 确保尺寸匹配
        if e3.size()[2] != d2.size()[2] or e3.size()[3] != d2.size()[3]:
            d2 = F.interpolate(d2, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)  # 256×256
        # 确保尺寸匹配
        if e2.size()[2] != d3.size()[2] or e2.size()[3] != d3.size()[3]:
            d3 = F.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)  # 512×512
        # 确保尺寸匹配
        if e1.size()[2] != d4.size()[2] or e1.size()[3] != d4.size()[3]:
            d4 = F.interpolate(d4, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)

        return self.outc(d4)