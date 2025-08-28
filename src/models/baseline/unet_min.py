"""
最小U-Net实现
简化版本，用于快速原型开发和测试
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 双卷积模块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)
    
# 下采样：MaxPool + DoubleConv
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))
    
# 上采样：双线性上采样 + 拼接(skip) + DoubleConv
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, x_skip):
        x = self.up(x)
        # shape align
        diffY = x_skip.size(2) - x.size(2)  # 高度之差（期望把 x pad 到和 skip 一样大）
        diffX = x_skip.size(3) - x.size(3)  # 宽度之差
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])  # [L, R, T, B]
        x = torch.cat([x_skip, x], dim=1)  # 拼接
        return self.conv(x)
    
# 输出1x1卷积：把特征映射到类别数
class OutConv(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
# 
class UNetMin(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, base=32):
        super().__init__()

        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.bot   = DoubleConv(base*8, base*16)

        # 上采样
        self.up1 = Up(base*16 + base*8, base*8)
        self.up2 = Up(base*8 + base*4, base*4)
        self.up3 = Up(base*4 + base*2, base*2)
        self.up4 = Up(base*2 + base, base)

        self.outc = OutConv(base, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bot(x4)

        # 上采样
        y = self.up1(xb, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)

        logits = self.outc(y)
        return torch.sigmoid(logits)