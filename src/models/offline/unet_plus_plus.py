# unet_plus_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 需要优化显存消耗问题
# Convolutional Block used in UNet++
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch = None, dropout_p = 0.0):
        super().__init__()
        if mid_ch is None: # If no middle channel is specified, use the output channels
            mid_ch = out_ch
        
        # Define the convolutional block with two Conv2d layers, BatchNorm, ReLU, and optional Dropout
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace = True),

            nn.Conv2d(mid_ch, out_ch, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),

            nn.Dropout2d(p = dropout_p) if dropout_p > 0 else nn.Identity()
        )

    def forward(self, x): # Forward pass through the convolutional block
        return self.block(x)

class UpSample(nn.Module):
    """
    上采样模块：
      - 只负责把特征图放大到指定尺寸（不改通道）
      - 用 bilinear (align_corners=False) 更通用，避免插值伪影
    """
    def __init__(self, mode = 'bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, x, target_size):
        # Only pass align_corners for modes that support it
        if self.mode in {'linear', 'bilinear', 'bicubic', 'trilinear'}:
            return F.interpolate(x, size = target_size, mode = self.mode, align_corners = False)
        else:
            return F.interpolate(x, size = target_size, mode = self.mode)
        # return Fn.interpolate(x, size = target_size, mode = self.mode, align_corners = False)

class OutConv(nn.Module):
    """
    最后一层 1x1 conv，把通道压到 num_classes（输出 logits）
    """
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

# UNet++ Model
class UNetPlusPlus(nn.Module):
    """
    正宗 UNet++（Nested U-Net）实现：
      - filters 决定每个分辨率的通道数
      - X^{i,j} 结点按照论文公式构建（dense skip）
      - deep supervision 可选
    """

    def __init__(
        self,
        in_ch: int = 3,
        num_classes: int = 6, # need to change
        base_ch: int = 32,
        deep_supervision: bool = False,
        dropout_p: float = 0.0,
        up_mode: str = 'bilinear',
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Define the number of filters at each level
        # i=0..4 correspond to scales 1/1, 1/2, 1/4, 1/8, 1/16
        filters = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.up = UpSample(mode = up_mode)

        # Define the convolutional blocks for each node in the UNet++ architecture
        # Encoder backbone: X^{0,0}..X^{4,0}
        self.conv0_0 = ConvBlock(in_ch,      filters[0], dropout_p = dropout_p)
        self.conv1_0 = ConvBlock(filters[0], filters[1], dropout_p = dropout_p)
        self.conv2_0 = ConvBlock(filters[1], filters[2], dropout_p = dropout_p)
        self.conv3_0 = ConvBlock(filters[2], filters[3], dropout_p = dropout_p)
        self.conv4_0 = ConvBlock(filters[3], filters[4], dropout_p = dropout_p)

        # Define the max pooling layers for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 下面是“嵌套密集跳连”的解码网格：
        # 每个节点 X^{i,j} 的输入通道 =  同层前序节点拼接 + 下层 j-1 节点上采样后拼接 的通道总和
        # 为了实现简洁，我们直接在 forward 里 cat，然后这里把 ConvBlock 的 in_ch 写成 “任意”占位，
        # 真正的 in_ch 由 forward 里的 cat 决定（PyTorch 允许这么做）。
        # 但为了清晰，我们仍按 filters[i] 作为该层输出通道。

        # Decoder with nested dense skip connections: X^{i,j}, j=1..4
        # X^{0,1}, X^{1,1}, X^{2,1}, X^{3,1}

        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0], dropout_p = dropout_p) # cat[x0_0, up(x1_0)]
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1], dropout_p = dropout_p)
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2], dropout_p = dropout_p)
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3], dropout_p = dropout_p)

        # X^{0,2}, X^{1,2}, X^{2,2}
        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0], dropout_p = dropout_p) # cat[x0_0, x0_1, up(x1_1)]
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1], dropout_p = dropout_p)
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2], dropout_p = dropout_p)

        # X^{0,3}, X^{1,3}
        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0], dropout_p = dropout_p)
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1], dropout_p = dropout_p)

        # X^{0,4}
        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0], dropout_p = dropout_p)

        # output header
        if self.deep_supervision:
            self.outconv1 = OutConv(filters[0], num_classes)
            self.outconv2 = OutConv(filters[0], num_classes)
            self.outconv3 = OutConv(filters[0], num_classes)
            self.outconv4 = OutConv(filters[0], num_classes)
        else:
            self.outconv = OutConv(filters[0], num_classes)

    def forward(self, x):
        # Encoder path
        # backbone
        x0_0 = self.conv0_0(x)               # H, W
        x1_0 = self.conv1_0(self.pool(x0_0)) # H/2, W/2
        x2_0 = self.conv2_0(self.pool(x1_0)) # H/4, W/4
        x3_0 = self.conv3_0(self.pool(x2_0)) # H/8, W/8
        x4_0 = self.conv4_0(self.pool(x3_0)) # H/16, W/16

        # Decoder path with nested dense skip connections
        # X^{i, j}, j=1..4
        # first column j=1
        # X^{0,1} = conv([X^{0,0}, up(X^{1,0})])
        x1_0_up = self.up(x1_0, x0_0.shape[2:]) # upsample to x0_0 size
        x0_1 = self.conv0_1(torch.cat([x0_0, x1_0_up], dim=1)) # H, W

        x2_0_up = self.up(x2_0, x1_0.shape[2:])
        x1_1 = self.conv1_1(torch.cat([x1_0, x2_0_up], dim=1)) # H/2, W/2

        x3_0_up = self.up(x3_0, x2_0.shape[2:])
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_0_up], dim=1)) # H/4, W/4

        x4_0_up = self.up(x4_0, x3_0.shape[2:])
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0_up], dim=1)) # H/8, W/8

        # second column j=2
        # X^{0,2} = Conv([X^{0,0}, X^{0,1}, Up(X^{1,1})])
        x1_1_up = self.up(x1_1, x0_0.shape[2:])
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1_up], dim=1)) # H, W

        x2_1_up = self.up(x2_1, x1_0.shape[2:])
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1_up], dim=1)) # H/2, W/2

        x3_1_up = self.up(x3_1, x2_0.shape[2:])
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1_up], dim=1)) # H/4, W/4

        # third column j=3
        # X^{0,3} = Conv([X^{0,0}, X^{0,1}, X^{0,2}, Up(X^{1,2})])
        x1_2_up = self.up(x1_2, x0_0.shape[2:])
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2_up], dim=1))

        x2_2_up = self.up(x2_2, x1_0.shape[2:])
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2_up], dim=1))

        # fourth column j=4
        # X^{0,4} = Conv([X^{0,0}, X^{0,1}, X^{0,2}, X^{0,3}, Up(X^{1,3})])
        x1_3_up = self.up(x1_3, x0_0.shape[2:])
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3_up], dim=1))

        # output
        if self.deep_supervision:
            # Return logits at four scales with the same resolution; external code should handle loss weighting
            return [
                self.outconv1(x0_1),
                self.outconv2(x0_2),
                self.outconv3(x0_3),
                self.outconv4(x0_4),
            ]
        else:
            return self.outconv(x0_4) # Return final single output logits, no activation
