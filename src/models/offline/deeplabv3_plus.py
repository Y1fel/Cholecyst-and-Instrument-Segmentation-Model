# deeplabv3_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Block: Conv-BN-ReLU
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias = False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace = True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ResNet Bottleneck Block
class Bottleneck(nn.Module):
    # classic ResNet50/101 bottleneck
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, downsample = None):
        """
        inplanes: 输入通道
        planes   : Block 内部的“基准通道数”（输出会是 planes*4）
        stride   : 仅在 3x3 conv 上使用
        dilation : 仅在 3x3 conv 上使用（空洞率）
        downsample: 换维/下采样分支（1x1 conv），用于匹配残差的维度/步幅
        """
        super().__init__()
        width = planes

        # 1x1 downsample conv
        self.conv1 = ConvBNReLU(inplanes, width, kernel_size = 1, stride=1, padding = 0)
        # 3x3 conv (with stride and dilation)
        self.conv2 = ConvBNReLU(width, width, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation)
        # 1x1 upsample conv, no activation
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        
        return self.relu(out)
    
# ResNet Backbone (C1...C5)
class ResNetBackbone(nn.Module):
    """
    实现 ResNet-50/101 主干，并按 output_stride(=8/16) 设置 stride/dilation。
    输出: C5(深层特征), low_level(用于解码器，取 C2 = layer1 输出)
    """
    def __init__(self, layers = (3, 4, 6, 3), in_ch = 3, output_stride = 16):
        """
        layers: ResNet 每层 Block 数（50: (3,4,6,3), 101: (3,4,23,3)）
        """
        super().__init__()
        assert output_stride in {8, 16}, "Only output_stride 8 or 16 is supported."

        # Stem: 7x7 s=2 + 3x3 maxpool s=2  -> /4
        self.stem = nn.Sequential(
            ConvBNReLU(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # every layer's inplanes is planes * Bottleneck.expansion
        self.inplanes = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1, dilation=1)  # C2, /4
        # layer2 default stride=2 -> /8
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilation=1)  # C3, /8
        
        if output_stride == 16:
            # layer3 default stride=2 -> /16, layer4 stay /16 + dilation = 2
            self.layer3 = self._make_layer(256, layers[2], stride=2, dilation=1)  # C4, /16
            self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=2)  # C5, /16
        else:  # output_stride == 8
            # layer3: /8 + dilation = 2, layer4: /8 + dilation = 4
            self.layer3 = self._make_layer(256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=4)

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        """
        planes : 该层的“基准通道”
        blocks : Bottleneck 堆叠数
        stride : 仅用于第一个 bottleneck 的 3x3
        dilation: 第一个 bottleneck 的 3x3 用 dilation，其后续 block 的有效膨胀率与 stride/dilation 一致
        """
        downsample = None
        outplanes = planes * Bottleneck.expansion

        # if channels or stride don't match, use downsample branch (1x1 conv)
        if stride != 1 or self.inplanes != outplanes:
            # Build downsample branch when dimension change or downsampling is needed
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers = [Bottleneck(self.inplanes, planes, stride, dilation, downsample)]
        # layers.append(Bottleneck(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = outplanes
        # Remaining blocks: stride=1, dilation consistent with first block (maintain equivalent receptive field)
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, stride=1, dilation = dilation, downsample = None))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)    # C1-ish feature, /4, 64ch
        c2 = self.layer1(x)  # C2, /4, 256ch (64*4 by Bottleneck.expansion)
        c3 = self.layer2(c2) # C3, /8, 512ch (128*4)
        c4 = self.layer3(c3) # C4, /16, 1024ch (256*4)
        c5 = self.layer4(c4) # C5, /16, 2048ch (512*4)
        return  c5, c2  # return C5 (deep feature) and C2 (low-level feature)
    
# Atrous Spatial Pyramid Pooling (ASPP) Module
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    分支: 1x1, 3x3(d6), 3x3(d12), 3x3(d18)（OS=16），+ image pooling
    OS=8 时 dilation 通常加倍（12,24,36）
    """
    def __init__(self, in_ch, out_ch = 256, atrous_rates = (6, 12, 18), dropout = 0.1):
        super().__init__()
        r1, r2, r3 = atrous_rates

        self.b0 = ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1, padding=0)                 # 1x1 conv
        self.b1 = ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=1, padding=r1, dilation=r1)   # 3x3 conv, rate=r1
        self.b2 = ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=1, padding=r2, dilation=r2)   # 3x3 conv, rate=r2
        self.b3 = ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=1, padding=r3, dilation=r3)   # 3x3 conv, rate=r3

        # Image Pooling branch: global average pooling + 1x1 conv -> upsample return to input size
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        # Fuse
        self.project = nn.Sequential(
            ConvBNReLU(out_ch * 5, out_ch, kernel_size = 1, stride = 1, padding = 0),
            nn.Dropout2d(p = dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        h, w = x.shape[2:]
        x0 = self.b0(x)
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        
        x_pool = self.image_pool(x)
        # bilinear upsample to input size (default align_corners = False)
        x_pool = F.interpolate(x_pool, size = (h, w), mode = 'bilinear', align_corners = False)
        
        y = torch.cat([x0, x1, x2, x3, x_pool], dim = 1)  # concat along channel dim
        return self.project(y)

# DeepLabV3+ Model
class DeepLabV3Plus(nn.Module):
    def __init__(
        self, in_channels = 3, num_classes = 6, backbone = 'resnet50', output_stride = 16, base = 32,
        aspp_out_ch = 256, low_level_out_ch = 48, dropout = 0.1
    ):
        super().__init__()
        assert output_stride in (8, 16)
        assert backbone in ('resnet', 'resnet101')

        # main backbone
        layers = (3, 4, 6, 3) if output_stride == 16 else (3, 4, 23, 3)
        self.backbone = ResNetBackbone(layers = layers, in_ch = in_channels, output_stride = output_stride)

        # ASPP module
        rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.aspp = ASPP(in_ch = 512 * Bottleneck.expansion, out_ch = aspp_out_ch,
                         atrous_rates = rates, dropout = dropout)
        
        # Reduce low-level feature channels
        self.low_proj = ConvBNReLU(256, low_level_out_ch, kernel_size = 1, stride = 1, padding = 0)
        self.refine = nn.Sequential(
            ConvBNReLU(aspp_out_ch + low_level_out_ch, 256, kernel_size = 3, stride = 1, padding = 1),
            ConvBNReLU(256, 256, kernel_size = 3, stride = 1, padding = 1),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size = 1, bias = True)

    def forward(self, x):
        h, w = x.shape[2:]

        # Backbone
        c5, c2 = self.backbone(x) # c5: deep feature, c2: low-level feature

        # ASPP on c5
        y = self.aspp(c5)

        # upsample to c2 size
        y = F.interpolate(y, size = c2.shape[2:], mode = 'bilinear', align_corners = False)

        # reduce low-level feature channels
        c2p = self.low_proj(c2)

        # concat and refine
        y = torch.cat([y, c2p], dim = 1)
        y = self.refine(y)

        # back to original size
        y = F.interpolate(y, size = (h, w), mode = 'bilinear', align_corners = False)

        return self.classifier(y)
