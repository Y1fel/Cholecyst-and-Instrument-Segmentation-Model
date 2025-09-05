import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAdapter(nn.Module):
    """适配器层，用于在线学习时快速适应新数据分布"""

    def __init__(self, in_channels, reduction_ratio=4):
        super(ConvAdapter, self).__init__()
        self.down_sample = nn.Conv2d(in_channels, in_channels // reduction_ratio,
                                     kernel_size=1, stride=1, padding=0)
        self.non_linear = nn.ReLU(inplace=True)
        self.up_sample = nn.Conv2d(in_channels // reduction_ratio, in_channels,
                                   kernel_size=1, stride=1, padding=0)
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        identity = x
        x = self.down_sample(x)
        x = self.non_linear(x)
        x = self.up_sample(x)
        return identity + x * self.scale + self.bias


class AdaptiveUNet(nn.Module):
    """带有适配器层的UNet模型，支持在线学习"""

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(AdaptiveUNet, self).__init__()

        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.adapter1 = ConvAdapter(features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.adapter2 = ConvAdapter(features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.adapter3 = ConvAdapter(features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.adapter4 = ConvAdapter(features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.bottleneck_adapter = ConvAdapter(features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.decoder_adapter4 = ConvAdapter(features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.decoder_adapter3 = ConvAdapter(features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.decoder_adapter2 = ConvAdapter(features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        self.decoder_adapter1 = ConvAdapter(features)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        self.freeze_encoder_decoder()

    def freeze_encoder_decoder(self):
        """冻结编码器和解码器参数，只保留适配器可训练"""
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = self.adapter1(enc1)

        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.adapter2(enc2)

        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.adapter3(enc3)

        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.adapter4(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.bottleneck_adapter(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.decoder_adapter4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self.decoder_adapter3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.decoder_adapter2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.decoder_adapter1(dec1)

        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1)