import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# prompt: I have a dataset of rgb images, depth images and encoded images. I want to use MIMO-UNet to retrieve rgb and deoth from encoded images. Assume pytorch. Add training and testing code

# Define the MIMO-UNet model
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_p=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers_1 = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),  # 3, 1
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers_1.append(nn.Dropout2d(p=dropout_p))  # dropout on feature maps

        layers_2 = [
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),  # 3, 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0:
            layers_2.append(nn.Dropout2d(p=dropout_p))  # dropout on feature maps
        layers = layers_1 + layers_2
        self.double_conv = nn.Sequential(*layers)


    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p=0.0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # The input channels to the conv layer after upsampling and concatenation
            # will be the sum of the channels from the upsampled tensor and the skip connection
            # The skip connection (x2) comes from a layer with half the channels of x1 before upsampling
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-leavesanddirt/blob/master/model/unet_parts.py
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class OutConv_V2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(OutConv_V2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.double_conv = DoubleConv(in_channels, in_channels, dropout_p=dropout_p)
        self.last_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.last_conv(x)
        return self.sigmoid(x)

class MIMO_UNet(nn.Module):
    def __init__(self, n_channels=3, n_rgb_classes=3, n_depth_classes=1, bilinear=True):
        super(MIMO_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_rgb_classes = n_rgb_classes
        self.n_depth_classes = n_depth_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        # The input channels to the Up modules' DoubleConv should be the sum of the channels
        # from the upsampled tensor and the skip connection.
        # The skip connection comes from the corresponding downsampling layer.
        # The output channels should be the desired output channels for that stage.
        self.up1 = Up(1024, 512, bilinear) # Concatenates 1024 (from down4) + 512 (from down3) -> 1536 input channels to DoubleConv
        self.up2 = Up(512, 256, bilinear)  # Concatenates 512 (from up1 output) + 256 (from down2) -> 768 input channels to DoubleConv
        self.up3 = Up(256, 128, bilinear)  # Concatenates 256 (from up2 output) + 128 (from down1) -> 384 input channels to DoubleConv
        self.up4 = Up(128, 64, bilinear)   # Concatenates 128 (from up3 output) + 64 (from inc) -> 192 input channels to DoubleConv


        # Output convolutions for RGB and Depth
        self.outconv_rgb = OutConv(64, n_rgb_classes)
        self.outconv_depth = OutConv(64, n_depth_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


        logits_rgb = self.outconv_rgb(x)
        logits_depth = self.outconv_depth(x)

        return logits_rgb, logits_depth

class MIMO_UNet_V2(nn.Module):
    def __init__(self, n_channels=3, n_rgb_classes=3, n_depth_classes=1, bilinear=True, dropout_p=0.0):
        super(MIMO_UNet_V2, self).__init__()
        self.n_channels = n_channels
        self.n_rgb_classes = n_rgb_classes
        self.n_depth_classes = n_depth_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, dropout_p=dropout_p)
        self.down1 = Down(64, 128, dropout_p=dropout_p)
        self.down2 = Down(128, 256, dropout_p=dropout_p)
        self.down3 = Down(256, 512, dropout_p=dropout_p)
        # self.down4 = Down(512, 1024)
        # The input channels to the Up modules' DoubleConv should be the sum of the channels
        # from the upsampled tensor and the skip connection.
        # The skip connection comes from the corresponding downsampling layer.
        # The output channels should be the desired output channels for that stage.
        # self.up1 = Up(1024, 512, bilinear) # Concatenates 1024 (from down4) + 512 (from down3) -> 1536 input channels to DoubleConv
        self.up1 = Up(512, 256, bilinear, dropout_p=dropout_p)  # Concatenates 512 (from up1 output) + 256 (from down2) -> 768 input channels to DoubleConv
        self.up2 = Up(256, 128, bilinear, dropout_p=dropout_p)  # Concatenates 256 (from up2 output) + 128 (from down1) -> 384 input channels to DoubleConv
        self.up3 = Up(128, 64, bilinear, dropout_p=dropout_p)   # Concatenates 128 (from up3 output) + 64 (from inc) -> 192 input channels to DoubleConv


        # Output convolutions for RGB and Depth
        self.outconv_rgb = OutConv_V2(64, n_rgb_classes, dropout_p=dropout_p)
        self.outconv_depth = OutConv_V2(64, n_depth_classes, dropout_p=dropout_p)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits_rgb = self.outconv_rgb(x)
        logits_depth = self.outconv_depth(x)

        return logits_rgb, logits_depth

def summarize_model():
    # Create a dummy input tensor (batch_size, channels, height, width)
    # Assuming input image size of 256x256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    # Instantiate the model
    model = MIMO_UNet().to(device)
    # --- Forward pass to get sizes ---
    print("Input size:", dummy_input.size())
    # Encoder
    x1 = model.inc(dummy_input)
    print("inc output size:", x1.size())
    x2 = model.down1(x1)
    print("down1 output size:", x2.size())

    x3 = model.down2(x2)
    print("down2 output size:", x3.size())

    x4 = model.down3(x3)
    print("down3 output size:", x4.size())

    x5 = model.down4(x4)
    print("down4 output size:", x5.size())

    # Decoder
    x = model.up1(x5, x4)
    print("up1 output size:", x.size())

    x = model.up2(x, x3)
    print("up2 output size:", x.size())

    x = model.up3(x, x2)
    print("up3 output size:", x.size())

    x = model.up4(x, x1)
    print("up4 output size:", x.size())

    # Output
    logits_rgb = model.outconv_rgb(x)
    print("outconv_rgb output size:", logits_rgb.size())

    logits_depth = model.outconv_depth(x)
    print("outconv_depth output size:", logits_depth.size())

def summarize_model_V2():
    # Create a dummy input tensor (batch_size, channels, height, width)
    # Assuming input image size of 256x256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    # Instantiate the model
    model = MIMO_UNet_V2().to(device)
    # --- Forward pass to get sizes ---
    print("Input size:", dummy_input.size())
    # Encoder
    x1 = model.inc(dummy_input)
    print("inc output size:", x1.size())
    x2 = model.down1(x1)
    print("down1 output size:", x2.size())

    x3 = model.down2(x2)
    print("down2 output size:", x3.size())

    x4 = model.down3(x3)
    print("down3 output size:", x4.size())

    # Decoder
    x = model.up1(x4, x3)
    print("up1 output size:", x.size())

    x = model.up2(x, x2)
    print("up2 output size:", x.size())

    x = model.up3(x, x1)
    print("up3 output size:", x.size())

    # Output
    logits_rgb = model.outconv_rgb(x)
    print("outconv_rgb output size:", logits_rgb.size())

    logits_depth = model.outconv_depth(x)
    print("outconv_depth output size:", logits_depth.size())