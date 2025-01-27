import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Convolutional block with optional downsampling"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.conv2 = ConvBlock(mid_channels, in_channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class WeatherClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        
        # Feature extraction backbone
        self.layers = nn.Sequential(
            # Initial downsampling
            ConvBlock(3, 32, 3, padding=1),
            ConvBlock(32, 64, 3, stride=2, padding=1),
            ResidualBlock(64),
            
            # Stage blocks with downsampling
            ConvBlock(64, 128, 3, stride=2, padding=1),
            *[ResidualBlock(128) for _ in range(3)],
            
            ConvBlock(128, 256, 3, stride=2, padding=1),
            *[ResidualBlock(256) for _ in range(9)],
            
            ConvBlock(256, 512, 3, stride=2, padding=1),
            *[ResidualBlock(512) for _ in range(9)],
            
            ConvBlock(512, 1024, 3, stride=2, padding=1),
            *[ResidualBlock(1024) for _ in range(5)],
        )

        # Fully convolutional classification head with global context
        self.global_context = nn.Sequential(
            ConvBlock(1024, 512, 1),  # Channel reduction
            nn.Conv2d(512, num_classes, 1)  # Class prediction
        )
        
        # Global average pooling for any input size
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Extract features
        features = self.layers(x)
        
        # Global context and classification
        x = self.global_context(features)
        x = self.avgpool(x)
        
        return x.flatten(1)  # [batch, num_classes]

