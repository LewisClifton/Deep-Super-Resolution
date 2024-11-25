import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64 , kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64 , kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        z = self.conv1(x)

        z = self.bn1(z)
        z = self.prelu1(z)

        z = self.conv2(z)
        z = self.bn2(z)

        x = x + z 

        return x

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels):
        super(PixelShuffleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=256 , kernel_size=3, stride=1, padding=1)

        self.shuffler1 = nn.PixelShuffle(upscale_factor=2)

        self.prelu1 = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffler1(x)
        x = self.prelu1(x)

        return x


class Generator(nn.Module):
    def __init__(self, factor=8, residual_blocks_count=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=9, stride=1, padding=4)
        self.prelu1 = self.prelu1 = nn.PReLU()

        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(residual_blocks_count)])

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64 , kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        pixel_shuffles = 3

        self.pixel_shuffle_blocks = nn.Sequential(*[PixelShuffleBlock(in_channels=64) for _ in range(pixel_shuffles)])

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3 , kernel_size=9, stride=1, padding=4)

        self.out = nn.Sigmoid()

    def forward(self, x):

        z = self.conv1(x)
        x = self.prelu1(z)
        z = self.residual_blocks(x)
        z = self.conv2(z)
        z = self.bn1(z)

        z = x + z 

        z = self.pixel_shuffle_blocks(z)

        z = self.conv3(z)
        
        z = self.out(z)
        return z

        