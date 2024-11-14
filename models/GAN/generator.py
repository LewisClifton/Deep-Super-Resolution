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

        x = x + z # Element-wise sum layer

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
    def __init__(self, residual_blocks_count=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=9, stride=1, padding=4)
        self.prelu1 = self.prelu1 = nn.PReLU()

        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(residual_blocks_count)])

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64 , kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.pixel_shuffle_block1 = PixelShuffleBlock(in_channels=64)
        self.pixel_shuffle_block2 = PixelShuffleBlock(in_channels=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3 , kernel_size=9, stride=1, padding=4)

    def forward(self, x):

        z = self.conv1(x)
        x = self.prelu1(z)
        z = self.residual_blocks(x)
        z = self.conv2(z)
        z = self.bn1(z)

        #print(z.shape)
        z = x + z 
        #print(z.shape)
        z = self.pixel_shuffle_block1(z)
        #print(z.shape)
        z = self.pixel_shuffle_block2(z)

        z = self.conv3(z)

        return z

        