import torch.nn as nn
import torch

class DiscriminatorConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscriminatorConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels , kernel_size=3, stride=stride, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, HR_image_shape):
        super(Discriminator, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=3, stride=1, padding=1)

        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2)

        self.convblocks = nn.Sequential(*[DiscriminatorConvBlock(in_channels=64, out_channels=64, stride=2),
                                         DiscriminatorConvBlock(in_channels=64, out_channels=128, stride=1),
                                         DiscriminatorConvBlock(in_channels=128, out_channels=128, stride=2),
                                         DiscriminatorConvBlock(in_channels=128, out_channels=256, stride=1),
                                         DiscriminatorConvBlock(in_channels=256, out_channels=256, stride=2),
                                         DiscriminatorConvBlock(in_channels=256, out_channels=512, stride=1),
                                         DiscriminatorConvBlock(in_channels=512, out_channels=512, stride=2)])
        
        dense1_shape = self.fc_input_shape(HR_image_shape)

        self.dense1 = nn.Linear(in_features=dense1_shape, out_features=1024)

        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2)

        self.dense2 = nn.Linear(1024, 1)

        self.sigmoid = nn.Sigmoid()


    def fc_input_shape(self, HR_image_shape):
        with torch.no_grad():
            x = torch.ones(1, 3, HR_image_shape[0], HR_image_shape[1])
            x = self.conv(x)
            x = self.leakyrelu1(x)
            x = self.convblocks(x)
            x = x.view(x.size(0), -1) 

            return x.shape[1]

    def forward(self, x):

        x = self.conv(x)
        x = self.leakyrelu1(x)

        x = self.convblocks(x)

        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.leakyrelu2(x)

        x = x.view(x.size(0), -1)
        x = self.dense2(x)

        x = self.sigmoid(x)

        return x






