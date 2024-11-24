from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import torch



class Vgg19Loss(nn.Module):

    # From reference paper: "We define the VGG loss based on the ReLU
    # activation layers of the pre-trained 19 layer VGG network
    # described in Simonyan and Zisserman [49]. With φi,j we
    # indicate the feature map obtained by the j-th convolution
    # (after activation) before the i-th maxpooling layer within the
    # VGG19 network, which we consider given. We then define
    # the VGG loss as the euclidean distance between the feature
    # representations of a reconstructed image and the reference image"

    # VGG19 architecture is show below, obtained using vgg19(weights = VGG19_Weights.DEFAULT).features
    # Sequential(
    #   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (1): ReLU(inplace=True)
    #   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (3): ReLU(inplace=True)
    #   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (6): ReLU(inplace=True)
    #   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (8): ReLU(inplace=True)
    #   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (11): ReLU(inplace=True)
    #   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (13): ReLU(inplace=True)
    #   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (15): ReLU(inplace=True)
    #   (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (17): ReLU(inplace=True)
    #   (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #   (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (20): ReLU(inplace=True)
    #   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (22): ReLU(inplace=True)
    #   (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (24): ReLU(inplace=True)
    #   (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (26): ReLU(inplace=True)
    #   (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (29): ReLU(inplace=True)
    #   (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (31): ReLU(inplace=True)
    #   (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (33): ReLU(inplace=True)
    #   (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (35): ReLU(inplace=True)
    #   (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # )

    # In the reference paper they choose i=5, j=4 so we want the feature map produced
    # after the fourth conv. layer in the fifth block in the network i.e. the very last conv.
    # so we extract only up to the last MaxPool2d and get the output of the relu at index 35
    # i.e. the feature map at that point in the network

    def __init__(self):
        super(Vgg19Loss, self).__init__()

        with torch.no_grad():
            # Get the layers of the pretrained vgg network
            vgg_layers = vgg19(weights = VGG19_Weights.DEFAULT).features

            # Only use layers up to the fourth conv. in the fifth block (layer 36)
            self.net = nn.Sequential(vgg_layers[:36]) 

            self.mse = nn.MSELoss()

            # We don't need gradients as the network is pre-trained and we just want the activations
            for param in self.net.parameters():
                param.requires_grad = False
    
    def forward(self, image1, image2):
        
        with torch.no_grad():
            # Get the vgg feature maps of the images
            feature_map1 = self.net(image1)
            feature_map2 = self.net(image2)

            # VGG loss is simply mse of the feature maps from the VGG network
            vgg_loss = self.mse(feature_map1, feature_map2)

            return vgg_loss
        

# Generator loss
def get_adversarial_loss(fake_output, bce_loss):
    adversarial_loss = bce_loss(fake_output, torch.ones_like(fake_output))
    return adversarial_loss

# Discriminator loss
def get_loss_D(real_output, fake_output, bce_loss):
    real_loss = bce_loss(real_output, torch.ones_like(real_output))
    fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))
    loss_D = real_loss + fake_loss
    return loss_D

# Get generator training loss function
class PerceptualLoss(nn.Module):
    def __init__(self):
        self.vgg_loss = Vgg19Loss()

    def forward(self, fake_output_G, HR_images, fake_output_D, bce_loss):

        with torch.no_grad():
            # Content less: MSE loss or vgg loss
            content_loss = self.vgg_loss(fake_output_G, HR_images)

            # Adversarial loss
            adversarial_loss_ = get_adversarial_loss(fake_output_D, bce_loss)

            # Perceptual loss
            perceptual_loss = content_loss + adversarial_loss_

            return perceptual_loss