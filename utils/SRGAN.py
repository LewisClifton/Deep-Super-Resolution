import torch.nn.functional as F

from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import torch
import numpy as np



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

        # Get the vgg feature maps of the images
       
        feature_map1 = self.net(image1)
        feature_map2 = self.net(image2)

        # VGG loss is simply mse of the feature maps from the VGG network
        vgg_loss = self.mse(feature_map1, feature_map2)

        return vgg_loss


def get_train_patches(LR_image, HR_image, scale_factor, HR_patch_size=(96, 96), num_patches=16):
    
    # Get size of HR image and it's patch size
    _, HR_H, HR_W = HR_image.size()
    HR_patch_W, HR_patch_H = HR_patch_size

    LR_patches = torch.empty((num_patches, 3, int(HR_patch_H/scale_factor), int(HR_patch_W/scale_factor)), dtype=torch.float32)
    HR_patches = torch.empty((num_patches, 3, HR_patch_H, HR_patch_W), dtype=torch.float32)

    for i in range(num_patches):
        # Get center of the HR patch
        HR_center_x = np.random.randint(HR_patch_W // 2, HR_W - HR_patch_W // 2)
        HR_center_y = np.random.randint(HR_patch_H // 2, HR_H - HR_patch_H // 2)

        # Get the HR and LR patch edges
        HR_left = int( HR_center_x - HR_patch_W // 2 )
        HR_right = int( HR_left + HR_patch_W )
        HR_top = int( HR_center_y - HR_patch_H // 2 ) 
        HR_bottom = int( HR_top + HR_patch_H )


        LR_left = int( HR_left / scale_factor )
        LR_right = int( HR_right / scale_factor )
        LR_top = int( HR_top / scale_factor )
        LR_bottom = int( HR_bottom / scale_factor )

        # Extract HR and LR patches
        HR_patch = HR_image[:, HR_top:HR_bottom, HR_left:HR_right]

        LR_patch = LR_image[:, LR_top:LR_bottom, LR_left:LR_right]


        # Add to patches tensor stack
        HR_patches[i] = HR_patch
        LR_patches[i] = LR_patch

    return LR_patches, HR_patches