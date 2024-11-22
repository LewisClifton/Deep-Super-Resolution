from torch.utils.data import Dataset
import cv2
import os
from torchvision import transforms
import numpy as np
import torch
from utils.degradation import *


def get_image_pair(dataset_config, idx):
    # Get the low resolution image
    LR_image =  cv2.imread(os.path.join(dataset_config.LR_dir, dataset_config.LR_images[idx]))
    LR_image = cv2.cvtColor(LR_image, cv2.COLOR_BGR2RGB)

    if dataset_config.downsample:
        LR_image = downsample(LR_image)
                              
    # Apply noise degredation if necessary:
    if dataset_config.noise_type is not None:
        if dataset_config.noise_type['type'] == 'SaltAndPepper':
            LR_image = add_salt_pepper_noise(LR_image, s=dataset_config.noise_type['s'], p=dataset_config.noise_type['p'])
        elif dataset_config.noise_type['type'] == 'Gaussian':
            LR_image = add_gaussian_noise(LR_image, std=dataset_config.noise_type['std'])
                              
    # Get the expected dimensions of the HR image given the LR image shape and the scale factor
    width_HR = dataset_config.scale_factor * LR_image.shape[1]
    height_HR = dataset_config.scale_factor * LR_image.shape[0]

    if dataset_config.HR_dir is None:
        # If testing without GT
        HR_image = np.zeros(shape = (width_HR, height_HR))
    else:
        # Read the HR GT image
        HR_image = cv2.imread(os.path.join(dataset_config.HR_dir, dataset_config.HR_images[idx]))
        HR_image = cv2.cvtColor(HR_image, cv2.COLOR_BGR2RGB)


    # The following ensures GT HR is scale factor times bigger than LR without making HR bigger 
    # if LR size times by scale factor exceeds HR size
    # using the if statement instead of only doing lines 39-46 ensures we keep the 
    # same LR size if possible.

    if width_HR > HR_image.shape[1] and height_HR > LR_image.shape[0]:
    
        width_HR = int((width_HR // dataset_config.scale_factor) * dataset_config.scale_factor)  # Find the largest multiple of z <= x
        height_HR = int((height_HR // dataset_config.scale_factor) * dataset_config.scale_factor)

        width_LR = int(width_HR / dataset_config.scale_factor)
        height_LR = int(height_HR / dataset_config.scale_factor)

        HR_image = cv2.resize(HR_image, (width_HR, height_HR))
        LR_image = cv2.resize(LR_image, (width_LR, height_LR))

    else:
        HR_image = cv2.resize(HR_image, (width_HR, height_HR))

    # Convert to tensor
    HR_image = transforms.ToTensor()(HR_image)
    LR_image = transforms.ToTensor()(LR_image)
    
    # Get the filename of the input
    filename, _ = os.path.splitext(dataset_config.LR_images[idx])

    return LR_image, HR_image, filename

def scale_images(LR_image, HR_image):
    # Scale LR images to [0,1] per the reference paper
    LR_image /= 255.0
    
    # Scale HR images to [-1,1] per the reference paper
    HR_image /= 255.0
    HR_image *= 2   
    HR_image -= 1

    return LR_image, HR_image


# Would've been neater to use inheritance from a DIV2KDataset class but these classes are only short anyway

class DIPDIV2KDataset(Dataset):
    def __init__(self, LR_dir, scale_factor, downsample=False, noise_type=None, num_images=-1, HR_dir=None):
        super(DIPDIV2KDataset, self).__init__()

        self.downsample = downsample
        self.noise_type = noise_type

        self.scale_factor = scale_factor

        self.LR_dir = LR_dir
        self.HR_dir = HR_dir

        self.LR_images = os.listdir(LR_dir)
        self.HR_images = os.listdir(HR_dir)

        if num_images > 0:
            self.LR_images = self.LR_images[:num_images]
            self.HR_images = self.HR_images[:num_images]

    def __getitem__(self, idx):

        LR_image, HR_image, filename = get_image_pair(self, idx)

        LR_image, HR_image = scale_images(LR_image, HR_image)

        return LR_image, HR_image, filename
    
    def __len__(self):
        return len(self.LR_images)
    

class GANDIV2KDataset(Dataset):
    def __init__(self, LR_dir, scale_factor, downsample=False, noise_type=None, num_images=-1, HR_dir=None, LR_patch_size=(96,96), train=False):
        super(GANDIV2KDataset, self).__init__()

        self.train = train

        self.downsample = downsample
        self.noise_type = noise_type

        self.scale_factor = scale_factor

        self.LR_dir = LR_dir
        self.HR_dir = HR_dir

        self.LR_images = os.listdir(LR_dir)
        self.HR_images = os.listdir(HR_dir)

        if num_images > 0:
            self.LR_images = self.LR_images[:num_images]
            self.HR_images = self.HR_images[:num_images]

        self.LR_patch_size = LR_patch_size

    def get_train_patches(self, LR_image, HR_image):
    
        # Get size of HR image and it's patch size
        _, LR_H, LR_W = LR_image.size()
        LR_patch_W, LR_patch_H = self.LR_patch_size

        # Get center of the HR patch
        LR_center_x = np.random.randint(LR_patch_W // 2, LR_W - LR_patch_W // 2)
        LR_center_y = np.random.randint(LR_patch_H // 2, LR_H - LR_patch_H // 2)

        # Get LR patch edges
        LR_left = int( LR_center_x - LR_patch_W // 2 )
        LR_right = int( LR_left + LR_patch_W )
        LR_top = int( LR_center_y - LR_patch_H // 2 ) 
        LR_bottom = int( LR_top + LR_patch_H )

        # Get HR patch edges
        HR_left = LR_left * self.scale_factor 
        HR_right = LR_right * self.scale_factor 
        HR_top = LR_top * self.scale_factor 
        HR_bottom = LR_bottom * self.scale_factor 

        # Extract LR and LR patches
        LR_patch = LR_image[:, LR_top:LR_bottom, LR_left:LR_right]
        HR_patch = HR_image[:, HR_top:HR_bottom, HR_left:HR_right]

        return LR_patch, HR_patch


    def __getitem__(self, idx):

        LR_image, HR_image, filename = get_image_pair(self, idx)

        LR_image, HR_image = scale_images(LR_image, HR_image)

        # If training use image patches:
        if self.train:
            LR_patches, HR_patches = self.get_train_patches(LR_image, HR_image)

            return LR_patches, HR_patches, filename

        else:
            return LR_image, HR_image, filename

    def __len__(self):
        return len(self.LR_images)