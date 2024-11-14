from torch.utils.data import Dataset
import cv2
import os
from torchvision import transforms
import numpy as np
from utils.SRGAN import get_train_patches


def get_image_pair(dataset_config, idx):
    # Get the low resolution image
    LR_image =  cv2.imread(os.path.join(dataset_config.LR_dir, dataset_config.LR_images[idx]))
    LR_image = cv2.cvtColor(LR_image, cv2.COLOR_BGR2RGB)

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
    
        width_HR = (width_HR // dataset_config.scale_factor) * dataset_config.scale_factor  # Find the largest multiple of z <= x
        height_HR = (height_HR // dataset_config.scale_factor) * dataset_config.scale_factor

        width_LR = width_HR / dataset_config.scale_factor
        height_LR = height_HR / dataset_config.scale_factor

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


class DIV2KDataset(Dataset):
    def __init__(self, LR_dir, scale_factor, num_images=-1, HR_dir=None):
        super(DIV2KDataset, self).__init__()

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
    

class GANTrainDIV2KDataset(Dataset):
    def __init__(self, LR_dir, HR_dir, scale_factor, num_images=-1, HR_patch_size=(96,96), num_patches=16):
        super(GANTrainDIV2KDataset, self).__init__()

        self.scale_factor = scale_factor

        self.LR_dir = LR_dir
        self.HR_dir = HR_dir

        self.LR_images = os.listdir(LR_dir)
        self.HR_images = os.listdir(HR_dir)

        if num_images > 0:
            self.LR_images = self.LR_images[:num_images]
            self.HR_images = self.HR_images[:num_images]

        self.HR_patch_size = HR_patch_size
        self.num_patches = num_patches


    def __getitem__(self, idx):

        LR_image, HR_image, filename = get_image_pair(self, idx)

        LR_image, HR_image = scale_images(LR_image, HR_image)

        LR_patches, HR_patches = get_train_patches(LR_image, HR_image, self.scale_factor, self.HR_patch_size, self.num_patches)

        return LR_patches, HR_patches, filename
    
    def __len__(self):
        return len(self.LR_images)