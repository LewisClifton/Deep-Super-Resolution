from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
from utils.degradation import *


def get_image_pair(dataset_config, idx):
        # Read the HR GT image
    HR_image_path = os.path.join(dataset_config.HR_dir, dataset_config.HR_images[idx])
    HR_image = Image.open(HR_image_path).convert("RGB")

    # Get the filename of the input
    filename, _ = os.path.splitext(dataset_config.HR_images[idx])

    # Get the LR image
    LR_image_path = os.path.join(dataset_config.LR_dir, f'{filename}x8.png')
    LR_image = Image.open(LR_image_path).convert("RGB") 

    # Unfortunately the images in the dataset are too big to use in the forward pass so apply downsampling by default
    LR_image = downsample(LR_image, 2)
    HR_image = downsample(HR_image, 2)

    # Get the expected dimensions of the HR image given the LR image shape and the scale factor
    width_LR, height_LR = LR_image.size
    width_HR = dataset_config.scale_factor * width_LR
    height_HR = dataset_config.scale_factor * height_LR

    # Ensure GT HR is scale-factor times bigger than LR without exceeding original HR size
    if width_HR > HR_image.size[0] and height_HR > HR_image.size[1]:
        # Adjust dimensions to the largest multiples of scale factor less than HR size
        width_HR = (HR_image.size[0] // dataset_config.scale_factor) * dataset_config.scale_factor
        height_HR = (HR_image.size[1] // dataset_config.scale_factor) * dataset_config.scale_factor
        width_LR = width_HR // dataset_config.scale_factor
        height_LR = height_HR // dataset_config.scale_factor

        HR_image = HR_image.resize((width_HR, height_HR), Image.BICUBIC)
        LR_image = LR_image.resize((width_LR, height_LR), Image.BICUBIC)
    else:
        HR_image = HR_image.resize((width_HR, height_HR), Image.BICUBIC)


    LR_image = np.array(LR_image)
    HR_image = np.array(HR_image)

    # Apply further downsampling if investigating greater scale factors
    if dataset_config.downsample:
        LR_image = downsample(LR_image)

    # Apply noise degradation if necessary:
    if dataset_config.noise_type is not None:
        if dataset_config.noise_type['type'] == 'SaltAndPepper':
            LR_image = add_salt_pepper_noise(LR_image, s=dataset_config.noise_type['s'], p=dataset_config.noise_type['p'])
        elif dataset_config.noise_type['type'] == 'Gaussian':
            LR_image = add_gaussian_noise(LR_image, std=dataset_config.noise_type['std'])

    # Convert to tensor
    LR_image = transforms.ToTensor()(LR_image)
    HR_image = transforms.ToTensor()(HR_image)

    return LR_image, HR_image, filename




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

        return LR_image, HR_image, filename
    
    def __len__(self):
        return len(self.LR_images)
    

class GANDIV2KDataset(Dataset):
    def __init__(self, LR_dir, scale_factor, downsample=False, noise_type=None, num_images=-1, HR_dir=None, LR_patch_size=None, train=False):
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

        # Extract LR and LR patch
        LR_patch = LR_image[:, LR_top:LR_bottom, LR_left:LR_right]
        HR_patch = HR_image[:, HR_top:HR_bottom, HR_left:HR_right]

        return LR_patch, HR_patch

    @staticmethod
    def scale_images(LR_image, HR_image):
        # Scale LR images to [0,1] per the reference paper
        LR_image /= 255.0
        
        # Scale HR images to [-1,1] per the reference paper
        HR_image /= 255.0
        HR_image *= 2   
        HR_image -= 1

        return LR_image, HR_image

    def __getitem__(self, idx):

        LR_image, HR_image, filename = get_image_pair(self, idx)

        LR_image, HR_image = GANDIV2KDataset.scale_images(LR_image, HR_image)

        # If training use image patches:
        if self.train:
            LR_patches, HR_patches = self.get_train_patches(LR_image, HR_image)

            return LR_patches, HR_patches, filename

        else:
            return LR_image, HR_image, filename

    def __len__(self):
        return len(self.LR_images)