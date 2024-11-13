from torch.utils.data import Dataset
import cv2
import os
from torchvision import transforms
import numpy as np

class DIV2KDataset(Dataset):
    def __init__(self, LR_dir, scale_factor, HR_dir=None):
        super(DIV2KDataset, self).__init__()

        self.scale_factor = scale_factor

        self.LR_dir = LR_dir
        self.HR_dir = HR_dir

        self.LR_images = os.listdir(LR_dir)
        self.HR_images = os.listdir(HR_dir)

    def __getitem__(self, idx):

        # Get the low resolution image
        LR_image =  cv2.imread(os.path.join(self.LR_dir, self.LR_images[idx]))
        LR_image = cv2.cvtColor(LR_image, cv2.COLOR_BGR2RGB)

        # Get the expected dimensions of the HR image given the LR image shape and the scale factor
        width_HR = self.scale_factor * LR_image.shape[1]
        height_HR = self.scale_factor * LR_image.shape[0]

        if self.HR_image is None:
            # If testing without GT
            HR_image = np.zeros(shape = (width_HR, height_LR))
        else:
            # Read the HR GT image
            HR_image = cv2.imread(os.path.join(self.HR_dir, self.HR_images[idx]))
            HR_image = cv2.cvtColor(HR_image, cv2.COLOR_BGR2RGB)


        # The following ensures GT HR is scale factor times bigger than LR without making HR bigger 
        # if LR size times by scale factor exceeds HR size
        # using the if statement instead of only doing lines 39-46 ensures we keep the 
        # same LR size if possible.

        if width_HR > HR_image.shape[1] and height_HR > LR_image.shape[0]:
        
            width_HR = (width_HR // self.scale_factor) * self.scale_factor  # Find the largest multiple of z <= x
            height_HR = (height_HR // self.scale_factor) * self.scale_factor

            width_LR = width_HR / self.scale_factor
            height_LR = height_HR / self.scale_factor

            HR_image = cv2.resize(HR_image, (width_HR, height_HR))
            LR_image = cv2.resize(LR_image, (width_LR, height_LR))
        else:
            HR_image = cv2.resize(HR_image, (width_HR, height_HR))

        # Convert to tensor
        HR_image = transforms.ToTensor()(HR_image)
        LR_image = transforms.ToTensor()(LR_image)
        
        # Get the filename of the input
        filename, _ = os.path.splitext(self.LR_images[idx])
            
        return LR_image, HR_image, filename
    
    def __len__(self):
        return len(self.LR_images)