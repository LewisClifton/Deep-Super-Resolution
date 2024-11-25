import numpy as np
from utils.common import *
from PIL import Image

def add_gaussian_noise(image, std=1):
    image = np.clip(image + np.random.normal(scale=std*255, size=image.shape), 0, 255).astype(np.uint8)
    return image

def add_salt_pepper_noise(image, s=0.01, p=0.01):

    print(image.shape)
    
    salt = np.random.rand(image.shape[0], image.shape[1]) < s
    pepper = np.random.rand(image.shape[0], image.shape[1]) < p

    print(salt.shape)
    
    image[salt] = 255
    image[pepper] = 0
    
    return image

def downsample(image, factor=2, interpolation=Image.BICUBIC):
    return image.resize((image.width // factor, image.height // factor), resample=interpolation)
