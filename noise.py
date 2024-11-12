import numpy as np
from utils.common import *

def add_gaussian_noise(image, mean=0, std=1):
    return np.clip(image + np.random.normal(loc=mean, scale=std, size=image.shape), 0, 1).astype(np.float32)