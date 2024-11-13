import torch

from utils.SRGAN import *
from models.INR.siren import *
from dataset import *

# Torch setup
torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")