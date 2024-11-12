import torch

from utils.INR import *
from models.INR.siren import *
from dataset import *

# Torch setup
torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def INR_ISR(net, LR_image, HR_GT):
    pass


# Get the training data loader
train_data = DIV2KDataset(LR_dir='data/DIV2K_train_LR_x8/',
                          HR_dir='data/DIV2K_train_HR/')

# initialize a SIREN model
siren = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True)
siren = siren.to(device)



# Choose an image
idx = 0
if eval:
    LR_image, HR_GT = train_data[idx]

    # Perform ISR by optimising DIP over the image
    _, net_metrics = DIP_ISR(net, LR_image, HR_GT)
else:
    LR_image, _ = train_data[idx]

    # Perform ISR by optimising DIP over the image
    resolved, _ = DIP_ISR(net, LR_image)