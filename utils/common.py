import re
from typing import OrderedDict
import torch
import numpy as np
from PIL import Image
import numpy as np
from datetime import datetime
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# Setup for multi-gpu loading
def setup_gpu(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Get the dataloaders
def get_data_loader(dataset, rank, world_size, batch_size=32):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

def save_image(image, image_name, out_dir):

    # image = torch_to_np(image)
    image_pil = Image.fromarray(image)

    out_dir = os.path.join(out_dir, 'images/')

    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f'{image_name}_resolved.png')

    image_pil.save(path)

    print(f"Saved to {path}")

def save_log(out_dir, **kwargs):

    path = os.path.join(out_dir, f'{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}_log.txt')
    with open(path, 'w') as f:        
        if kwargs:
            for key, value in kwargs.items():
                f.write(f"{key}: {str(value)}\n")

    print(f"Log file saved to {path}")

def save_model(model, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    path = os.path.join(out_dir, f'{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}.pth')
    torch.save(model.state_dict(), path)

    print(f'Model saved to {path}')

# Load model
def load_model(model, model_path):
    
    model_state_dict = torch.load(model_path, weights_only=True)

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in model_state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = model_state_dict

    model.load_state_dict(model_dict)

    return model

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def lpips(im0, im1, lpips_model):
    with torch.no_grad():
        loss =  lpips_model.forward(im0,im1).item()
        del lpips_model
        return loss