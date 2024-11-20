import torch
import numpy as np
from PIL import Image
import numpy as np
from datetime import datetime
import os

# create folder with time as name

def save_image(image, image_name, output_dir, verbose=False):

    # image = torch_to_np(image)
    image_pil = Image.fromarray(image)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = os.path.join(output_dir, f'{image_name}_resolved.png')

    image_pil.save(path)

    if verbose:
        print(f"Saved to {path}")

def save_log(num_images, runtime, avg_psnr, avg_ssim, avg_lpips, output_dir, **kwargs):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = os.path.join(output_dir, f'{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}_log.txt')

    with open(path, 'w') as f:
        f.write(f"Time of log file generation: {str(datetime.now())}\n")
        
        f.write(f"Performance metrics:\n")
        f.write(f"Number of images: {str(num_images)}\n")
        f.write(f"Time to run: {str(runtime)}\n")
        f.write(f"Average PSNR: {str(avg_psnr)}\n")
        f.write(f"Average SSIM: {str(avg_ssim)}\n")
        f.write(f"Average LPIPS: {str(avg_lpips)}\n")
        
        if kwargs:
            for key, value in kwargs.items():
                f.write(f"{key}: {str(value)}\n")

    print(f"Log file saved to {path}")

def save_model(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = os.path.join(output_dir, f'{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}.pth')
    torch.save(model.state_dict(), path)

    print(f'Model saved to {path}')

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