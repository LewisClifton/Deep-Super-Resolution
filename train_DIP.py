import torch
import os
from datetime import datetime
import time

from utils.downsampler import Downsampler
from models.DIP import get_DIP_network
from dataset import DIPDIV2KDataset
from utils.DIP import *
from utils.metrics import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def DIP_ISR(LR_image, HR_image, scale_factor, training_config, use_GT=False, verbose=False, reg_noise_std=0.05):
    # Perform DIP ISR on a single image

    # Define DIP network
    net = get_DIP_network(input_depth=4, pad='reflection').to(device)

    # Define loss
    mse = torch.nn.MSELoss().to(device)

    # Get the downsampler used to optimise
    downsampler = Downsampler(n_planes=3, factor=scale_factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

    # Get fixed noise for the network input
    net_input = get_noise(4, 'noise', (LR_image.size()[1]*scale_factor, LR_image.size()[2]*scale_factor)).type(dtype).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # Put everything on the GPU
    #net_input = np_to_torch(net_input).squeeze(0).to(device)
    LR_image = LR_image.to(device).unsqueeze(0).detach()
    HR_image = HR_image.to(device).unsqueeze(0).detach()
    HR_image_np = HR_image.detach().cpu().numpy()[0] # For evaluation metrics

    # Optimise the network over the input
    i = 0
    params = get_params('net', net, net_input)

    # Define closure for training
    def closure():
        nonlocal i, net_input

         # Include regulariser noise
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)


        print(f'Iteration {i}: {torch.cuda.memory_allocated()}')

        # Get iteration start time
        start_time = time.time()

        # Get model output
        out_HR = net(net_input)

        out_LR = downsampler(out_HR)

        # Calculate loss
        total_loss = mse(out_LR, LR_image) 
        
        # Backpropagate loss
        total_loss.backward()

        # Print evaluation metrics if required
        if verbose and use_GT:
            if i % 200 == 0 :
                out_HR_np = torch_to_np(out_HR)
                print(f"Epoch {i+1}/{training_config['num_epochs']}:")
                print(f"PSNR: {psnr(out_HR_np, HR_image_np)}")
                print(f"SSIM: {ssim(out_HR_np, HR_image_np, channel_axis=0, data_range=1.0)}")
                print(f"LPIPS: N/A as it uses too much GPU memory")
                print(f"Iteration runtime: {time.time() - start_time} seconds")

        i += 1
        out_HR.detach().cpu()
        out_LR.detach().cpu()
        del out_HR
        del out_LR

        return total_loss

    # Iteratively optimise over the noise 
    optimize(training_config['optimiser_type'], params, closure, training_config['learning_rate'], training_config['num_epochs'])
    resolved_image = net(net_input).detach().cpu()
    
    # Delete everything to ensure GPU memory is freed up
    net_input.detach().cpu()
    LR_image.detach().cpu()
    HR_image.detach().cpu()
    downsampler.cpu()
    net.cpu()

    del net_input
    del LR_image, HR_image, HR_image_np
    del net, mse
    del downsampler
    torch.cuda.empty_cache()
    
    return resolved_image


def DIP_ISR_Batch_eval(factor, dataset, training_config, output_dir, save_resolved_images=False, num_images=5, verbose=False):
    # Use to evaluate DIP for SISR
    # Run DIP across all images while keeping track of performance metrics

    # Initialise performance metrics
    running_psnr = 0
    running_ssim = 0
    running_lpips = 0
    start_time = time.time()

    # Perform SISR using DIP for num_images many images
    for idx in range(num_images):   
        
        LR_image, HR_image, image_name = dataset[idx]

        if verbose:
             print(f"Starting on {image_name} (image {idx+1}/{num_images}) for {training_config['num_epochs']} iterations. ")

        # Perform DIP SISR for the current image
        resolved_image = DIP_ISR(LR_image, HR_image, factor, training_config, use_GT=True, verbose=verbose)

        # Accumulate metrics
        resolved_image = resolved_image.to(device)
        HR_image = HR_image.to(device)
        running_lpips += lpips(resolved_image, HR_image)
        resolved_image = np.clip(resolved_image.cpu().numpy()[0], 0, 1)
        HR_image = np.clip(HR_image.cpu().numpy(), 0, 1)
    
        running_psnr += psnr(resolved_image, HR_image)
        running_ssim += ssim(resolved_image, HR_image, data_range=1, channel_axis=0)

        if verbose:
            print("Done.")

        if save_resolved_images:
            resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
            save_image(resolved_image, image_name, output_dir)

        if verbose:
            print("\n")

    print(f"Done for all {num_images} images.")

    # Get run time
    end_time = time.time()
    runtime = end_time - start_time

    # Calculate metric averages
    avg_psnr = running_psnr / num_images
    avg_ssim = running_ssim / num_images
    avg_lpips = running_lpips / num_images
    
    # Save metrics log
    save_log(num_images, runtime, avg_psnr, avg_ssim, avg_lpips, output_dir, **{'Iterations per image' : training_config['num_epochs']})


def DIP_ISR_Batch_inf(factor, dataset, training_config, output_dir, num_images, verbose=False):
    # Perform SISR using DIP for num_images many images

    # Get start time
    start_time = time.time()

    for idx in range(num_images):   

        LR_image, HR_image, image_name = dataset[idx]

        if verbose:
            print(f"Starting on {image_name} (image {idx+1}/{num_images}) for {training_config['num_epochs']} iterations. ")

        # Perform DIP SISR for the current image
        resolved_image = DIP_ISR(LR_image, HR_image, factor, training_config, use_GT=False, verbose=True)

        if verbose:
            print("Done.")

        resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        save_image(resolved_image, image_name, output_dir)

        if verbose:
            print("\n")

    
    print(f"Done for all {num_images} images.")
    
    # Get run time
    runtime = time.time() - start_time
    
    # Save metrics log
    save_log(num_images, runtime, 'N/a', 'N/a', 'N/a', output_dir, **{'Iterations per image' : training_config['num_epochs']})


    
if __name__ == '__main__':

    # Determine program behaviour
    verbose = True
    batch_mode = 'eval' # 'inf'
    num_images = 1 # -1 for entire dataset, 1 for a running DIP on a single image

    # DIP evaluation settings
    save_batch_output = True

    # Set the output directory
    output_dir = os.path.join(os.getcwd(), rf'out\DIP\{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}')

    # Super resolution scale factor (excluding additional 2x downsampling)
    factor = 4

    # Get the dataset with or without GT as required
    LR_dir = 'data/DIV2K_train_LR_x8/'
    HR_dir = 'data/DIV2K_train_HR/'

    # use_GT is used as a check to get performance metrics or not which require ground truth
    use_GT = True if HR_dir else False
    if verbose: assert(use_GT), 'Verbose makes the performance metrics visible during training which require ground truths.'
    
    # Degredation
    downsample = False
    if downsample: factor *= 2
    noise_type = {
        'type' : 'Gaussian',
        'std': 200,
    }
    noise_type = {
        'type' : 'SaltAndPepper',
        's' : 0.01,
        'p' : 0.01
    }
    # noise_type = None

    # Hyperparameters
    learning_rate = 0.01
    reg_noise_std = 0.05
    optimiser_type = 'adam'

    if downsample:
        num_epochs = 8000
        reg_noise_std = 0.07
    else:
        num_epochs = 2000
        reg_noise_std = 0.05

    # Define the training configuration using above
    training_config = {
        "learning_rate" : learning_rate,
        "num_epochs" : num_epochs,
        "reg_noise_std" : reg_noise_std,
        "optimiser_type" : optimiser_type
    }

    if verbose:
        print(f"Performing DIP SISR on {num_images} images.")
        print(f"Output directory: {output_dir}")

    if batch_mode == 'eval':
        dataset = DIPDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, HR_dir=HR_dir, num_images=num_images)

        DIP_ISR_Batch_eval(factor, dataset, training_config, output_dir, save_batch_output, num_images, verbose)
    elif batch_mode == 'inf':
        dataset = DIPDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images)

        DIP_ISR_Batch_inf(factor, dataset, training_config, output_dir ,num_images, verbose)
    else:
        assert(False), 'Pick either DIP evaluation (eval) or DIP inference (inf) as your batch mode'