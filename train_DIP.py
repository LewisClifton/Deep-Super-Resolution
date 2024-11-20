import torch
import os
from datetime import datetime
import time

from utils.downsampler import Downsampler
from models.DIP import get_DIP_network
from dataset import DIV2KDataset
from utils.DIP import *
from utils.metrics import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def DIP_ISR(net, LR_image, HR_image, scale_factor, training_config, use_GT=False, verbose=False):
    # Perform DIP ISR on a single image

    # Get fixed noise for the network input
    net_input = get_noise(32, 'noise', (LR_image.size()[1]*scale_factor, LR_image.size()[2]*scale_factor)).type(dtype).detach()
    noise = net_input.detach().clone()

    # Include regulariser noise
    if reg_noise_std > 0:
        net_input = noise + (noise.normal_() * reg_noise_std)

    # Put everything on the GPU
    net_input = net_input.to(device)
    LR_image = LR_image.to(device).unsqueeze(0)
    HR_image = HR_image.to(device).unsqueeze(0)
    HR_image_np = HR_image.detach().cpu().numpy()[0] # For evaluation metrics

    # Optimise the network over the input
    i = 0
    params = get_params('net', net, net_input)

    # Define closure for training
    def closure():
        nonlocal i

        print(i)

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
                print(f"Iteration {i+1}/{training_config['num_iter']}:")
                print(f"PSNR: {psnr(out_HR_np, HR_image_np)}")
                print(f"SSIM: {ssim(out_HR_np, HR_image_np, channel_axis=0, data_range=1.0)}")
                print(f"LPIPS: {lpips(out_HR, HR_image)}")
                print(f"Iteration runtime: {time.time() - start_time} seconds")

        i += 1

        return total_loss

    # Iteratively optimise over the noise 
    optimize(training_config['optimiser_type'], params, closure, training_config['learning_rate'], training_config['num_iter'])

    resolved_image = net(net_input)
    
    return resolved_image


def DIP_ISR_Batch_eval(net, factor, dataset, training_config, output_dir, save_resolved_images=False, batch_size=5, verbose=False):
    # Use to evaluate DIP for SISR
    # Run DIP across all images while keeping track of performance metrics

    # Initialise performance metrics
    running_psnr = 0
    running_ssim = 0
    running_lpips = 0
    start_time = time.time()

    # Perform SISR using DIP for batch_size many images
    for idx in range(batch_size):   
        LR_image, HR_image, image_name = dataset[idx]

        if verbose:
             print(f"Starting on {image_name} (image {idx+1}/{batch_size}) for {training_config['num_iter']} iterations. ")

        # Perform DIP SISR for the current image
        resolved_image = DIP_ISR(net, LR_image, HR_image, factor, training_config, use_GT=True, verbose=verbose)
        
       # Accumulate metrics
        HR_image = HR_image.to(device)
        running_lpips += lpips(resolved_image, HR_image)
       
        resolved_image = np.clip(torch_to_np(resolved_image), 0, 1)
        HR_image = np.clip( HR_image.detach().cpu().numpy(), 0, 1)

        running_psnr += psnr(resolved_image, HR_image)
        running_ssim += ssim(resolved_image, HR_image, data_range=1, channel_axis=0)

        if verbose:
            print("Done.")

        if save_resolved_images:
            resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
            save_image(resolved_image, image_name, output_dir)

        if verbose:
            print("\n")

    
    print(f"Done for all {batch_size} images.")

    # Get run time
    end_time = time.time()
    runtime = end_time - start_time

    # Calculate metric averages
    avg_psnr = running_psnr / batch_size
    avg_ssim = running_ssim / batch_size
    avg_lpips = running_lpips / batch_size
    
    # Save metrics log
    save_log(batch_size, runtime, avg_psnr, avg_ssim, avg_lpips, output_dir, **{'Iterations per image' : training_config['num_iter']})


def DIP_ISR_Batch_inf(net, factor, dataset, training_config, output_dir, batch_size, verbose=False):
    # Perform SISR using DIP for batch_size many images

    # Get start time
    start_time = time.time()

    for idx in range(batch_size):   
        LR_image, HR_image, image_name = dataset[idx]

        if verbose:
            print(f"Starting on {image_name} (image {idx+1}/{batch_size}) for {training_config['num_iter']} iterations. ")

        # Perform DIP SISR for the current image
        resolved_image = DIP_ISR(net, LR_image, HR_image, factor, training_config, use_GT=False, verbose=True)

        if verbose:
            print("Done.")

        resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        save_image(resolved_image, image_name, output_dir)

        if verbose:
            print("\n")

    
    print(f"Done for all {batch_size} images.")
    

    # Get run time
    runtime = time.time() - start_time
    
    # Save metrics log
    save_log(batch_size, runtime, 'N/a', 'N/a', 'N/a', output_dir, **{'Iterations per image' : training_config['num_iter']})


    
if __name__ == '__main__':
    # Determine program behaviour
    verbose = True
    batch_mode = 'eval' # 'eval'
    batch_size = 1 # -1 for entire dataset

    # DIP evaluation settings
    save_batch_output = True

    # Set the output directory
    output_dir = os.path.join(os.getcwd(), rf'out\DIP\{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}')
    
    # If using single image (batch_size=1)
    image_idx = 0

    # Super resolution scale factor
    factor = 4

    # Get the dataset with or without GT as required
    LR_dir = 'data/DIV2K_train_LR_x8/'
    HR_dir = 'data/DIV2K_train_HR/' # = None if not using HR GT for evaluation
    dataset = DIV2KDataset(LR_dir=LR_dir, scale_factor=factor, HR_dir=HR_dir)

    # use_GT is used as a check to get performance metrics or not which require ground truth
    use_GT = True if HR_dir else False
    if verbose: assert(use_GT), 'Verbose makes the performance metrics visible during training which require ground truths.'

    # Get the downsampler used to optimise
    downsampler = Downsampler(n_planes=3, factor=factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

    # Hyperparameters
    learning_rate = 0.01
    num_iter = 1
    reg_noise_std = 0.05
    optimiser_type = 'adam'

    # Define the training configuration using above
    training_config = {
        "learning_rate" : learning_rate,
        "num_iter" : num_iter,
        "reg_noise_std" : reg_noise_std,
        "optimiser_type" : optimiser_type
    }

    # Define loss
    mse = torch.nn.MSELoss().type(dtype)

    # Define DIP network
    net = get_DIP_network(input_depth=32, pad='reflection').to(device)

    # Cap batch size at length of dataset
    batch_size = min(len(dataset), batch_size)

    # Run DIP on entire dataset if required
    if batch_size == -1:
        batch_size = len(dataset)

    if verbose:
        print(f"Performing DIP SISR on {batch_size} images.")
        print(f"Output directory: {output_dir}")


    if batch_mode == 'eval':
        DIP_ISR_Batch_eval(net, factor, dataset, training_config, output_dir, save_batch_output, batch_size, verbose)
    elif batch_mode == 'inf':
        DIP_ISR_Batch_inf(net, factor, dataset, training_config, output_dir ,batch_size, verbose)
    else:
        assert(False), 'Pick either DIP evaluation (eval) or DIP inference (inf) as your batch mode'