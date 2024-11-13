import torch
import torch.optim
import os

from utils.downsampler import Downsampler
from models.DIP import get_DIP_network
from dataset import DIV2KDataset
from utils.DIP import *
from utils.metrics import *
from datetime import datetime
import time

# Torch setup
torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def DIP_ISR(net, LR_image, HR_image, scale_factor, training_config, use_GT=False, verbose=False):

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
                out_HR_np = out_HR.detach().cpu().numpy()[0]
                print(f"Iteration {i}:")
                print(f"psnr: {psnr(out_HR_np, HR_image_np)}")
                print(f"ssim: {ssim(out_HR_np, HR_image_np, channel_axis=0, data_range=1.0)}")
                print(f"lpips: {lpips(out_HR, HR_image)}\n")

        i += 1

        return total_loss

    # Iteratively optimise over the noise 
    optimize(training_config['optimiser_type'], params, closure, training_config['learning_rate'], training_config['num_iter'])

    # Return the super-resolved image after training the network
    resolved_image = np.clip(torch_to_np(net(net_input)), 0, 1)

    # Return eval metrics
    if use_GT:
        net_metrics_ = {
            "psnr" : psnr(resolved_image, HR_image),
            "ssim" : ssim(resolved_image, HR_image),
            "lpips" : lpips(resolved_image, HR_image)
        }
        return resolved_image, net_metrics_
    else:
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
            print(f"Starting on {image_name}.  {idx}/{batch_size}")

        # Perform DIP SISR for the current image
        resolved_image, net_metrics = DIP_ISR(net, LR_image, HR_image, factor, training_config, use_GT=True, verbose=verbose)

        # Accumulate metrics
        running_psnr += net_metrics["psnr"]
        running_ssim += net_metrics["ssim"]
        running_lpips += net_metrics["lpips"]

        if verbose:
            print("Done.")

        if save_resolved_images:
            save_image(resolved_image, image_name, output_dir, verbose=verbose)

        if verbose:
            print("\n")

    if verbose:
        print(f"Done for all {batch_size} images.")

    # Get run time
    end_time = time.time()
    runtime = end_time - start_time

    # Calculate metric averages
    avg_psnr = running_psnr / batch_size
    avg_ssim = running_ssim / batch_size
    avg_lpips = running_lpips / batch_size
    
    # Save metrics log
    save_log(batch_size, runtime, avg_psnr, avg_ssim, avg_lpips, output_dir)

def DIP_ISR_Batch_inf(net, factor, dataset, training_config, output_dir, batch_size, verbose=False):
    # Perform SISR using DIP for batch_size many images
    for idx in range(batch_size):   
        LR_image, HR_image, image_name = dataset[idx]

        if verbose:
            print(f"Starting on {image_name}.  {idx}/{batch_size}")

        # Perform DIP SISR for the current image
        resolved_image, _ = DIP_ISR(net, LR_image, HR_image, factor, training_config)

        if verbose:
            print("Done.")

        save_image(resolved_image, image_name, output_dir)

        if verbose:
            print("\n")

    if verbose:
        print(f"Done for all {batch_size} images.")
        

def do_DIP():
    # Cap batch size at length of dataset
    batch_size = max(min(dataset), batch_size)

    # Run DIP on entire dataset if required
    if batch_size == -1:
        batch_size = len(dataset)

    if verbose:
        print(f"Performing DIP SISR on {batch_size} images.")
        print(f"Output directory: {output_dir}")

    if batch_mode == 'eval':
        DIP_ISR_Batch_eval(net, factor, dataset, training_config, output_dir, save_batch_output, batch_size, verbose)
    elif batch_mode == 'inf':
        DIP_ISR_Batch_inf(net, factor, dataset, training_config, output_dir, batch_size, verbose)
    else:
        assert(False), 'Pick either DIP evaluation (eval) or DIP inference (inf) as your batch mode'


if __name__ == '__main__':
    # Determine DIP behaviour
    batch_size = 1 # -1 for entire dataset
    batch_mode = 'inf' # 'single_eval', 'eval'
    save_batch_output = True
    verbose = True

    # Set the output directory
    output_dir = os.path.join('out/DIP/')
    output_dir = os.path.join(output_dir, datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S/"))

    # If using single image (batch_size=1)
    image_idx = 0

    # Super resolution scale factor
    factor = 4


    # IMPORTANT: Set to true if wanting to evaluate DIP and get metrics, False for SISR on unseen images without GT
    #            (use_GT = True implies that any HR images in the dataset are ground truths!!)
    use_GT = True
    
    # Get the dataset with or without GT as required
    LR_dir = 'data/DIV2K_train_LR_x8/'
    HR_dir = 'data/DIV2K_train_HR/' # = '' if not using HR GT for evaluation

    if use_GT: 
        dataset = DIV2KDataset(LR_dir=LR_dir,
                               scale_factor=factor,
                               HR_dir=HR_dir)
    else:
        dataset = DIV2KDataset(LR_dir=LR_dir,
                               scale_factor=factor,
                               HR_dir=None)
                            
    if verbose: assert(use_GT), 'Verbose makes the performance metrics visible during training which require ground truths.'

    # Get the downsampler used to optimise
    downsampler = Downsampler(n_planes=3, factor=factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

    # Hyperparameters
    learning_rate = 0.01
    num_iter = 4000
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

    do_DIP()

    
