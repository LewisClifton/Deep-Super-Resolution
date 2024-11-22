import torch
import os
from datetime import datetime
import time
import argparse
import sys

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

    training_metrics = {
        'psnr' : [],
        'ssim' : []
    }

    # Define closure for training
    def closure():
        nonlocal i, net_input

         # Include regulariser noise
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

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
        if i % 1 == 0  and use_GT:
            epoch_psnr = psnr(out_HR_np, HR_image_np)
            epoch_ssim = ssim(out_HR_np, HR_image_np, channel_axis=0, data_range=1.0)
            training_metrics['psnr'].append(epoch_psnr)
            training_metrics['ssim'].append(epoch_ssim)

            if verbose:
                out_HR_np = torch_to_np(out_HR)
                print(f"Epoch {i+1}/{training_config['num_epochs']}:")
                print(f"PSNR: {epoch_psnr}")
                print(f"SSIM: {epoch_ssim}")
                print(f"Iteration runtime: {time.time() - start_time} seconds")

        i += 1
        out_HR.detach().cpu()
        out_LR.detach().cpu()
        del out_HR
        del out_LR

        return total_loss

    # Iteratively optimise over the noise 
    optimize('adam', params, closure, training_config['learning_rate'], training_config['num_epochs'])
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
    
    if use_GT:
        return resolved_image, training_metrics
    else:
        return resolved_image, None


def DIP_ISR_Batch_eval(factor, dataset, training_config, output_dir, save_resolved_images=False, num_images=5, verbose=False):
    # Use to evaluate DIP for SISR
    # Run DIP across all images while keeping track of performance metrics

    # Initialise performance metrics
    running_psnr = 0
    running_ssim = 0
    running_lpips = 0
    start_time = time.time()

    training_metrics = {
        'psnr' : np.zeros(shape=(training_config['num_epochs'])),
        'ssim' : np.zeros(shape=(training_config['num_epochs']))
    }

    # Perform SISR using DIP for num_images many images
    for idx in range(num_images):   
        
        LR_image, HR_image, image_name = dataset[idx]

        if verbose:
             print(f"Starting on {image_name} (image {idx+1}/{num_images}) for {training_config['num_epochs']} iterations. ")

        # Perform DIP SISR for the current image
        resolved_image, image_training_metrics = DIP_ISR(LR_image, HR_image, factor, training_config, use_GT=True, verbose=verbose)

        # Accumulate metrics
        resolved_image = resolved_image.to(device)
        HR_image = HR_image.to(device)
        running_lpips += lpips(resolved_image, HR_image)
        resolved_image = np.clip(resolved_image.cpu().numpy()[0], 0, 1)
        HR_image = np.clip(HR_image.cpu().numpy(), 0, 1)
    
        running_psnr += psnr(resolved_image, HR_image)
        running_ssim += ssim(resolved_image, HR_image, data_range=1, channel_axis=0)

        training_metrics['psnr'] += np.array(image_training_metrics)
        training_metrics['ssim'] += np.array(image_training_metrics)

        if verbose:
            print("Done.")

        if save_resolved_images:
            resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
            save_image(resolved_image, image_name, output_dir)

        if verbose:
            print("\n")

        del LR_image, HR_image, resolved_image

    print(f"Done for all {num_images} images.")

    # Get run time
    end_time = time.time()
    runtime = end_time - start_time

    # Calculate metric averages
    avg_psnr = running_psnr / num_images
    avg_ssim = running_ssim / num_images
    avg_lpips = running_lpips / num_images

    # Calculate training metric averages and convert to list for log saving
    training_metrics = {
        'Average PSNR per epoch' : np.divide(training_metrics['psnr'], num_images).tolist(),
        'Average SSIM per epoch' : np.divide(training_metrics['ssim'], num_images).tolist()
    }

    # Save metrics log (don't need to save model for DIP)
    save_log(num_images, runtime, avg_psnr, avg_ssim, avg_lpips, output_dir, **{**training_config, **training_metrics })


def DIP_ISR_Batch_inf(factor, dataset, training_config, output_dir, num_images, verbose=False):
    # Perform SISR using DIP for num_images many images

    # Get start time
    start_time = time.time()

    for idx in range(num_images):   

        LR_image, HR_image, image_name = dataset[idx]

        if verbose:
            print(f"Starting on {image_name} (image {idx+1}/{num_images}) for {training_config['num_epochs']} iterations. ")

        # Perform DIP SISR for the current image
        resolved_image, _ = DIP_ISR(LR_image, HR_image, factor, training_config, use_GT=False, verbose=True)

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

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--out_dir', type=str, help="Path to directory for dataset, saved images, saved models", required=True)
    parser.add_argument('--mode', type=str, help='"eval": get evaluation metrics over some images, "inf": do ISR to obtain HR images', required=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs when training (--mode=train)', default=1)
    parser.add_argument('--train_log_freq', type=int, help='How many epochs between logging metrics when training (--mode=train)', default=100)
    parser.add_argument('--save_output', type=bool, help='Whether to save output when evaluating (--model=eval)', default=False)
    parser.add_argument('--num_images', type=int, help='Number of images to use for training/evaluation', default=1)
    parser.add_argument('--noise_type', type=str, help='Type of noise to apply to LR images when evaluating (--mode=eval). "gauss": Gaussian noise, "saltpepper": salt and pepper noise. Requires the --noise_param flag to give noise parameter')
    parser.add_argument('--noise_param', type=float, help='Parameter for noise applied to LR images when evaluating (--mode=eval) In the range [0,1]. If --noise=gauss, noise param is the standard deviation. If --noise_type=saltpepper, noise_param is probability of applying salt or pepper noise to a pixel')
    parser.add_argument('--downsample', type=bool, help='Apply further 2x downsampling to LR images when evaluating (--model=eval)')
    parser.add_argument('--verbose', type=bool, help='Informative command line output during execution', default=False)
    args = parser.parse_args()

    cwd = args.out_dir

    if not os.path.exists(cwd) or not os.path.isdir(cwd):
        print(f'{cwd} not found.')
        sys.exit(1)

    # Get dataset
    LR_dir = os.path.join(cwd, 'data/DIV2K_train_LR_x8/')
    HR_dir = os.path.join(cwd, 'data/DIV2K_train_HR/') # = '' if not using HR GT for evaluation
    
    # Set the output and trained model directory
    output_dir = os.path.join(cwd, rf'out\GAN\{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}')
    trained_dir = os.path.join(cwd, r'trained\GAN')

    # Program mode i.e. 'train' for training, 'eval' for evaluation
    mode = args.mode

    # Display information during running
    verbose = args.verbose

    # Number of images from the dataset to use
    num_images = args.num_images # -1 for entire dataset, 1 for a running GAN on a single image

    if num_images < -1 or num_images == 0:
        print(f'Please provide a valid number of images to use with --num_images=-1 for entire dataset or --num_images > 0')
        sys.exit(1)

    # Super resolution scale factor
    factor = 8
    
    # Degredation
    downsample = args.downsample
    if downsample:
        factor *= 2

    # Noise
    noise_type = args.noise_type 
    if not noise_type and args.noise_param:
        print(f'Must provide noise type with --noise_type if providing noise parameter with --noise_param')
        sys.exit(1)

    if noise_type:
        if not args.noise_param:
                print(f'Must provide a noise parameter with --noise_param to use noise.')
                sys.exit(1)
        if args.noise_param < 0 or args.noise_param > 1:
            print(f'Noise parameter must be in range [0,1].')
            sys.exit(1)
            
        if noise_type == 'gauss':
            noise_type = {
                'type' : 'Gaussian',
                'std': args.noise_param,
            }
        elif noise_type == 'saltpepper':
            noise_type = {
                'type' : 'SaltAndPepper',
                's' : args.noise_param,
                'p' : args.noise_param
            }
        else:
            print(f'Noise type {args.noise_type} not supported. Use either --noise_type=gauss or --noise_type=saltpepper')
            sys.exit(1)

    # Whether to save output when evaluating
    save_output = args.save_output

    # use_GT is used as a check to get performance metrics or not which require ground truth
    use_GT = True if HR_dir else False
    if verbose: assert(use_GT), 'Verbose makes the performance metrics visible during training which require ground truths.'
    
    # Hyperparameters
    learning_rate = 0.01

    if downsample:
        num_epochs = 8000
        reg_noise_std = 0.07
    else:
        num_epochs = 2000
        reg_noise_std = 0.05

    if args.num_epochs:
        num_epochs = args.num_epochs

    # Define the training configuration using above
    training_config = {
        "learning_rate" : learning_rate,
        "num_epochs" : num_epochs,
        "reg_noise_std" : reg_noise_std
    }

    if verbose:
        print(f"Performing DIP SISR on {num_images} images.")
        print(f"Output directory: {output_dir}")

    if mode == 'eval':
        dataset = DIPDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, HR_dir=HR_dir, num_images=num_images)

        DIP_ISR_Batch_eval(factor, dataset, training_config, output_dir, save_output, num_images, verbose)
    elif mode == 'inf':
        dataset = DIPDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images)

        DIP_ISR_Batch_inf(factor, dataset, training_config, output_dir ,num_images, verbose)
    else:
        assert(False), 'Pick either DIP evaluation (eval) or DIP inference (inf) as your batch mode'