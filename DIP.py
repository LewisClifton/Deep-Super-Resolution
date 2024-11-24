import torch
import os
from datetime import datetime
import time
import argparse
import sys
import lpips as lpips_
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from utils.downsampler import Downsampler
from models.DIP import get_DIP_network
from dataset import DIPDIV2KDataset
from utils.DIP import *
from utils.metrics import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def DIP_ISR(net, LR_image, HR_image, scale_factor, training_config, train_log_freq, device):
    # Perform DIP ISR on a single image

    # Define loss
    mse = torch.nn.MSELoss().to(device)

    # Get the downsampler used to optimise
    downsampler = Downsampler(n_planes=3, factor=scale_factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).to(device)

    # Get fixed noise for the network input
    net_input = get_noise(4, 'noise', (LR_image.size()[1]*scale_factor, LR_image.size()[2]*scale_factor)).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # Put everything on the GPU
    LR_image = LR_image.to(device).detach()
    HR_image = HR_image.to(device).detach()
    HR_image_np = HR_image.detach().cpu().numpy()[0] # For evaluation metrics

    # Optimise the network over the input
    iter = 0
    psnrs = []
    ssims = []
    
    # Define closure for training
    def closure():
        nonlocal iter, net_input

         # Include regulariser noise
        if training_metrics['reg_noise_std'] > 0:
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

        # Log evaluation metrics
        if iter % train_log_freq == 0:
            out_HR_np = torch_to_np(out_HR)
            epoch_psnr = psnr(out_HR_np, HR_image_np)
            epoch_ssim = ssim(out_HR_np, HR_image_np, channel_axis=0, data_range=1.0)
            psnrs.append(epoch_psnr)
            ssims.append(epoch_ssim)

            out_HR_np = torch_to_np(out_HR)
            print(f"Iteration {iter+1}/{training_config['num_iter']}:")
            print(f"PSNR: {epoch_psnr}")
            print(f"SSIM: {epoch_ssim}")
            print(f"Iteration runtime: {time.time() - start_time} seconds")
            
            del out_HR_np

        iter += 1
        out_HR.detach().cpu()
        out_LR.detach().cpu()
        del out_HR
        del out_LR

        return total_loss

    # Iteratively optimise over the noise 
    params = get_params('net', net, net_input)
    optimize('adam', params, closure, training_config['learning_rate'], training_config['num_iter'])

    # Get the final resolved image
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

    training_metrics = {
        'psnr' : psnrs,
        'ssim' : ssims,
    }
    
    return resolved_image, training_metrics


def main(world_size,
         rank, 
         LR_dir, 
         HR_dir, 
         output_dir, 
         factor, 
         num_images, 
         save_output,
         train_log_freq):
    
    # setup the process groups
    setup_gpu(rank, world_size)

    # Load the dataset
    dataset = DIPDIV2KDataset(LR_dir=LR_dir, HR_dir=HR_dir, scale_factor=factor, num_images=num_images)
    data_loader = get_data_loader(dataset, rank, world_size, batch_size=1)

    print(f"Performing DIP SISR on {num_images} images.")
    print(f"Output directory: {output_dir}")

    # Initialise final performance metrics averages
    running_psnr = 0
    running_ssim = 0
    running_lpips = 0

    # Initialise performance over training metrics
    metrics = {
        'avg_psnrs' : np.zeros(shape=(training_config['num_iter'])),
        'avg_ssims' : np.zeros(shape=(training_config['num_iter']))
    }

    # Get LPIPS model
    lpips_model = lpips_.LPIPS(net='alex').to(device)

    start_time = time.time()

    # Perform SISR using DIP for num_images many images
    for idx, (LR_image, HR_image, image_name) in enumerate(data_loader):   

        print(f"Starting on {image_name} (image {idx+1}/{num_images}) for {training_config['num_iter']} iterations. ")
        
        # Define DIP network
        net = get_DIP_network(input_depth=4, pad='reflection').to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        # Perform DIP SISR for the current image
        resolved_image, image_train_metrics = DIP_ISR(net, LR_image, HR_image, factor, training_config, train_log_freq, device=rank)

        # Accumulate metrics
        resolved_image = resolved_image.to(device)
        HR_image = HR_image.to(device)
        running_lpips += lpips(resolved_image, HR_image, lpips_model)
        resolved_image = np.clip(resolved_image.cpu().numpy()[0], 0, 1)
        HR_image = np.clip(HR_image.cpu().numpy(), 0, 1)
        
        # Accumulate the final metrics
        running_psnr += psnr(resolved_image, HR_image)
        running_ssim += ssim(resolved_image, HR_image, data_range=1, channel_axis=0)

        # Accumulate the metrics over iterations
        metrics['avg_psnrs'] += np.array(image_train_metrics['psnr'])
        metrics['avg_ssims'] += np.array(image_train_metrics['ssim'])

        print("Done.")

        # Save resolved image
        if save_output:
            resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
            save_image(resolved_image, image_name, output_dir)

        del LR_image, HR_image, resolved_image

    print(f"Done for all {num_images} images.")

    # Get run time
    metrics['runtime'] = time.time() - start_time

    # Get average final metrics for each resolved image
    metrics['final_psnr'] = running_psnr / num_images
    metrics['final_ssim'] = running_ssim / num_images
    metrics['final_lpips'] = running_lpips / num_images

    # Wait for all gpus to get to this point
    dist.barrier()

    # Send all the gpu node metrics back to the main gpu
    torch.cuda.set_device(rank)
    metrics_gpus = [None for _ in range(world_size)]
    dist.all_gather_object(metrics_gpus, metrics)

    if rank == 0:
        print('Done training')

        # Get runtime
        runtime = np.max([gpu_metrics['runtime'] for gpu_metrics in metrics_gpus])
        runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

        # Calculate mean across GPUs for each epoch
        avg_psnrs = [gpu_metrics['avg_psnrs'] for gpu_metrics in metrics_gpus]
        avg_ssims = [gpu_metrics['avg_ssims'] for gpu_metrics in metrics_gpus]
        avg_psnrs = np.mean(np.vstack(avg_psnrs), axis=0)
        avg_ssims = np.mean(np.vstack(avg_ssims), axis=0)

        # Calculate average final metrics across GPUs
        avg_final_psnr = np.mean([gpu_metrics['final_psnr'] for gpu_metrics in metrics_gpus])
        avg_final_ssim = np.mean([gpu_metrics['final_ssims'] for gpu_metrics in metrics_gpus])
        avg_final_lpip = np.mean([gpu_metrics['final_lpips'] for gpu_metrics in metrics_gpus])

        # Final train metric for the log
        final_metrics = {
            "Number of images evaluated over" : num_images,
            "Train runtime" : runtime,
            "Average PSNR per epoch" : avg_psnrs.tolist(),
            "Average SSIM per epoch" : avg_ssims.tolist(),
            "Average LPIPS per epoch" : 'Not tracked during training due to VGG-19 memory overhead',
            "Average final PSNR" : avg_final_psnr,
            'Average final SSIM' : avg_final_ssim,
            "Average final LPIPS" : avg_final_lpip,
        }

        # Output directory
        date = datetime.now()
        out_dir = os.path.join(out_dir, f'DIP/{date.strftime("%Y_%m_%d_%p%I_%M")}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save metrics log and model
        save_log(output_dir, **final_metrics)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for dataset, saved images, saved models", required=True)
    parser.add_argument('--num_iter', type=int, help='Number of iter when training', default=1)
    parser.add_argument('--train_log_freq', type=int, help='How many iterations between logging metrics when training', default=100)
    parser.add_argument('--save_output', type=bool, help='Whether to save super-resolved output', default=False)
    parser.add_argument('--num_images', type=int, help='Number of images to use for training/evaluation', default=1)
    parser.add_argument('--noise_type', type=str, help='Type of noise to apply to LR images when evaluating. "gauss": Gaussian noise, "saltpepper": salt and pepper noise. Requires the --noise_param flag to give noise parameter')
    parser.add_argument('--noise_param', type=float, help='Parameter for noise applied to LR images when evaluating. In the range [0,1]. If --noise=gauss, noise param is the standard deviation. If --noise_type=saltpepper, noise_param is probability of applying salt or pepper noise to a pixel')
    parser.add_argument('--downsample', type=bool, help='Apply further 2x downsampling to LR images when evaluating')
    args = parser.parse_args()

    data_dir = args.data_dir
    cwd = args.out_dir

    if not os.path.exists(cwd) or not os.path.isdir(cwd):
        print(f'{cwd} not found.')
        sys.exit(1)

    # Get dataset
    LR_dir = os.path.join(data_dir, 'data/DIV2K_train_LR_x8/')
    HR_dir = os.path.join(data_dir, 'data/DIV2K_train_HR/')
    
    # Set the output and trained model directory
    output_dir = os.path.join(cwd, rf'out\DIP\{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}')

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

    # Hyperparameters
    learning_rate = 0.01

    if downsample:
        reg_noise_std = 0.07
    else:
        reg_noise_std = 0.05

    # Number of iterations when training
    num_iter = args.num_iter

    # How many iterations between saving metrics when training
    train_log_freq = args.train_log_freq

    # Define the training configuration using above
    training_config = {
        "learning_rate" : learning_rate,
        "num_iter" : num_iter,
        "reg_noise_std" : reg_noise_std
    }

    # Initialise gpus
    world_size = args.num_gpus 
    mp.spawn(
        main,
        args=(world_size, 
              LR_dir, 
              HR_dir, 
              output_dir, 
              factor, 
              num_images, 
              save_output,
              train_log_freq),
        nprocs=world_size)