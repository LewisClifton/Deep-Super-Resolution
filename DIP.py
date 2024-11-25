import torch
import os
from datetime import datetime
import time
import argparse
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from utils.downsampler import Downsampler
from models.DIP import get_net
from dataset import DIPDIV2KDataset
from utils.DIP import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True


def DIP_ISR(net, LR_image, HR_image, scale_factor, training_config, train_log_freq, psnr, ssim, lpips, device):
    # Perform DIP ISR on a single image

    # Define loss
    mse = torch.nn.MSELoss()

    # Get the downsampler used to optimise
    downsampler = Downsampler(n_planes=3, factor=scale_factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).to(device)

    # Get fixed noise for the network input
    net_input = get_noise(3, 'noise', (HR_image.shape[2], HR_image.shape[3])).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # Put everything on the GPU
    LR_image = LR_image.to(device).detach()
    HR_image = HR_image.to(device).detach()

    # Optimise the network over the input
    iter = 0
    psnrs = []
    ssims = []
    lpipss = []

    # Define closure for training
    def closure():
        nonlocal iter, net_input

         # Include regulariser noise
        if training_config['reg_noise_std'] > 0:
            net_input = net_input_saved + (noise.normal_() * training_config['reg_noise_std'])
        
        # Get iteration start time
        start_time = time.time()

        net_input = net_input.to(device)

        # Get model output
        out_HR = net.module(net_input)

        out_LR = downsampler(out_HR)

        # Calculate loss
        total_loss = mse(out_LR, LR_image) 
        
        # Backpropagate loss
        total_loss.backward()

        # Log evaluation metrics
        if iter % train_log_freq == 0:
            
            epoch_psnr = psnr(out_HR, HR_image).item()
            epoch_ssim = ssim(out_HR, HR_image).item()
            epoch_lpips = lpips(out_HR, HR_image).item()

            psnrs.append(epoch_psnr)
            ssims.append(epoch_ssim)
            lpipss.append(epoch_lpips)

            if device == 0:
                print(f"Iteration {iter+1}/{training_config['num_iter']}:")
                print(f"PSNR: {epoch_psnr}")
                print(f"SSIM: {epoch_ssim}")
                print(f'LPIPS: {epoch_lpips}')
                print(f"Iteration runtime: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))} seconds")
            
            del epoch_psnr, epoch_ssim, epoch_lpips

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
    resolved_image = net(net_input).detach()
    
    # Delete everything to ensure GPU memory is freed up
    net_input.detach().cpu()
    LR_image.detach().cpu()
    HR_image.detach().cpu()
    downsampler.cpu()
    net.cpu()

    del net_input
    del LR_image, HR_image
    del net, mse, psnr, ssim, lpips
    del downsampler
    torch.cuda.empty_cache()

    training_metrics = {
        'psnrs' : psnrs,
        'ssims' : ssims,
        'lpipss' : lpipss
    }
    
    return resolved_image, training_metrics


def main(rank,
         world_size, 
         LR_dir, 
         HR_dir, 
         out_dir, 
         factor, 
         num_images,
         training_config, 
         save_output,
         train_log_freq):
    
    # setup the process groups
    setup_gpu(rank, world_size)

    # Load the dataset
    dataset = DIPDIV2KDataset(LR_dir=LR_dir, HR_dir=HR_dir, scale_factor=factor, num_images=num_images)
    data_loader = get_data_loader(dataset, rank, world_size, batch_size=1)

    if rank == 0:
        print(f"Performing DIP SISR on {num_images} images.")
        print(f"Output directory: {out_dir}")

    # Initialise final performance metrics averages
    running_psnr = 0
    running_ssim = 0
    running_lpips = 0

    # Initialise performance over training metrics
    metrics = {
        'psnrs' : np.zeros(shape=(training_config['num_iter'] // train_log_freq)),
        'ssims' : np.zeros(shape=(training_config['num_iter'] // train_log_freq)),
        'lpipss' : np.zeros(shape=(training_config['num_iter'] // train_log_freq))
    }

    # Get metrics models
    psnr = PSNR().to(rank)
    ssim = SSIM(data_range=1.).to(rank)
    lpips = LPIPS(net_type='alex').to(rank)

    start_time = time.time()

    # Perform SISR using DIP for num_images many images
    for idx, (LR_image, HR_image, image_name) in enumerate(data_loader): 
        image_name = image_name[0]  

        if rank == 0:
            print(f"Starting on {image_name} (image {idx+1}/{num_images}) for {training_config['num_iter']} iterations. ")
        
        # Define DIP network
        net = get_net(3, 'skip', 'reflection',
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').to(rank)
        net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        # Perform DIP SISR for the current image
        resolved_image, image_train_metrics = DIP_ISR(net, LR_image, HR_image, factor, training_config, train_log_freq, psnr=psnr, ssim=ssim, lpips=lpips, device=rank)

        # Accumulate running lpips
        HR_image = HR_image.to(rank)
        
        # Accumulate running psnr, ssim, lpips
        running_psnr += psnr(resolved_image, HR_image).item()
        running_ssim += ssim(resolved_image, HR_image).item()
        running_lpips += lpips(resolved_image, HR_image).item()
        
        # Accumulate the metrics over iterations
        metrics['psnrs'] += np.array(image_train_metrics['psnrs'])
        metrics['ssims'] += np.array(image_train_metrics['ssims'])
        metrics['lpipss'] += np.array(image_train_metrics['lpipss'])

        # Save resolved image
        if save_output and rank == 0:
            print("Done.")

            resolved_image = torch_to_np(resolved_image)
            resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
            save_image(resolved_image, image_name, out_dir)

        del LR_image, HR_image, resolved_image, net

    if rank == 0:
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

        # Get runtime
        runtime = np.max([gpu_metrics['runtime'] for gpu_metrics in metrics_gpus])
        runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

        # Calculate mean across GPUs for each epoch
        avg_psnrs = [gpu_metrics['psnrs'] for gpu_metrics in metrics_gpus]
        avg_ssims = [gpu_metrics['ssims'] for gpu_metrics in metrics_gpus]
        avg_lpipss = [gpu_metrics['lpipss'] for gpu_metrics in metrics_gpus]
        avg_psnrs = np.mean(np.vstack(avg_psnrs), axis=0)
        avg_ssims = np.mean(np.vstack(avg_ssims), axis=0)
        avg_lpipss = np.mean(np.vstack(avg_lpipss), axis=0)

        # Calculate average final metrics across GPUs
        avg_final_psnr = np.mean([gpu_metrics['final_psnr'] for gpu_metrics in metrics_gpus])
        avg_final_ssim = np.mean([gpu_metrics['final_ssim'] for gpu_metrics in metrics_gpus])
        avg_final_lpip = np.mean([gpu_metrics['final_lpips'] for gpu_metrics in metrics_gpus])

        # Final train metric for the log
        final_metrics = {
            "Number of images evaluated over" : num_images,
            "Train runtime" : runtime,
            "Average PSNR per epoch" : avg_psnrs.tolist(),
            "Average SSIM per epoch" : avg_ssims.tolist(),
            "Average LPIPS per epoch" : avg_lpipss.tolist(),
            "Average final PSNR" : avg_final_psnr,
            'Average final SSIM' : avg_final_ssim,
            "Average final LPIPS" : avg_final_lpip,
        }

        # Save metrics log and model
        save_log(out_dir, **final_metrics)

    dist.barrier()
    if dist.is_initialized():
            dist.destroy_process_group()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for dataset, saved images, saved models", required=True)
    parser.add_argument('--num_iter', type=int, help='Number of iter when training', default=1)
    parser.add_argument('--num_gpus', type=int, help='Number of gpus to run models with', default=2)
    parser.add_argument('--train_log_freq', type=int, help='How many iterations between logging metrics when training', default=100)
    parser.add_argument('--save_output', type=bool, help='Whether to save super-resolved output', default=False)
    parser.add_argument('--num_images', type=int, help='Number of images to use for training/evaluation', default=1)
    parser.add_argument('--noise_type', type=str, help='Type of noise to apply to LR images when evaluating. "gauss": Gaussian noise, "saltpepper": salt and pepper noise. Requires the --noise_param flag to give noise parameter')
    parser.add_argument('--noise_param', type=float, help='Parameter for noise applied to LR images when evaluating. In the range [0,1]. If --noise=gauss, noise param is the standard deviation. If --noise_type=saltpepper, noise_param is probability of applying salt or pepper noise to a pixel')
    parser.add_argument('--downsample', type=bool, help='Apply further 2x downsampling to LR images when evaluating')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        print(f'{out_dir} not found.')
        sys.exit(1)

    # Get dataset
    LR_dir = os.path.join(data_dir, 'DIV2K_train_LR_x8/')
    HR_dir = os.path.join(data_dir, 'DIV2K_train_HR/')
    
    # Set the output and trained model directory
    out_dir = os.path.join(out_dir, rf'out/DIP/{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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
              out_dir, 
              factor, 
              num_images, 
              training_config,
              save_output,
              train_log_freq),
        nprocs=world_size)