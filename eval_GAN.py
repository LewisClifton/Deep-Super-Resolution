import torch
import os
import argparse
import sys
from datetime import datetime
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM

from models.GAN.generator import Generator
from dataset import GANDIV2KDataset
from utils.GAN import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True


def GAN_ISR_Batch_eval(gan_G, val_loader, out_dir, batch_size, device):
    # Perform SISR using GAN on a batch of images and evaluate performance

    # Initialise performance metrics
    running_psnr = 0
    running_ssim = 0
    running_lpips = 0

    # Get metrics models
    psnr = PSNR()
    ssim = SSIM(data_range=1.)
    lpips = LPIPS(net_type='alex').to(device)

    print(f'Starting GAN evaluation..')

    # Perform SISR using the generator for batch_size many images
    for idx, (LR_image, HR_image, image_name) in enumerate(val_loader): 

        HR_image = HR_image.to(device)
        LR_image = LR_image.to(device)
        image_name = image_name.squeeze(0)

        print(f"Starting on {image_name}.  ({idx}/{num_images})")

        # Perform DIP SISR for the current image
        resolved_image = gan_G(LR_image)

        # Get PSNR, SSIM, LPIPS metrics
        running_psnr += psnr(resolved_image, HR_image)
        running_ssim += ssim(resolved_image, HR_image)
        running_lpips += lpips(resolved_image, HR_image)

        print("Done.")

        resolved_image = torch_to_np(resolved_image)
        resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        save_image(resolved_image, image_name, out_dir)

        # Delete everything to ensure gpu memory is low
        del LR_image, HR_image
        del resolved_image

    
    print(f"Done for all {batch_size} images.")

    # Calculate metric averages
    eval_metrics = {
        'avg_psnr' : running_psnr / batch_size,
        'avg_ssim' : running_ssim / batch_size,
        'avg_lpips' : running_lpips / batch_size,
    }
    
    return eval_metrics

def main(rank,
         world_size, 
         LR_dir, 
         HR_dir, 
         out_dir, 
         model_path, 
         factor, 
         num_images, 
         downsample, 
         noise_type):
    
    # setup the process groups
    setup_gpu(rank, world_size)

    # Get generator
    gan_G = Generator(factor=factor).to(rank)
    gan_G = load_model(gan_G, model_path)
    gan_G = DDP(gan_G, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    dataset = GANDIV2KDataset(LR_dir=LR_dir, HR_dir=HR_dir, scale_factor=factor, num_images=num_images, noise_type=noise_type, downsample=downsample)
    batch_size = 1 # set batch size to 1 when evaluating as the images can be very large

    # Create a dataloader           
    data_loader = get_data_loader(dataset, rank, world_size, batch_size)

        # Set Generator to evaluation mode
    gan_G.eval()

    start_time = time.time()
    
    # Evaluate
    eval_metrics = GAN_ISR_Batch_eval(gan_G, data_loader, out_dir, num_images, device=rank)

    # Get run time
    eval_metrics['Eval runtime'] = time.time() - start_time

    # Wait for all gpus to get to this point
    dist.barrier()

    # Send all the gpu node metrics back to the main gpu
    torch.cuda.set_device(rank)
    eval_metrics_gpus = [None for _ in range(world_size)]
    dist.all_gather_object(eval_metrics_gpus, eval_metrics)

    if rank == 0:
        print('Done evaluating')

        # Get runtime
        runtime = np.max([gpu_metrics['Eval runtime'] for gpu_metrics in eval_metrics_gpus])
        runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

        # Average the metrics over each GPU output
        avg_psnr = np.mean([gpu_metrics['avg_psnr'] for gpu_metrics in eval_metrics_gpus])
        avg_ssim = np.mean([gpu_metrics['avg_ssim'] for gpu_metrics in eval_metrics_gpus])
        avg_lpips = np.mean([gpu_metrics['avg_lpips'] for gpu_metrics in eval_metrics_gpus])

        # Final evalutaion metric for the log
        final_eval_metrics = {
            "Number of images evaluated over" : num_images,
            "Eval runtime" : runtime,
            "Average PSNR": avg_psnr,
            "Average SSIM": avg_ssim,
            "Average LPIPS": avg_lpips,
        }

        # Save metrics log
        save_log(out_dir, **final_eval_metrics)


# Setup all the parameters for the GAN script
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for evaluation log/saved images", required=True)
    parser.add_argument('--model_path', type=str, help="Path of model to evaluate")
    parser.add_argument('--num_gpus', type=int, help='Number of gpus to run models with', default=2)
    parser.add_argument('--num_images', type=int, help='Number of images to use for evaluation', default=-1)
    parser.add_argument('--save_images', type=bool, help='Whether to save super-resolved images', default=False)
    parser.add_argument('--model_path', type=str, help='Path to trained model for evaluation', required=True)
    parser.add_argument('--noise_type', type=str, help='Type of noise to apply to LR images when evaluating. "gauss": Gaussian noise, "saltpepper": salt and pepper noise. Requires the --noise_param flag to give noise parameter')
    parser.add_argument('--noise_param', type=float, help='Parameter for noise applied to LR images when evaluating. In the range [0,1]. If --noise=gauss, noise param is the standard deviation. If --noise_type=saltpepper, noise_param is probability of applying salt or pepper noise to a pixel')
    parser.add_argument('--downsample', type=bool, help='Apply further 2x downsampling to LR images when evaluating')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    if not os.path.isdir(data_dir):
        print(f'{data_dir} not found.')
        sys.exit(1)

    if not os.path.isdir(out_dir):
        print(f'{out_dir} not found.')
        sys.exit(1)

    # Get dataset
    LR_dir = os.path.join(data_dir, 'DIV2K_validation_LR_x8/')
    HR_dir = os.path.join(data_dir, 'DIV2K_validation_HR/')

    # Output directory
    date = datetime.now()
    out_dir = os.path.join(out_dir, f'out/GAN/{date.strftime("%Y_%m_%d_%p%I_%M")}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Path of the trained model
    model_path = args.model_path

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

    # Initialise gpus
    world_size = args.num_gpus 
    mp.spawn(
        main,
        args=(world_size,
              LR_dir,
              HR_dir,
              out_dir,
              model_path,
              factor,
              num_images,
              downsample,
              noise_type),
        nprocs=world_size)