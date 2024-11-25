import torch
import os
import argparse
import sys
from datetime import datetime
import time
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
import torch.nn.functional as F

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
    psnr = PSNR().to(device)
    ssim = SSIM(data_range=1.).to(device)
    lpips = LPIPS(net_type='alex').to(device)

    # Perform SISR using the generator for batch_size many images
    for _, (LR_image, HR_image, image_name) in enumerate(val_loader): 

        HR_image = HR_image.to(device)
        LR_image = LR_image.to(device)
        image_name = image_name[0] 

        print(f"Starting on {image_name}.")

        # Perform DIP SISR for the current image
        resolved_image = gan_G(LR_image)

        # Get PSNR, SSIM, LPIPS metrics
        running_psnr += psnr(resolved_image, HR_image).item()
        running_ssim += ssim(resolved_image, HR_image).item()
        running_lpips += lpips(resolved_image, HR_image).item()

        print(f"Done evaluating over {image_name}.")

        resolved_image = torch_to_np(resolved_image)
        resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        save_image(resolved_image, image_name, out_dir)
        print()

        # Delete everything to ensure gpu memory is low
        del LR_image, HR_image
        del resolved_image

    # Calculate metric averages
    eval_metrics = {
        'avg_psnr' : running_psnr / batch_size,
        'avg_ssim' : running_ssim / batch_size,
        'avg_lpips' : running_lpips / batch_size,
    }
    
    return eval_metrics

def main(LR_dir, 
         HR_dir, 
         out_dir, 
         model_path, 
         factor, 
         num_images, 
         noise_type,
         device):

    print(f'Starting GAN evaluation..')
    print()

    # Get generator
    gan_G = Generator(factor=factor).to(device)
    gan_G = load_model(gan_G, model_path)

    dataset = GANDIV2KDataset(LR_dir=LR_dir, HR_dir=HR_dir, scale_factor=factor, num_images=num_images, noise_type=noise_type, downsample=downsample)
    batch_size = 1 # set batch size to 1 when evaluating as the images can be very large

    # Create a dataloader           
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # Set Generator to evaluation mode
    gan_G.eval()

    start_time = time.time()
    
    # Evaluate
    
    eval_metrics = GAN_ISR_Batch_eval(gan_G, data_loader, out_dir, num_images, device=device)

    # Get run time
    runtime = time.time() - start_time
    runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

    print(f'Done evaluating for all {num_images} images.')

    # Final evalutaion metric for the log
    eval_metrics["Number of images evaluated over"] = num_images
    eval_metrics["Eval runtime"] = runtime

    # Save metrics log and model
    if noise_type is None:
        save_log(out_dir, **eval_metrics)
    else:
        save_log(out_dir, **eval_metrics, **noise_type)


# Setup all the parameters for the GAN script
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for evaluation log/saved images", required=True)
    parser.add_argument('--model_path', type=str, help="Path of model to evaluate", required=True)
    parser.add_argument('--num_images', type=int, help='Number of images to use for evaluation', default=-1)
    parser.add_argument('--save_images', type=bool, help='Whether to save super-resolved images', default=False)
    parser.add_argument('--noise_type', type=str, help='Type of noise to apply to LR images when evaluating. "gauss": Gaussian noise, "saltpepper": salt and pepper noise. Requires the --noise_param flag to give noise parameter')
    parser.add_argument('--noise_param', type=float, help='Parameter for noise applied to LR images when evaluating. In the range [0,1]. If --noise=gauss, noise param is the standard deviation. If --noise_type=saltpepper, noise_param is probability of applying salt or pepper noise to a pixel')
    parser.add_argument('--factor', type=bool, help='If evaluating a 8x GAN or 16x', default=8)
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
    LR_dir = os.path.join(data_dir, 'DIV2K_valid_LR_x8/')
    HR_dir = os.path.join(data_dir, 'DIV2K_valid_HR/')

    # Super resolution scale factor
    factor = args.factor

    # Output directory
    date = datetime.now()
    out_dir = os.path.join(out_dir, f'out/GANx{factor}/{date.strftime("%Y_%m_%d_%p%I_%M")}')
    os.makedirs(out_dir, exist_ok=True)

    # Path of the trained model
    model_path = args.model_path

    # Number of images from the dataset to use
    num_images = args.num_images # -1 for entire dataset, 1 for a running GAN on a single image

    if num_images < -1 or num_images == 0:
        print(f'Please provide a valid number of images to use with --num_images=-1 for entire dataset or --num_images > 0')
        sys.exit(1)


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

    main(LR_dir,
        HR_dir,
        out_dir,
        model_path,
        factor,
        num_images,
        downsample,
        noise_type)