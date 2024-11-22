import torch
import os
import argparse
import sys
from torch.utils.data import DataLoader
from datetime import datetime
import time

from models.GAN.discriminator import Discriminator
from models.GAN.generator import Generator
from dataset import GANDIV2KDataset
from utils.SRGAN import *
from utils.metrics import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generator loss
def get_adversarial_loss(fake_output, bce_loss):
    adversarial_loss = bce_loss(fake_output, torch.ones_like(fake_output))
    return adversarial_loss

# Discriminator loss
def get_loss_D(real_output, fake_output, bce_loss):
    real_loss = bce_loss(real_output, torch.ones_like(real_output))
    fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))
    loss_D = real_loss + fake_loss
    return loss_D

# Get generator training loss function
class PerceptualLoss():
    def __init__(self, content_loss_fn):
        self.content_loss_fn = content_loss_fn

    def __call__(self, fake_output_G, HR_images, fake_output_D, bce_loss):
        # Content less: MSE loss or vgg loss
        self.content_loss_fn.to(device)
        content_loss = self.content_loss_fn(fake_output_G, HR_images)
        self.content_loss_fn.cpu() # keep on cpu until needed for loss calculation

        # Adversarial loss
        bce_loss.to(device)
        adversarial_loss_ = get_adversarial_loss(fake_output_D, bce_loss)
        bce_loss.cpu()

        # Perceptual loss
        perceptual_loss = content_loss + adversarial_loss_

        return perceptual_loss


def GAN_ISR_train(gan_G, gan_D, train_loader, output_dir, num_epoch=5, verbose=False):
    # Train GAN to perform SISR

    #print(f"Start of training: {torch.cuda.memory_allocated() / (1024. ** 3)}GB")
    # Get loss functions
    bce_loss = nn.BCELoss()#.to(device)
    vgg_loss = Vgg19Loss()#.to(device)
    # mse_loss = nn.MSELoss().to(device)
    perceptualLoss = PerceptualLoss(vgg_loss) # PerceptualLoss(mse)
    
    optim_G = torch.optim.Adam(gan_G.parameters(), lr=1e-4)
    optim_D = torch.optim.Adam(gan_D.parameters(), lr=1e-4)
    
    def do_epoch(LR_patches, HR_patches):

        # print(f"Before LR, HR on gpu: {torch.cuda.memory_allocated() / (1024. ** 3)}GB")
        LR_patches = LR_patches.to(device)
        HR_patches = HR_patches.to(device)

        #print(f"Before train D: {torch.cuda.memory_allocated() / (1024. ** 3)}GB")
        # Train Discriminator
        real_output_D = gan_D(HR_patches) # Discriminator output for real HR images
        
        fake_output_G = gan_G(LR_patches) # Generator output for LR images
        fake_output_D = gan_D(fake_output_G.detach()) # Discriminator output for fake HR images (i.e. generated from LR by generator)
        loss_D = get_loss_D(real_output_D, fake_output_D, bce_loss)

        # Update Discriminator
        gan_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        #print(f"After train D: {torch.cuda.memory_allocated() / (1024. ** 3)}GB")

        # Train Generator
        fake_output_G = gan_G(LR_patches)
        fake_output_D = gan_D(fake_output_G.detach())
        loss_G = perceptualLoss(fake_output_G, HR_patches, fake_output_D, bce_loss)

        # Update Generator
        gan_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        #print(f"After train G: {torch.cuda.memory_allocated() / (1024. ** 3)}GB")

        del fake_output_D, fake_output_G
        del real_output_D
        del LR_patches, HR_patches

        #print(f"After del: {torch.cuda.memory_allocated() / (1024. ** 3)}GB")

        return loss_D, loss_G
    
    if verbose:
        print(f'Starting GAN training..')

    train_start_time = time.time()


    avg_psnr = []
    avg_ssim =[]

    for epoch in range(num_epoch):

        # Get iteration start time
        start_time = time.time()

        # Keep track of generator and discriminator losses
        iteration_losses_D = []
        iteration_losses_G = []

        epoch_psnr = []
        epoch_ssim = []

        batches = len(train_loader)
        
        # Iterate over a batch
        for _, (LR_patches, HR_patches, _) in enumerate(train_loader):

            loss_D, loss_G = do_epoch(LR_patches, HR_patches)

            iteration_losses_D.append(loss_D.detach().item())
            iteration_losses_G.append(loss_G.detach().item())

            if epoch % 1 == 0:
                with torch.no_grad():
                    batch_psnr = []
                    batch_ssim = []
                    
                    for i in range(batch_size):
                        LR_patch = LR_patches[i].unsqueeze(0).to(device)
                        out_G = gan_G(LR_patch).detach().cpu().numpy().squeeze(0)
                        HR_patch = HR_patches[i].numpy()
                        batch_psnr.append(psnr(out_G, HR_patch))
                        batch_ssim.append(ssim(out_G, HR_patch, channel_axis=0, data_range=1.0))
                        
                        del LR_patch, out_G
                    epoch_psnr.append(sum(epoch_psnr)/batch_size)
                    epoch_ssim.append(sum(epoch_ssim)/batch_size)

        if epoch % 1  == 0:
            avg_psnr.append(sum(epoch_psnr)/batches)
            avg_ssim.append(sum(epoch_ssim)/batches)

            if verbose:
                    print(f"Epoch {epoch+1}/{num_epoch}:")
                    print(f"Discriminator loss: {iteration_losses_D[-1]:.4f}")
                    print(f"Generator loss: {iteration_losses_G[-1]:.4f}")
                    print(f"Epoch run time: {time.time() - start_time}")

    # Get run time
    train_runtime = time.time() - train_start_time
    
    print("Done.")

    if verbose:
        print(f"Epoch {num_epoch}/{num_epoch}:")
        print(f"Discriminator loss: {iteration_losses_D[-1]:.4f}")
        print(f"Generator loss: {iteration_losses_G[-1]:.4f}")
        print(f"Epoch run time: {time.time() - start_time}")

    # Save metrics log and model
    save_log(num_images, train_runtime, avg_psnr, avg_ssim, "N/a (LPIPS uses too much GPU memory when training)", output_dir)
    save_model(gan_G.module, output_dir)


def GAN_ISR_Batch_eval(gan_G, dataset, output_dir, batch_size, verbose=False):
    # Perform SISR using GAN on a batch of images and evaluate performance

    # Initialise performance metrics
    running_psnr = 0
    running_ssim = 0
    running_lpips = 0
    start_time = time.time()

    # Perform SISR using the generator for batch_size many images
    for idx in range(batch_size):   
        LR_image, HR_image, image_name = dataset[idx]
        HR_image = HR_image.to(device).unsqueeze(0)
        LR_image = LR_image.to(device).unsqueeze(0)

        if verbose:
            print(f"Starting on {image_name}.  ({idx}/{batch_size})")

        # Perform DIP SISR for the current image
        resolved_image = gan_G(LR_image)

        # Accumulate metrics
        running_lpips += lpips(resolved_image, HR_image)
       
        resolved_image = np.clip(torch_to_np(resolved_image), 0, 1)
        HR_image = np.clip( torch_to_np(HR_image), 0, 1)
        
        running_psnr += psnr(resolved_image, HR_image)
        running_ssim += ssim(resolved_image, HR_image, data_range=1, channel_axis=0)

        if verbose:
            print("Done.")

        resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        save_image(resolved_image, image_name, output_dir)

        if verbose:
            print("\n")

    if verbose:
        print(f"Done for all {batch_size} images.")

    # Get run time
    runtime = time.time() - start_time

    # Calculate metric averages
    avg_psnr = running_psnr / batch_size
    avg_ssim = running_ssim / batch_size
    avg_lpips = running_lpips / batch_size
    
    # Save metrics log
    save_log(batch_size, runtime, avg_psnr, avg_ssim, avg_lpips, output_dir)


def GAN_ISR_Batch_inf(gan_G, dataset, output_dir, batch_size, verbose=False):
    # Perform SISR using GAN on a batch of images
    if verbose:
        print(f'Starting ISR inference on {batch_size} images.')

    # Perform SISR using the generator for batch_size many images
    for idx in range(batch_size):   

        LR_image, _, image_name = dataset[idx]
        LR_image = LR_image.to(device).unsqueeze(0)

        if verbose:
            print(f"Starting on {image_name}.  {idx}/{batch_size}")

        # Perform DIP SISR for the current image
        resolved_image = gan_G(LR_image)
        resolved_image = np.clip(torch_to_np(resolved_image), 0, 1)

        if verbose:
            print("Done.")

        # Save the image
        resolved_image = (resolved_image.transpose(1, 2, 0) * 255).astype(np.uint8)
        save_image(resolved_image, image_name, output_dir)

        if verbose:
            print("\n")

    if verbose:
        print(f"Done for all {batch_size} images.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--out_dir', type=str, help="Path to directory for dataset, saved images, saved models", required=True)
    parser.add_argument('--mode', type=str, help='"train": train model, "eval": get evaluation metrics of trained model over test set', required=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs when training (--mode=train)', default=1)
    parser.add_argument('--num_images', type=int, help='Number of images to use for training/evaluation', default=-1)
    parser.add_argument('--model_path', type=str, help='Path to trained model for evaluation (--mode="eval")', required=False)
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

    num_epoch = args.num_epoch

    model_path = ''
    if args.model_path:
        # Inference settings (if mode = 'inf')
        model_path = args.model_path 

    if model_path == '' and mode == 'eval':
        print(f'Must provide a model with --model flag to perform evaluation with.')
        sys.exit(1)

    # Display information during running
    verbose = args.verbose

    # Number of images from the dataset to use
    num_images = args.num_images # -1 for entire dataset, 1 for a running GAN on a single image

    if num_images <= -1 or num_images == 0:
        print(f'Please provide a valid number of images to use with --num_images=-1 for entire dataset or --num_images > 0')
        sys.exit(1)

    # Super resolution scale factor
    factor = 8

    # Generator input size
    HR_patch_size = (96,96)
    
    # Degredation
    downsample = args.downsample
    if downsample:
        factor *= 2

    LR_patch_size = (int(HR_patch_size[0] / factor), int(HR_patch_size[1] / factor))

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

    if (noise_type and mode == 'train') or (downsample and mode == 'train'):
        print(f'Noise and downsampling are only supported when evaluating (--mode=eval)')
        sys.exit(1)

    #print(f"Before loading models: {torch.cuda.memory_allocated() / (1024. ** 3)}GB")
    # Get generator and discriminator
    gan_G = nn.DataParallel(Generator(factor=factor)).to(device)
    if mode == 'train':
        gan_D = nn.DataParallel(Discriminator(HR_patch_size)).to(device)
    #print(f"After loading models: {torch.cuda.memory_allocated() / (1024. ** 3)}GB")

    # Number of minibatch image patches when training(16 in reference)
    batch_size = 16

    # Decide to train or do inference on batch of LR images
    if mode == 'train':
        dataset = GANDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images, LR_patch_size=LR_patch_size, HR_dir=HR_dir, downsample=downsample, noise_type=noise_type, train=True)

        batch_size = num_images if batch_size > num_images else batch_size

        # Train the data using the dataloader
        train_loader = DataLoader(dataset, batch_size=batch_size)
        GAN_ISR_train(gan_G, gan_D, train_loader, trained_dir, num_epoch, verbose=verbose)

    elif mode == 'eval':
        # Load trained model
        gan_G.load_state_dict(torch.load(model_path))

        # Set to evaluation mode
        gan_G.eval()

        # Get dataset for evaluation
        dataset = GANDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images, HR_dir=HR_dir)

        GAN_ISR_Batch_eval(gan_G, dataset, output_dir, num_images, verbose=verbose)

    elif mode == 'inf':
        # Load trained model
        gan_G.load_state_dict(torch.load(model_path))

        # Set to evaluation mode
        gan_G.eval()

        # Get dataset for inference (i.e. without ground truth!)
        dataset = GANDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images)

        GAN_ISR_Batch_inf(gan_G, dataset, output_dir, num_images, verbose=verbose)
