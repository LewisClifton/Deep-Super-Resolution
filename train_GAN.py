import torch
import os
import argparse
import sys
from datetime import datetime
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from models.GAN.discriminator import Discriminator
from models.GAN.generator import Generator
from dataset import GANDIV2KDataset
from utils.GAN import *
from utils.metrics import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True


def GAN_ISR_train(gan_G, gan_D, train_loader, num_epoch, train_log_freq, device):
    # Train GAN to perform SISR

    # Get loss functions
    bce_loss = nn.BCELoss()
    vgg_loss = Vgg19Loss()
    perceptualLoss = PerceptualLoss(vgg_loss)
    
    # Get optimisers for both models
    optim_G = torch.optim.Adam(gan_G.parameters(), lr=1e-4)
    optim_D = torch.optim.Adam(gan_D.parameters(), lr=1e-4)
    
    def do_epoch(LR_patches, HR_patches):

        LR_patches = LR_patches.to(device)
        HR_patches = HR_patches.to(device)

        # Train Discriminator
        real_output_D = gan_D(HR_patches) # Discriminator output for real HR images
        
        fake_output_G = gan_G(LR_patches) # Generator output for LR images
        fake_output_D = gan_D(fake_output_G.detach()) # Discriminator output for fake HR images (i.e. generated from LR by generator)
        loss_D = get_loss_D(real_output_D, fake_output_D, bce_loss)

        # Update Discriminator
        gan_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        # Train Generator
        fake_output_G = gan_G(LR_patches)
        fake_output_D = gan_D(fake_output_G.detach())
        loss_G = perceptualLoss(fake_output_G, HR_patches, fake_output_D, bce_loss)

        # Update Generator
        gan_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        # Delete everything to ensure gpu memory is low
        del fake_output_D, fake_output_G
        del real_output_D
        del LR_patches, HR_patches

        return loss_D, loss_G
    
    print(f'Starting GAN training..')

    avg_psnrs = []
    avg_ssims =[]

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

            if epoch % train_log_freq == 0:
                with torch.no_grad():
                    batch_psnr = []
                    batch_ssim = []
                    batch_size = LR_patches[0]
                    
                    for i in range(batch_size):
                        LR_patch = LR_patches[i].unsqueeze(0).to(device)
                        out_G = gan_G(LR_patch).detach().cpu().numpy().squeeze(0)
                        HR_patch = HR_patches[i].numpy()
                        batch_psnr.append(psnr(out_G, HR_patch))
                        batch_ssim.append(ssim(out_G, HR_patch, channel_axis=0, data_range=1.0))
                        
                        del LR_patch, out_G
                    epoch_psnr.append(sum(epoch_psnr)/batch_size)
                    epoch_ssim.append(sum(epoch_ssim)/batch_size)

        if epoch % train_log_freq  == 0:
            avg_psnrs.append(sum(epoch_psnr)/batches)
            avg_ssims.append(sum(epoch_ssim)/batches)

            print(f"Epoch {epoch+1}/{num_epoch}:")
            print(f"Discriminator loss: {iteration_losses_D[-1]:.4f}")
            print(f"Generator loss: {iteration_losses_G[-1]:.4f}")
            print(f"Epoch run time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")
    
    print("Done.")
    
    train_metrics = {
        'avg_psnr' : avg_psnrs,
        'avg_ssim' : avg_ssims,
        'd_loss' : iteration_losses_D[-1],
        'g_loss' : iteration_losses_G[-1]
    }

    return gan_G, train_metrics


def main(world_size,
         rank, 
         LR_dir, 
         HR_dir, 
         output_dir, 
         factor, 
         num_images, 
         num_epoch,
         LR_patch_size,
         HR_patch_size,
         train_log_freq):
    
    # setup the process groups
    setup_gpu(rank, world_size)

    # Get generator and wrap with DDP
    gan_G = Generator(factor=factor).to(rank)
    gan_G = DDP(gan_G, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Get discriminator and wrap with DDP
    gan_D = Discriminator(HR_patch_size).to(rank)
    gan_D = DDP(gan_D, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Set Generator and Discriminator to train mode
    gan_G.train(); gan_D.train()

    # Number of minibatch image patches when training(16 in reference)
    batch_size = 16

    # Load the required dataset
    dataset = GANDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images, LR_patch_size=LR_patch_size, HR_dir=HR_dir, train=True)

    # Create a dataloader           
    data_loader = get_data_loader(dataset, rank, world_size, batch_size)

    start_time = time.time()

    # Train
    trained_model, train_metrics = GAN_ISR_train(gan_G, gan_D, data_loader, num_epoch, train_log_freq, device=rank)

    # Get run time
    train_metrics['runtime'] = time.time() - start_time

    # Wait for all gpus to get to this point
    dist.barrier()

    # Send all the gpu node metrics back to the main gpu
    torch.cuda.set_device(rank)
    train_metrics_gpus = [None for _ in range(world_size)]
    dist.all_gather_object(train_metrics_gpus, train_metrics)

    if rank == 0:
        print('Done training')

        # Get runtime
        runtime = np.max([gpu_metrics['runtime'] for gpu_metrics in train_metrics_gpus])
        runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

        # Calculate mean across GPUs for each epoch
        avg_psnrs = [gpu_metrics['avg_psnrs'] for gpu_metrics in train_metrics_gpus]
        avg_ssims = [gpu_metrics['avg_ssims'] for gpu_metrics in train_metrics_gpus]
        avg_psnrs = np.mean(np.vstack(avg_psnrs), axis=0)
        avg_ssims = np.mean(np.vstack(avg_ssims), axis=0)

        # Calculate average final discriminator loss and generator loss across gpus
        d_loss = np.mean([gpu_metrics['d_loss'] for gpu_metrics in train_metrics_gpus])
        g_loss = np.mean([gpu_metrics['g_loss'] for gpu_metrics in train_metrics_gpus])

        # Final train metric for the log
        final_train_metrics = {
            "Number of images used for training" : num_images,
            "Train runtime" : runtime,
            "Average PSNR" : avg_psnrs.tolist(),
            "Average SSIM" : avg_ssims.tolist(),
            "Average LPIPS" : 'Not tracked during training due to VGG-19 memory overhead',
            'Final Generator loss' : d_loss,
            'Final Discriminator loss' : g_loss
        }

        # Output directory
        date = datetime.now()
        out_dir = os.path.join(out_dir, f'GAN/trained/{date.strftime("%Y_%m_%d_%p%I_%M")}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save metrics log and model
        save_log(output_dir, **final_train_metrics)
        save_model(trained_model.module, output_dir)




# Setup all the parameters for the GAN script
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for dataset, saved images, saved models", required=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs when training', default=1)
    parser.add_argument('--num_gpus', type=int, help='Number of gpus to run models with', default=2)
    parser.add_argument('--train_log_freq', type=int, help='How many epochs between logging metrics when training', default=100)
    parser.add_argument('--num_images', type=int, help='Number of images to use for training', default=-1)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.out_dir

    if not os.path.isdir(data_dir):
        print(f'{data_dir} not found.')
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f'{output_dir} not found.')
        sys.exit(1)

    # Get dataset
    LR_dir = os.path.join(data_dir, 'DIV2K_train_LR_x8/')
    HR_dir = os.path.join(data_dir, 'DIV2K_train_HR/')

    # Training epochs
    num_epoch = args.num_epochs

    # How many epochs between saving metrics when training
    train_log_freq = args.train_log_freq

    # Number of images from the dataset to use
    num_images = args.num_images # -1 for entire dataset, 1 for a running GAN on a single image

    if num_images < -1 or num_images == 0:
        print(f'Please provide a valid number of images to use with --num_images=-1 for entire dataset or --num_images > 0')
        sys.exit(1)

    # Super resolution scale factor
    factor = 8

    # Generator input size
    HR_patch_size = (96,96)
    LR_patch_size = (int(HR_patch_size[0] / factor), int(HR_patch_size[1] / factor))

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
              num_epoch,
              LR_patch_size,
              HR_patch_size,
              train_log_freq),
        nprocs=world_size)