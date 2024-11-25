import torch
import os
import argparse
import sys
from datetime import datetime
import time
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from models.GAN.discriminator import Discriminator
from models.GAN.generator import Generator
from dataset import GANDIV2KDataset
from utils.GAN import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True


def GAN_ISR_train(gan_G, gan_D, lr, train_loader, num_epoch, train_log_freq, device):
    # Train GAN to perform SISR

    # Get loss functions
    bce_loss = nn.BCELoss()
    perceptualLoss = PerceptualLoss().to(device)

    # Get metrics
    psnr = PSNR().to(device)
    ssim = SSIM(data_range=1.0).to(device)
    lpips = LPIPS(net_type='alex').to(device)
    
    # Get optimisers for both models
    optim_G = torch.optim.Adam(gan_G.parameters(), lr=lr)
    optim_D = torch.optim.Adam(gan_D.parameters(), lr=lr)
    
    def do_epoch(LR_patches, HR_patches):

        LR_patches = LR_patches.to(device)
        HR_patches = HR_patches.to(device)

        # Train Discriminator
        real_output_D = gan_D(HR_patches) # Discriminator output for real HR images
        
        fake_output_G = gan_G(LR_patches).detach() # Generator output for LR images
        fake_output_D = gan_D(fake_output_G) # Discriminator output for fake HR images (i.e. generated from LR by generator)
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
    avg_ssims = []
    avg_lpipss = []

    for epoch in range(num_epoch):

        # Get iteration start time
        start_time = time.time()

        # Keep track of generator and discriminator losses
        iteration_losses_D = []
        iteration_losses_G = []

        epoch_psnrs = []
        epoch_ssims = []
        epoch_lpipss = []

        batches = len(train_loader)
        
        # Iterate over a batch
        for _, (LR_patches, HR_patches, _) in enumerate(train_loader):

            loss_D, loss_G = do_epoch(LR_patches, HR_patches)

            iteration_losses_D.append(loss_D.detach().item())
            iteration_losses_G.append(loss_G.detach().item())

            if epoch % train_log_freq == 0:
                with torch.no_grad():

                    LR_patches = LR_patches.to(device)
                    HR_patches = HR_patches.to(device)
                    
                    out_G = gan_G(LR_patches)
                    epoch_psnrs.append(psnr(out_G, HR_patches).item())
                    epoch_ssims.append(ssim(out_G, HR_patches).item())
                    epoch_lpipss.append(lpips(out_G, HR_patches).item())

                    del LR_patches, HR_patches, out_G
                    

        if epoch % train_log_freq  == 0:
            avg_psnrs.append(sum(epoch_psnrs)/batches)
            avg_ssims.append(sum(epoch_ssims)/batches)
            avg_lpipss.append(sum(epoch_lpipss)/batches)

            print(f"Epoch {epoch+1}/{num_epoch}:")
            print(f"Discriminator loss: {iteration_losses_D[-1]:.4f}")
            print(f"Generator loss: {iteration_losses_G[-1]:.4f}")
            print(f"Epoch run time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}")

    
    train_metrics = {
        "Average PSNR during training" : avg_psnrs,
        "Average SSIM during training" : avg_ssims,
        "Average LPIPS during training" : avg_lpipss,
        'Final Generator loss' : iteration_losses_D[-1],
        'Final Discriminator loss' : iteration_losses_G[-1]
    }

    return gan_G, gan_D, train_metrics


def main(LR_dir, 
         HR_dir, 
         out_dir, 
         factor, 
         num_images, 
         pre_train_epochs,
         fine_tune_epochs,
         pre_train_lr,
         fine_tune_lr,
         LR_patch_size,
         HR_patch_size,
         train_log_freq,
         pre_trained_model_path,
         device):

    # Get generator and wrap with DDP
    gan_G = Generator(factor=factor).to(device)

    # Get discriminator and wrap with DDP
    gan_D = Discriminator(HR_patch_size).to(device)

    if pre_trained_model_path is not None:
        gan_G = torch.load(os.path.join(pre_trained_model_path, 'pre_trained_srgan_G.pth'), weights_only=True)
        gan_D = torch.load(os.path.join(pre_trained_model_path, 'pre_trained_srgan_D.pth'), weights_only=True)
    
    # Set Generator and Discriminator to train mode
    gan_G.train(); gan_D.train()

    # Number of minibatch image patches when training(16 in reference)
    batch_size = 16

    # Load the required dataset
    dataset = GANDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images, LR_patch_size=LR_patch_size, HR_dir=HR_dir, train=True)

    # Create a dataloader           
    data_loader =  DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)

    start_time = time.time()

    # Pre-train
    if pre_trained_model_path is None:
        print("Beginnning pre-training stage..")
        pre_trained_G, pre_trained_D, train_metrics = GAN_ISR_train(gan_G, gan_D, pre_train_lr, data_loader, pre_train_epochs, train_log_freq, device=device)
        print("Done pre-training.")

        # Save metrics log and model
        save_log(out_dir, **train_metrics)
        save_model(pre_trained_G, 'pre_trained_srgan_G', out_dir)
        save_model(pre_trained_D, 'pre_trained_srgan_D', out_dir)

    # Train
    print('Beginning fine-tuning stage')
    trained_model, _, train_metrics = GAN_ISR_train(gan_G, gan_D, fine_tune_lr, data_loader, fine_tune_epochs, train_log_freq, device=device)
    print('Done fine-tuning stage.')

    # Save metrics log and model
    save_log(out_dir, **train_metrics)
    save_model(trained_model, 'fine_tuned_srgan_', out_dir)

    # Get run time
    runtime = time.time() - start_time
    runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

    # Final train metric for the log
    train_metrics["Number of images used for training"] = num_images
    train_metrics["Train runtime"] = runtime

    dist.barrier()
    if dist.is_initialized():
            dist.destroy_process_group()

# Setup all the parameters for the GAN script
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for dataset, saved images, saved models", required=True)
    parser.add_argument('--pre_train_epochs', type=int, help='Number of epochs when pre-training', default=8000)
    parser.add_argument('--fine_tune_epochs', type=int, help='Number of epochs when fine tuning', default=4000)
    parser.add_argument('--pre_train_learning_rate', type=float, help='Learning rate during pre-training', default=1e-4)
    parser.add_argument('--fine_tune_learning_rate', type=float, help='Learning rate during fine tuning', default=1e-5)
    parser.add_argument('--pre_trained_models_path', type=str, help='Path to model pre-trained model discriminator and generator (avoids pre-training again)')
    parser.add_argument('--train_log_freq', type=int, help='How many epochs between logging metrics when training', default=100)
    parser.add_argument('--num_images', type=int, help='Number of images to use for training', default=-1)
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
    LR_dir = os.path.join(data_dir, 'DIV2K_train_LR_x8/')
    HR_dir = os.path.join(data_dir, 'DIV2K_train_HR/')

    # Output directory
    date = datetime.now()
    out_dir = os.path.join(out_dir, f'trained/GAN/{date.strftime("%Y_%m_%d_%p%I_%M")}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Epochs
    pre_train_epochs = args.pre_train_epochs
    fine_tune_epochs = args.fine_tune_epochs

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

    # Learning rate
    pre_train_lr = args.pre_train_learning_rate
    fine_tune_lr = args.fine_tune_learning_rate

    # Pre-trained model path
    pre_trained_model_path = args.pre_trained_model_path

    # Initialise gpus
    device = 0
    main(LR_dir, 
         HR_dir, 
         out_dir, 
         factor, 
         num_images, 
         pre_train_epochs,
         fine_tune_epochs,
         pre_train_lr,
         fine_tune_lr,
         LR_patch_size,
         HR_patch_size,
         train_log_freq,
         pre_trained_model_path,
         device)