import torch
import os
import argparse
import sys
from datetime import datetime
import time
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset import DIV2KDataset
from models.SRCNN.cnn import CNN
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True

def CNN_ISR_train(cnn, data_loader, num_epochs, criterion, optimizer , device):

    # Get metrics
    psnr = PSNR().to(device)
    ssim = SSIM(data_range=1.0).to(device)
    lpips = LPIPS(net_type='alex').to(device)

    avg_psnrs = []
    avg_ssims = []
    avg_lpipss = []

    batches = len(data_loader)

    for epoch in range(num_epochs):

        epoch_psnrs = []
        epoch_ssims = []
        epoch_lpipss = []

        for _, (LR_image, HR_image, _) in data_loader:

            LR_image = LR_image.to(device)
            HR_image = HR_image.to(device)

            resolved_image = cnn(LR_image)

            loss = criterion(resolved_image, HR_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % train_log_freq == 0:
                with torch.no_grad():
                
                    epoch_psnrs.append(psnr(resolved_image, HR_image).item())
                    epoch_ssims.append(ssim(resolved_image, HR_image).item())
                    epoch_lpipss.append(lpips(resolved_image, HR_image).item())

        if epoch % train_log_freq  == 0:
            avg_psnrs.append(sum(epoch_psnrs)/batches)
            avg_ssims.append(sum(epoch_ssims)/batches)
            avg_lpipss.append(sum(epoch_lpipss)/batches)
    
    train_metrics = {
        "Average PSNR during training" : avg_psnrs,
        "Average SSIM during training" : avg_ssims,
        "Average LPIPS during training" : avg_lpipss,
    }

    return cnn, train_metrics


def main(LR_dir, 
         HR_dir, 
         out_dir, 
         factor, 
         num_images, 
         num_epochs,
         train_log_freq,
         downsample,
         device):

    # Get generator and wrap with DDP
    cnn = CNN(scale_factor=factor).to(device)
    cnn.train()

    # Number of minibatch image patches when training(16 in reference)
    batch_size = 64

    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=1e-4)

    # Load the required dataset
    dataset = DIV2KDataset(LR_dir=LR_dir, HR_dir=HR_dir,scale_factor=factor, num_images=num_images, downsample=downsample)

    # Create a dataloader           
    data_loader =  DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)

    start_time = time.time()

    # Train
    print('Beginning training.')
    trained_model, train_metrics = CNN_ISR_train(cnn, data_loader, num_epochs, train_log_freq, criterion=criterion, optimizer=optimizer, device=device)
    print('Done training.')

    # Get run time
    runtime = time.time() - start_time
    runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))

    # Final train metric for the log
    train_metrics["Number of images used for training"] = num_images
    train_metrics["Train runtime"] = runtime

    # Save metrics log and model
    save_log(out_dir, **train_metrics)
    save_model(trained_model, 'cnn', out_dir)


# Setup all the parameters for the GAN script
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Get command line arguments for program behaviour
    parser.add_argument('--data_dir', type=str, help="Path to directory for dataset", required=True)
    parser.add_argument('--out_dir', type=str, help="Path to directory for dataset, saved images, saved models", required=True)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs when pre-training', default=400)
    parser.add_argument('--train_log_freq', type=int, help='How many epochs between logging metrics when training', default=100)
    parser.add_argument('--num_images', type=int, help='Number of images to use for training', default=-1)
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
    LR_dir = os.path.join(data_dir, 'DIV2K_train_LR_x8/')
    HR_dir = os.path.join(data_dir, 'DIV2K_train_HR/')

    # Super resolution scale factor
    factor = 8

    # Degredation
    downsample = args.downsample
    if downsample:
        factor *= 2

    # Output directory
    date = datetime.now()
    out_dir = os.path.join(out_dir, f'trained/CNNx{factor}/{date.strftime("%Y_%m_%d_%p%I_%M")}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # How many epochs between saving metrics when training
    train_log_freq = args.train_log_freq

    # Number of images from the dataset to use
    num_images = args.num_images # -1 for entire dataset, 1 for a running GAN on a single image

    if num_images < -1 or num_images == 0:
        print(f'Please provide a valid number of images to use with --num_images=-1 for entire dataset or --num_images > 0')
        sys.exit(1)

    num_epochs = args.num_epochs

    # Initialise gpus
    device = 0
    main(LR_dir, 
         HR_dir, 
         out_dir, 
         factor, 
         num_images, 
         num_epochs,
         train_log_freq,
         downsample,
         device)