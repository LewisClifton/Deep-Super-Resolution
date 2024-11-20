import torch
import os
from torch.utils.data import DataLoader
from datetime import datetime
import time

from models.GAN.discriminator import Discriminator
from models.GAN.generator import Generator
from dataset import DIV2KDataset, GANTrainDIV2KDataset
from utils.SRGAN import *
from utils.metrics import *
from utils.common import *


# Torch setup
torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def GAN_ISR_train(gan_G, gan_D, train_loader, output_dir, num_epoch=5, verbose=False):
    # Train GAN to perform SISR
    
    def do_epoch(LR_images, HR_images):

        # Train Discriminator
        real_output_D = gan_D(HR_images) # Discriminator output for real HR images
        
        fake_output_G = gan_G(LR_images) # Generator output for LR images
        fake_output_D = gan_D(fake_output_G.detach()) # Discriminator output for fake HR images (i.e. generated from LR by generator)
        loss_D = get_loss_D(real_output_D, fake_output_D)

        # Update Discriminator
        gan_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        # Train Generator
        fake_output_G = gan_G(LR_images)
        fake_output_D = gan_D(fake_output_G.detach())
        loss_G = perceptualLoss(fake_output_G, HR_images, fake_output_D)

        # Update Generator
        gan_G.zero_grad()
        loss_G.backward()
        optim_G.step()


        return loss_D, loss_G
    
    if verbose:
        print(f'Starting GAN training..')

    train_start_time = time.time()

    for epoch in range(num_epoch):

        # Get iteration start time
        start_time = time.time()

        # Keep track of generator and discriminator losses
        iteration_losses_D = []
        iteration_losses_G = []
        
        # Iterate over a batch
        for _, (LR_patches, HR_patches, _) in enumerate(train_loader):
            
            # Stack the patches along the batch dimension
            LR_patches = LR_patches.view(-1, 3, LR_patches.shape[3], LR_patches.shape[4])
            HR_patches = HR_patches.view(-1, 3, HR_patches.shape[3], HR_patches.shape[4])

            LR_patches = LR_patches.to(device)
            HR_patches = HR_patches.to(device)
            
            loss_D, loss_G = do_epoch(LR_patches, HR_patches)

            iteration_losses_D.append(loss_D.detach().item())
            iteration_losses_G.append(loss_G.detach().item())

        if verbose:
            if epoch % 200 == 0 :
                print(f"Epoch {epoch+1}/{num_epoch}:")
                print(f"Discriminator loss: {iteration_losses_D[-1]:.4f}")
                print(f"Generator loss: {iteration_losses_G[-1]:.4f}")
                print(f"Epoch run time: {time.time() - start_time}")

    train_runtime = time.time() - train_start_time
    
    print("Done.")

    save_log(num_images, train_runtime, "N/a", "N/a", "N/a", output_dir)

    save_model(gan_G, output_dir)


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

    # Determine program behaviour
    verbose = True
    mode = 'inf' # 'inf', 'train', 'eval'
    num_images = 1

    # Inference settings (if mode = 'inf')
    model_path = r'trained\GAN\2024_11_20_PM12_31.pth'
    
    # Set the output and trained model directory
    output_dir = os.path.join(os.getcwd(), rf'out\GAN\{datetime.now().strftime("%Y_%m_%d_%p%I_%M")}')
    trained_dir = os.path.join(os.getcwd(), r'trained\GAN')

    # Super resolution scale factor
    factor = 4

    # Get dataset
    LR_dir = 'data/DIV2K_train_LR_x8/'
    HR_dir = ''#'data/DIV2K_train_HR/' # = '' if not using HR GT for evaluation

    # Get loss functions
    bce_loss = nn.BCELoss().to(device)
    vgg_loss = Vgg19Loss().to(device)
    mse_loss = nn.MSELoss().to(device)

    # Generator loss
    def get_adversarial_loss(fake_output):
        adversarial_loss = bce_loss(fake_output, torch.ones_like(fake_output))
        return adversarial_loss

    # Discriminator loss
    def get_loss_D(real_output, fake_output):
        real_loss = bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))
        loss_D = real_loss + fake_loss
        return loss_D

    # Get generator training loss function
    class PerceptualLoss():
        def __init__(self, content_loss_fn):
            self.content_loss_fn = content_loss_fn

        def __call__(self, fake_output_G, HR_images, fake_output_D):
            # Content less: MSE loss or vgg loss
            content_loss = self.content_loss_fn(fake_output_G, HR_images)
            adversarial_loss_ = get_adversarial_loss(fake_output_D)

            perceptual_loss = content_loss + adversarial_loss_

            return perceptual_loss
        
    perceptualLoss = PerceptualLoss(vgg_loss) # PerceptualLoss(mse)

    # Get generator and discriminator
    gan_G = Generator().to(device)
    gan_D = Discriminator().to(device)

    # Train hyperparameters
    batch_size = 4
    num_epoch = 1 # 1e+5 in paper
    optim_G = torch.optim.Adam(gan_G.parameters(), lr=1e-4)
    optim_D = torch.optim.Adam(gan_D.parameters(), lr=1e-4)

    # Decide to train or do inference on bathc of LR images
    if mode == 'train':
        dataset = GANTrainDIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images, HR_dir=HR_dir)

        # Train the data using the dataloader
        train_loader = DataLoader(dataset, batch_size=batch_size)
        GAN_ISR_train(gan_G, gan_D, train_loader, trained_dir, num_epoch, verbose=verbose)

    elif mode == 'eval':
        # Load trained model
        gan_G.load_state_dict(torch.load(model_path))

        # Set to evaluation mode
        gan_G.eval()

        # Get dataset for evaluation
        dataset = DIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images, HR_dir=HR_dir)

        GAN_ISR_Batch_eval(gan_G, dataset, output_dir, batch_size, verbose=verbose)

    elif mode == 'inf':
        # Load trained model
        gan_G.load_state_dict(torch.load(model_path))

        # Set to evaluation mode
        gan_G.eval()

        # Get dataset for inference (i.e. without ground truth!)
        dataset = DIV2KDataset(LR_dir=LR_dir, scale_factor=factor, num_images=num_images)

        GAN_ISR_Batch_inf(gan_G, dataset, output_dir, num_images, verbose=verbose)
