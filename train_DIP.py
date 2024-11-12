import torch
import torch.optim

from utils.downsampler import Downsampler
from models.DIP import get_DIP_network

from dataset import DIV2KDataset
from utils.DIP import *
from utils.metrics import *

# Torch setup
torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def DIP_ISR(net, LR_image, scale_factor, eval=False, HR_GT=None, optimiser_type='adam'):

    assert (eval and HR_GT is not None) or (not eval and HR_GT is None), "Provide high-resolution ground truth when evaluating"


    # Get fixed noise for the network input
    net_input = get_noise(32, 'noise', (LR_image.size()[1]*scale_factor, LR_image.size()[2]*scale_factor)).type(dtype).detach()
    noise = net_input.detach().clone()

    # Include regulariser noise
    if reg_noise_std > 0:
        net_input = noise + (noise.normal_() * reg_noise_std)

    # Put everything on the GPU
    net_input = net_input.to(device)
    LR_image = LR_image.to(device).unsqueeze(0)

    if eval:
        HR_GT = HR_GT.to(device).unsqueeze(0)
        HR_GT_np = HR_GT.detach().cpu().numpy()[0]

    # Optimise the network over the input
    i = 0
    params = get_params('net', net, net_input)

    # Define closure for training
    def closure():
        nonlocal i

        # Get model output
        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        # Calculate loss
        total_loss = mse(out_LR, LR_image) 
        
        # Backpropagate loss
        total_loss.backward()

        if eval:
            if i % 200 == 0:
                out_HR_np = out_HR.detach().cpu().numpy()[0]
                print(f"Iteration {i}:")
                print(f"PSNR: {PSNR(out_HR_np, HR_GT_np)}")
                print(f"SSIM: {SSIM(out_HR_np, HR_GT_np, channel_axis=0, data_range=1.0)}")
                print(f"LPIPS: {LPIPS(out_HR, HR_GT)}\n")
        i += 1

        return total_loss

    optimize(optimiser_type, params, closure, learning_rate, num_iter)

    # Return the super-resolved image after training the network
    net_out = np.clip(torch_to_np(net(net_input)), 0, 1)

    # If evaluating, return eval metrics
    if eval:
        net_metrics_ = {
            "PSNR" : PSNR(net_out, HR_GT),
            "SSIM" : SSIM(net_out, HR_GT),
            "LPIPS" : LPIPS(net_out, HR_GT)
        }
        return net_out, net_metrics_
    else:
        return net_out, None


if __name__ == '__main__':
    # Whether to evaluate the DIP ISR model
    eval = True

    # Super resolution scale factor
    factor = 4

    # Get the training data loader
    train_data = DIV2KDataset(LR_dir='data/DIV2K_train_LR_x8/',
                            HR_dir='data/DIV2K_train_HR/',
                            scale_factor=factor)

    # Get the downsampler used to optimise
    downsampler = Downsampler(n_planes=3, factor=factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

    # Hyperparameters
    learning_rate = 0.01
    num_iter = 4000
    reg_noise_std = 0.05

    # Define optimiser
    optimiser_type = 'adam'

    # Define loss
    mse = torch.nn.MSELoss().type(dtype)

    # Get the DIP network
    net = get_DIP_network()
    net = net.to(device)

    # Choose an image
    idx = 0
    if eval:
        LR_image, HR_GT = train_data[idx]

        # Perform ISR by optimising DIP over the image
        _, net_metrics = DIP_ISR(net, LR_image, factor, True, HR_GT)
    else:
        LR_image, _ = train_data[idx]

        # Perform ISR by optimising DIP over the image
        resolved, _ = DIP_ISR(net, LR_image, factor)


