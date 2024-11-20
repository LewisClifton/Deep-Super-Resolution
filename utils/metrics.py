from skimage import metrics
import lpips as lpips_
import torch

def psnr(im0, im1):
    return metrics.peak_signal_noise_ratio(image_true=im0, image_test=im1)

def ssim(im0, im1, **kwargs):
    return metrics.structural_similarity(im0, im1, **kwargs)


def lpips(im0, im1):
    # Loading and deleting LPIPS is slow but it saves a lot of GPU memory
    loss_fn = lpips_.LPIPS(net='alex').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        loss =  loss_fn.forward(im0,im1).item()
    loss_fn.cpu()
    del loss_fn
    return loss