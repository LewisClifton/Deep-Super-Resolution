from skimage import metrics
import torch

def psnr(im0, im1):
    return metrics.peak_signal_noise_ratio(image_true=im0, image_test=im1)

def ssim(im0, im1, **kwargs):
    return metrics.structural_similarity(im0, im1, **kwargs)


def lpips(im0, im1, lpips_model):
    # Loading and deleting LPIPS is slow but it saves a lot of GPU memory
    with torch.no_grad():
        loss =  lpips_model.forward(im0,im1).item()
        del lpips_model
        return loss