from skimage import metrics
import lpips
import torch

def psnr(im0, im1):
    return metrics.peak_signal_noise_ratio(image_true=im0, image_test=im1)

def ssim(im0, im1, **kwargs):
    return metrics.structural_similarity(im0, im1, **kwargs)


loss_fn = lpips.lpips(net='alex').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
def lpips(im0, im1):
    return loss_fn.forward(im0,im1)