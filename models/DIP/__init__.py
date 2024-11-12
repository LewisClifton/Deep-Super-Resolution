from .skip import skip

# The code in this module is taken from the supplementary codebase
# provided for the reference paper "Deep Image Prior" by Ulyanov et al.
# Deep Image Prior Github repository: https://github.com/DmitryUlyanov/deep-image-prior
# Deep Image Prior paper: https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf

def get_DIP_network(input_depth=32, pad='reflection'):
    return skip(num_input_channels=input_depth, 
               num_output_channels=3, 
               num_channels_down = [128, 128, 128, 128, 128],
               num_channels_up = [128, 128, 128, 128, 128],
               num_channels_skip = [4, 4, 4, 4, 4],
               upsample_mode='bilinear', downsample_mode='stride',
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')