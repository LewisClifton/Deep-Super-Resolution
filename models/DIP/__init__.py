from .skip import skip

# The code in this module is taken from the supplementary codebase
# provided for the reference paper "Deep Image Prior" by Ulyanov et al.
# Deep Image Prior Github repository: https://github.com/DmitryUlyanov/deep-image-prior
# Deep Image Prior paper: https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf

def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    if NET_TYPE == 'skip':
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
    else:
        assert False

    return net