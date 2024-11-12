import torch

def get_mgrid(sidelen1,sidelen2, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''

    if sidelen1 >= sidelen2:
      # use sidelen1 steps to generate the grid
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen1)])
      mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)
      # crop it along one axis to fit sidelen2
      minor = int((sidelen1 - sidelen2)/2)
      mgrid = mgrid[:,minor:sidelen2 + minor]

    if sidelen1 < sidelen2:
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen2)])
      mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)

      minor = int((sidelen2 - sidelen1)/2)
      mgrid = mgrid[minor:sidelen1 + minor,:]

    # flatten the gird
    mgrid = mgrid.reshape(-1, dim)

    return mgrid