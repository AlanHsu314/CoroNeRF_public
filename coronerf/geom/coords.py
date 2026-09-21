##################################################
### coordinate transforms
##################################################

import torch
import math

def llr_to_sph(pos: torch.Tensor):
    '''
    given pos = [3,], coordinates in (lon, lat, r), return them in spherical (r, theta, phi)
    '''
    lon, lat, r = pos.unbind(-1)

    # convert to spherical
    obs_r = r
    obs_theta = math.pi/2 - lat 
    obs_phi = lon

    return torch.stack([obs_r, obs_theta, obs_phi], dim = -1)

def sph_to_xyz(pos: torch.Tensor):
    '''
    given coordinates in (r, theta, phi), return them in (x, y, z) in solar radii
    '''
    r, theta, phi = pos.unbind(-1)

    # convert to cartesian
    obs_x = r*torch.sin(theta)*torch.cos(phi)
    obs_y = r*torch.sin(theta)*torch.sin(phi)
    obs_z = r*torch.cos(theta)

    return torch.stack([obs_x, obs_y, obs_z], dim = -1)

def llr_to_xyz(pos: torch.Tensor):
    '''
    takes in x = [..., 3] with each row as, converts coordinates from llr to xyz
    '''
    return sph_to_xyz(llr_to_sph(pos))

