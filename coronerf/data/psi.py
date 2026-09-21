###################################################################
### FUNCTIONS TO LOAD IN PSI ARRAYS
###################################################################

from pathlib import Path
import numpy as np
import torch

from ..util.torch import np_to_torch

def load_psi_np(dataset_dir):
    '''
    loads raw fields
    '''
    dataset_dir = Path(dataset_dir)
    psi = np.load(dataset_dir / 'psi_fields.npz', allow_pickle = False)
    return psi

def get_psi_axes_np(dataset_dir, dtype = None):
    '''
    returns spherical axes of psi in llr
    
    lons: [0, 2pi)
    lats: [-pi/2, pi/2]
    rs: radius in R_sun
    '''
    psi = load_psi_np(dataset_dir)

    lons = psi['lons']
    colats = psi['colats']
    rs = psi['rs']

    # colat to lat (psi stores in colat)
    # the copy is to remove negative strides which is not supported in torch tensors later
    # [-pi/2, pi/2], do ::-1 to make -pi/2 the first entry and pi/2 last entry
    lats = np.pi / 2 - colats[::-1].copy()

    if dtype is not None:
        lons = lons.astype(dtype, copy = False)
        lats = lats.astype(dtype, copy = False)
        rs = rs.astype(dtype, copy = False)

    return lons, lats, rs

def get_psi_field_np(dataset_dir, key: str, log = False, dtype = None):
    '''
    loads the field and convert it to same lat ordering

    key: str, 'T' | 'ne'
    log: bool (whether or not to take log10)
    '''
    psi = load_psi_np(dataset_dir)
    field_colats = psi[key]

    # flip colat axis
    field_lats = field_colats[:,::-1, :].copy() # flip direction in field as well!!

    if log:
        field_lats = np.log10(field_lats)
    
    if dtype is not None:
        field_lats = field_lats.astype(dtype, copy = False)
    
    return field_lats

def get_psi_axes_torch(dataset_dir, torch_dtype = torch.float32):
    lons, lats, rs = get_psi_axes_np(dataset_dir)
    return (
        np_to_torch(lons, torch_dtype),
        np_to_torch(lats, torch_dtype),
        np_to_torch(rs, torch_dtype),
    )

def get_psi_field_torch(dataset_dir, key: str, log = False, torch_dtype = torch.float32):
    field = get_psi_field_np(dataset_dir, key = key, log = log)
    return np_to_torch(field, torch_dtype)

