####################################################
### implementation of density MLP using hash grids
####################################################

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import spherical_harmonics_encoding
from .hash_encoding import build_hash_encoder
from ..util.torch import match_ref


class DensityHashMLP(nn.Module):
    '''
    Hash-grid density model:

    Input: x_AABB in [-1, 1]^3
        -> map to [0, 1]^3
        -> multiresolution hash encoder
        -> small decoder MLP head
        -> scalar log10(n_e)

    Optional:
        - concatenate raw xyz
        - contacenta SH encoding of unit direction
    '''
    def __init__(self,
                 backend: str = 'torch',
                 backend_fallback: bool = True,
                 in_dim: int = 3,
                 n_levels: int = 16,
                 n_features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 base_resolution: int = 16,
                 finest_resolution: int = 512,
                 decoder_hidden_dim: int = 64,
                 decoder_num_hidden: int = 2,
                 dtype: torch.dtype = torch.float32,
                 strict_types: bool = False,
                 use_raw_xyz: bool = False,
                 use_sh_encoding: bool = False,
                 sh_degree: int = 2,
                 ):
        
        super().__init__()

        # attrs
        self.dtype = dtype
        self.strict_types = strict_types
        self.use_raw_xyz = use_raw_xyz
        self.use_sh_encoding = use_sh_encoding
        self.sh_degree = sh_degree

        # init hash encoder
        self.encoder = build_hash_encoder(
            backend = backend,
            backend_fallback = backend_fallback,
            n_input_dims = in_dim,
            n_levels = n_levels,
            n_features_per_level = n_features_per_level,
            log2_hashmap_size = log2_hashmap_size,
            base_resolution = base_resolution,
            finest_resolution = finest_resolution,
            dtype = dtype,
            strict_types = strict_types,
        )

        # params for MLP head
        in_features = self.encoder.out_dim

        if self.use_raw_xyz:
            in_features += 3
        
        if self.use_sh_encoding:
            self.sh_dim = (self.sh_degree + 1) ** 2
            in_features += self.sh_dim
        else:
            self.sh_dim = 0

        # build layers
        layers = []
        dims = [in_features] + [decoder_hidden_dim] * decoder_num_hidden + [1]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        self.to(dtype)

    def initialize_density_bias(self, value: float):
        '''
        used by renderer / setup to bias the initial density estimate
        '''
        with torch.no_grad():
            last = self.net[-1]
            if hasattr(last, 'bias') and last.bias is not None:
                last.bias.fill_(float(value))

    def forward(self, x_AABB: torch.Tensor) -> torch.Tensor:
        '''
        Inputs
        ------
        x_AABB: (...,3) point in AABB [-1, 1]^3

        Returns
        -------
        log10_ne: (...,)
        '''

        # strict types
        ref_param = next(self.net.parameters())

        if self.strict_types:
            assert x_AABB.dtype == ref_param.dtype, \
                f'DensityHashMLP dtype mismatch: x={x_AABB.dtype}, params={ref_param.dtype}'
            assert x_AABB.device == ref_param.device, \
                f'DensityHashMLP device mismatch: x={x_AABB.device}, params={ref_param.device}'
            x_AABB = x_AABB.contiguous()
        else:
            x_AABB = match_ref(ref_param, x_AABB)

        # if there are more than 1 leading dim, flatten and restore later
        orig_shape = x_AABB.shape[:-1] # last dim is always 3
        x_flat = x_AABB.reshape(-1, 3)

        # [-1, 1]^3 --> [0, 1]^3 for hash-grid bounds
        x_unit = 0.5 * (x_flat + 1.0)
        x_unit = x_unit.clamp(0.0, 1.0)

        # compute features
        feat_list = [self.encoder(x_unit)]

        # append optional features
        if self.use_raw_xyz:
            feat_list.append(x_flat)

        if self.use_sh_encoding:
            # same as density_MLP
            r = torch.linalg.norm(x_flat, dim = -1, keepdim = True)
            u = x_flat / (r + 1e-9)

            if self.strict_types:
                assert u.dtype == ref_param.dtype and u.device == ref_param.device
                u = u.contiguous()
            else:
                u = match_ref(ref_param, u)

            dir_enc = spherical_harmonics_encoding(u, degree = self.sh_degree)
            feat_list.append(dir_enc)

        # cat and feed into decoder
        x_in = torch.cat(feat_list, dim = -1)
        ne_out = self.net(x_in).squeeze(-1)

        return ne_out.reshape(*orig_shape)










