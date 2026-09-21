#########################################################
### multi-resolution hash encoding implementation
#########################################################

from __future__ import annotations

import math
import logging
from typing import Optional, Union, List

import torch
import torch.nn as nn

from ..util.torch import match_ref

# tcnn backend
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except Exception:
    tcnn = None
    TCNN_AVAILABLE = False

#####################
# HELPERS
#####################

def compute_per_level_scale(
    n_levels: int,
    base_resolution: int,
    finest_resolution: int,
) -> float:
    '''
    geometric scaling factor for smoothing coarse --> fine 
    l = exp(ln(N_{L-1} / N_0) / (L - 1))

    where N_l = N_0 * b**l
    '''
    if n_levels < 2:
        return 1.0
    return math.exp(
        math.log(float(finest_resolution) / float(base_resolution)) / float(n_levels - 1)
    )

def hash_fn_3d(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, table_size: int) -> torch.Tensor:
    '''
    xor-prime hash

    note: implemented in torch tensors
    '''
    # use int64 for xor
    ix = ix.long()
    iy = iy.long()
    iz = iz.long()

    # multiply
    x = ix * 1_540_863
    y = iy * 1_253_291
    z = iz * 1_151_021

    # xor
    return torch.remainder(x ^ y ^ z, table_size)

class TorchHashGridEncoding(nn.Module):
    '''
    PyTorch implementation of a multi-resolution hash grid encoding of a N-D field
    Coordinates: [0, 1]^3

    For each level l:
        - choose resolution N_l
        - find 8 lattice corners around x
        - hash each corner coordinate to a atrainable table row
        - trilinearly interpolate 8 feature vectors associated with 8 coords

    forward output shape:
        (N, n_levels * n_features_per_level)
    '''
    def __init__(
        self,
        n_input_dims: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        dtype: torch.dtype = torch.float32,
        strict_types: bool = False,
    ):
        super().__init__()
        assert n_input_dims == 3, "This implementations assumes a 3D field."

        self.logger = logging.getLogger('coroNeRF.TorchHashGridEncoding')

        # inputs
        self.n_input_dims = n_input_dims
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.dtype = dtype
        self.strict_types = strict_types

        # compute geometric scale 
        self.hashmap_size = 2 ** self.log2_hashmap_size
        self.per_level_scale = compute_per_level_scale(
            n_levels = self.n_levels,
            base_resolution = self.base_resolution,
            finest_resolution = self.finest_resolution,
        )

        # compute resolutions (geometric spacing)
        level_resolutions = []
        for l in range(self.n_levels):
            res_l = int(math.floor(self.base_resolution * (self.per_level_scale ** l)))
            res_l = max(2, res_l)
            level_resolutions.append(res_l)
        self.register_buffer(
            'level_resolutions',
            torch.tensor(level_resolutions, dtype = torch.long),
        )
        self.logger.info(f'registered buffer level_resolutions with n_levels={self.n_levels}, N_0={self.level_resolutions[0]}, N_L={self.level_resolutions[-1]}')

        # initialize multi-res table: (hashmap_size, n_features_per_level)
        tables = []
        for _ in range(n_levels):
            table = nn.Parameter(
                torch.empty(self.hashmap_size, n_features_per_level, dtype = dtype)
            )
            # small init near zero, tcnn-like, draws from U[-1e-4, 1e-4]
            nn.init.uniform_(table, -1e-4, 1e-4)
            tables.append(table)
        self.tables = nn.ParameterList(tables)

        self.logger.info(f'Initialized multires table with (hashmap_size, n_features/lvl)={self.hashmap_size,self.n_features_per_level}')
    
    def __repr__(self):
        return (
            f'TorchHashGridEncoding('
            f'n_levels={self.n_levels}, '
            f'n_features_per_level={self.n_features_per_level}, '
            f'log2_hahmap_size={self.log2_hashmap_size}, '
            f'base_resolution={self.base_resolution}, '
            f'finest_resolution={self.finest_resolution})'
        )

    @property
    def out_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    def _interp_one_level(self, x_unit: torch.Tensor, level: int) -> torch.Tensor:
        '''
        Trilinearly interpolate features for a single level

        Inputs
        ------
        x_unit: (N, 3) in [0, 1] batch of coordinates
        level: integer level index

        Returns
        -------
        feats_l: (N, n_features_per_level)
        '''

        table = self.tables[level]
        resolution = int(self.level_resolutions[level].item())

        # map [0,1] --> [0, resolution-1]
        scaled = x_unit * float(resolution - 1)

        # lower lattice corner
        idx0 = torch.floor(scaled).long()             # (N, 3)
        frac = scaled - idx0.to(dtype = x_unit.dtype) # (N, 3)

        # upper corner, clamped to valid lattice range
        idx1 = torch.clamp(idx0 + 1, max = resolution - 1)

        # grab coordinates and fractions
        x0 = idx0[:, 0]
        y0 = idx0[:, 1]
        z0 = idx0[:, 2]

        x1 = idx1[:, 0]
        y1 = idx1[:, 1]
        z1 = idx1[:, 2]

        tx = frac[:, 0:1]  # (N,1)
        ty = frac[:, 1:2]
        tz = frac[:, 2:3]

        one = torch.ones_like(tx)

        # 8 trilinear weights, each shape (N,1)
        w000 = (one - tx) * (one - ty) * (one - tz)
        w001 = (one - tx) * (one - ty) * tz
        w010 = (one - tx) * ty * (one - tz)
        w011 = (one - tx) * ty * tz
        w100 = tx * (one - ty) * (one - tz)
        w101 = tx * (one - ty) * tz
        w110 = tx * ty * (one - tz)
        w111 = tx * ty * tz

        # hash the 8 corners
        h000 = hash_fn_3d(x0, y0, z0, self.hashmap_size)
        h001 = hash_fn_3d(x0, y0, z1, self.hashmap_size)
        h010 = hash_fn_3d(x0, y1, z0, self.hashmap_size)
        h011 = hash_fn_3d(x0, y1, z1, self.hashmap_size)
        h100 = hash_fn_3d(x1, y0, z0, self.hashmap_size)
        h101 = hash_fn_3d(x1, y0, z1, self.hashmap_size)
        h110 = hash_fn_3d(x1, y1, z0, self.hashmap_size)
        h111 = hash_fn_3d(x1, y1, z1, self.hashmap_size)

        # index into table to get features
        f000 = table[h000]
        f001 = table[h001]
        f010 = table[h010]
        f011 = table[h011]
        f100 = table[h100]
        f101 = table[h101]
        f110 = table[h110]
        f111 = table[h111]

        # weighted sum -> (N,F)
        feats_l = (
            w000 * f000 +
            w001 * f001 +
            w010 * f010 +
            w011 * f011 +
            w100 * f100 +
            w101 * f101 +
            w110 * f110 +
            w111 * f111
        )

        return feats_l
    
    def forward(self, x_unit: torch.Tensor) -> torch.Tensor:
        '''
        Inputs
        ------
        x_unit: (N, 3) in [0, 1]

        Returns
        -------
        feats: (N, out_dim)
        '''

        # tyoing and shape check
        ref_param = self.tables[0]

        if self.strict_types:
            assert x_unit.dtype == ref_param.dtype, \
                f'TorchHashGridEncoding dtype mismatch: x={x_unit.dtype}, table={ref_param.dtype}'
            assert x_unit.device == ref_param.device, \
                f'TorchHashGridEncoding device mismatch: x={x_unit.device}, table={ref_param.device}'
            x_unit = x_unit.contiguous()
        else:
            x_unit = match_ref(ref_param, x_unit)

        assert x_unit.ndim == 2 and x_unit.shape[-1] == 3, \
            f'Expected x_unit shape (N,3) but got {tuple(x_unit.shape)}'
        
        # clamp to unit cube
        x_unit = x_unit.clamp(0.0, 1.0)

        # compute features
        level_feats: List[torch.Tensor] = []
        for l in range(self.n_levels):
            level_feats.append(self._interp_one_level(x_unit, l))
        
        # concatenate features
        return torch.cat(level_feats, dim = -1)

class TCNNHashGridEncoding(nn.Module):
    '''
    tiny-cuda-nn multiresolution HashGrid wrapper

    forward
    -------
    input: (N,3) in [0,1]
    output: (N, n_levels * n_features_per_level)
    '''
    def __init__(self,
        n_input_dims: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        dtype: torch.dtype = torch.float32,
        strict_types: bool = False
    ):
        super().__init__()

        if not TCNN_AVAILABLE:
            raise ImportError("tinycudann is not available, cannot use backend='tcnn'.")

        if dtype != torch.float32:
            raise ValueError("TCNNHashGridEncoding currently requires torch.float32.")

        # attrs
        self.n_input_dims = n_input_dims
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.dtype = dtype
        self.strict_types = strict_types

        self.per_level_scale = compute_per_level_scale(
            n_levels = self.n_levels,
            base_resolution = self.base_resolution,
            finest_resolution = self.finest_resolution,
        )

        # tcnn encoding
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
        }

        self.encoding = tcnn.Encoding(
            n_input_dims = self.n_input_dims,
            encoding_config = encoding_config,
        )

    @property
    def out_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    def forward(self, x_unit: torch.Tensor) -> torch.Tensor:
        if self.strict_types:
            assert x_unit.dtype == self.dtype, \
                f"TCNNHashGridEncoding dtype mismatch: x={x_unit.dtype}, expected={self.dtype}"
            assert x_unit.is_cuda, "TCNNHashGridEncoding requires CUDA tensor input when strict_types=True"
            x_unit = x_unit.contiguous()
        else:
            # tcnn safer with float32 contiguous tensors
            x_unit = x_unit.to(dtype = self.dtype)
            x_unit = x_unit.contiguous()

        x_unit = x_unit.clamp(0.0, 1.0)
        return self.encoding(x_unit)

def build_hash_encoder(backend: str = 'torch',
                       backend_fallback: bool = True,
                       **kwargs) -> nn.Module:
    '''
    Builder for hash-grid encoder backend
    '''

    if backend == 'torch':
        return TorchHashGridEncoding(**kwargs)

    if backend == 'tcnn':
        if TCNN_AVAILABLE:
            return TCNNHashGridEncoding(**kwargs)
        if backend_fallback:
            logging.getLogger('coroNeRF.build_hash_encoder').warning(
                "tinycudann unavailable; falling back to torch hash-grid encoder."
            )
            return TorchHashGridEncoding(**kwargs)
        raise ImportError("Requested backend='tcnn' but tinycudann is unavailable.")
    
    if backend == 'auto':
        if TCNN_AVAILABLE:
            return TCNNHashGridEncoding(**kwargs)
        return TorchHashGridEncoding(**kwargs)

    raise ValueError(f"Unknown hash encoder backend: {backend}")

