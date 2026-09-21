###############################################################
### Multiresolution hash grid model for generic plasma fields
###############################################################

import torch
import torch.nn as nn
import logging


from .encoding import spherical_harmonics_encoding
from .hash_encoding import build_hash_encoder
from ..util.torch import match_ref


def _make_mlp(in_features: int, hidden_dim: int, num_hidden: int, out_features: int) -> nn.Sequential:
    '''
    small helper for generic MLP
    '''
    dims = [in_features] + [hidden_dim] * num_hidden + [out_features]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)

class PlasmaHashField(nn.Module):
    '''
    Shared spatial hash encoder + optional shared trunk + separate heads
    Updated from original denisty_hash_mlp.py

    Input:
        x_AABB in [-1, 1]^3

    Output:
        target = 'ne'   --> (..., 1)
        target = 'ne_t  --> (..., 2)
    '''
    def __init__(
        self,
        backend: str = 'torch',
        backend_fallback: bool = True,
        in_dim: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        dtype: torch.dtype = torch.float32,
        strict_types: bool = False,
        use_raw_xyz: bool = False,
        use_sh_encoding: bool = False,
        sh_degree: int = 2,
        shared_hidden_dim: int = 64,
        shared_num_hidden: int = 0,
        head_hidden_dim: int = 64,
        head_num_hidden: int = 2,
        target: str = 'ne',
    ):
        super().__init__()

        # attrs
        self.dtype = dtype
        self.strict_types = strict_types
        self.use_raw_xyz = use_raw_xyz
        self.use_sh_encoding = use_sh_encoding
        self.sh_degree = sh_degree
        self.target = target
        self.logger = logging.getLogger('coroNeRF.PlasmaHashField')

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

        # concat features from hash table
        feat_dim = self.encoder.out_dim

        # add optional features
        if self.use_raw_xyz:
            feat_dim += 3
        
        if self.use_sh_encoding:
            self.sh_dim = (self.sh_degree + 1) ** 2
            feat_dim += self.sh_dim
        else:
            self.sh_dim = 0

        # make optional shared trunk
        if shared_num_hidden > 0:
            self.shared_trunk = _make_mlp(
                in_features = feat_dim,
                hidden_dim = shared_hidden_dim,
                num_hidden = shared_num_hidden,
                out_features = shared_hidden_dim,
            )
            trunk_dim = shared_hidden_dim
        else:
            self.shared_trunk = None
            trunk_dim = feat_dim

        # density head decoder
        self.head_ne = _make_mlp(
            in_features = trunk_dim,
            hidden_dim = head_hidden_dim,
            num_hidden = head_num_hidden,
            out_features = 1,
        )

        # temp head decoder
        if self.target == 'ne_t':
            self.head_temp = _make_mlp(
                in_features = trunk_dim,
                hidden_dim = head_hidden_dim,
                num_hidden = head_num_hidden,
                out_features = 1,
            )
        else:
            self.head_temp = None

        self.logger.debug(f'initialized PlasmaHashField with target {target}')
        
        # typing
        self.to(dtype)

    def initialize_density_bias(self, value: float):
        '''
        used by renderer / setup to bias the initial density estimate
        '''
        with torch.no_grad():
            last = self.head_ne[-1]
            if hasattr(last, 'bias') and last.bias is not None:
                last.bias.fill_(float(value))

    def initialize_temperature_bias(self, value: float):
        '''
        used by renderer / setup to bias the initial temperature estimate
        '''
        if self.head_temp is None:
            raise ValueError('tried to initialize temp bias, but PlasmaHashField does not predict temp')
        with torch.no_grad():
            last = self.head_temp[-1]
            if hasattr(last, 'bias') and last.bias is not None:
                last.bias.fill_(float(value))

    def _prepare_features(self, x_AABB: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        '''
        forward pass helper through hash grid and shared trunk
        '''

        # strict types
        ref_param = next(self.head_ne.parameters())

        if self.strict_types:
            assert x_AABB.dtype == ref_param.dtype, \
                f'PlasmaHashField dtype mismatch: x={x_AABB.dtype}, params={ref_param.dtype}'
            assert x_AABB.device == ref_param.device, \
                f'PlasmaHashField device mismatch: x={x_AABB.device}, params={ref_param.device}'
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
            r = torch.linalg.norm(x_flat, dim = -1, keepdim = True)
            u = x_flat / (r + 1e-9)

            if self.strict_types:
                assert u.dtype == ref_param.dtype and u.device == ref_param.device
                u = u.contiguous()
            else:
                u = match_ref(ref_param, u)

            dir_enc = spherical_harmonics_encoding(u, degree = self.sh_degree)
            feat_list.append(dir_enc)

        # concatenate features
        feats = torch.cat(feat_list, dim = -1)

        # feed into optional shared trunk
        if self.shared_trunk is not None:
            feats = self.shared_trunk(feats)

        return feats, orig_shape

    def forward(self, x_AABB: torch.Tensor) -> torch.Tensor:
        
        # grab features from has grid, optional shared trunk
        feats, orig_shape = self._prepare_features(x_AABB)

        # ne forward
        log_ne = self.head_ne(feats)

        # if also predicting temp, concatenate
        if self.target == 'ne':
            out = log_ne
        elif self.target == 'ne_t':
            log_temp = self.head_temp(feats)
            out = torch.cat([log_ne, log_temp], dim = -1)
        else:
            raise ValueError(f'Unknown target: {self.target}')
        
        return out.reshape(*orig_shape, out.shape[-1])






