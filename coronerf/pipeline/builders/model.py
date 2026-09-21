##################################################
### building model
##################################################

import logging
import numpy as np
import torch

from ...models.tests.test_encoding import test_posenc
from ...models.tests.test_density_mlp import test_mlp
from ...models.tests.test_density_hash_mlp import test_density_hash_mlp

from ...util.torch import get_torch_dtype

from ...models.density_mlp import DensityMLP
from ...models.density_spherical_grid import DensitySphericalGrid
from ...models.density_hash_mlp import DensityHashMLP
from ...models.plasma_hash_field import PlasmaHashField
from ...models.plasma_mlp import PlasmaMLP

from ...data.psi import get_psi_axes_torch


def build_model(state):
    logger = logging.getLogger('coroNeRF.setup_model')

    # unit testing positional encoder
    if state.cfg.model.get('test_pe', False):
        logger.info('Testing Positional Encoder Class')
        test_posenc(state)

    # get model params
    model_dtype = get_torch_dtype(state.cfg.model.get('dtype', 'float'))
    model_name = state.cfg.model.get('name', 'density_mlp')
    target = state.shared.get('reconstruction_target', 'ne')

    ######################
    # case
    ######################
    if model_name == 'density_mlp':
        # unit test model
        if state.cfg.model.get('test_model', False):
            logger.info('Testing Model Class')
            test_mlp(state)
        
        # define instance of model
        state.model = DensityMLP(**state.cfg.DensityMLP_kwargs, dtype = model_dtype)
        logger.debug('created DensityMLP model')
    elif model_name == 'density_spherical_grid':
        state.model = build_density_spherical_grid(state, model_dtype)
        logger.debug('created DensitySphericalGrid model')
    elif model_name == 'density_hash_mlp':
        if state.cfg.model.get('test_model', False):
            logger.info('Testing DensityHashMLP')
            test_density_hash_mlp(state)
        
        state.model = DensityHashMLP(**state.cfg.DensityHashMLP_kwargs, dtype = model_dtype)
        logger.debug('created DensityHashMLP model')
    elif model_name == 'plasma_mlp':
        raise NotImplementedError(f'PlasmaMLP not implemented yet')
    elif model_name == 'plasma_hash_field':
        kwargs = dict(state.cfg.PlasmaField_kwargs)
        state.model = PlasmaHashField(
            **kwargs,
            dtype = model_dtype,
            strict_types = state.cfg.model.get('strict_types', True),
            target = target,
        )
        logger.debug('created PlasmaHashField model')
    else:
        raise ValueError(f'Unknown model.name: {model_name}')

    # send to device
    state.model.to(state.device)
    logger.debug(f'sent model {model_name} to device: {state.device}')

    # match dataloader dtype and device with model!!
    # for vpgen, there is no data to match
    if state.cfg.get('mode', 'train') in ['train', 'resume', 'regen']:
        match_data_with_model(state, model_dtype)

def build_density_spherical_grid(state, model_dtype):
    '''
    builds trainable spherical density grid
    uses PSI psherical axes
    '''
    logger = logging.getLogger('coroNeRF.build_density_spherical_grid')

    # load psi axes
    lons, lats, rs = get_psi_axes_torch(state.dataset_dir, torch_dtype = model_dtype)

    # grab field kwargs
    kwargs = dict(state.cfg.DensitySphericalGrid_kwargs)
    #kwargs['aabb_scale'] = state.aabb_scale

    # instantiate model
    model = DensitySphericalGrid(
        lon_axis = lons,
        phi_axis = lats,
        r_axis = rs,
        dtype = model_dtype,
        strict_types = state.cfg.model.get('strict_types', True),
        shared_context = state.shared,
        **kwargs,
    )

    return model

def match_data_with_model(state, model_dtype):
    '''
    check dtype and device for data and match with model

    Notes:
    --we load to device per batch during training
    --so really only thing to check is dtype
    '''
    for split_name in ["train_data", "test_data"]:
        split = getattr(state, split_name)
        split = {k: (v.to(dtype=model_dtype) if v.is_floating_point() else v)
                for k, v in split.items()}
        setattr(state, split_name, split)


