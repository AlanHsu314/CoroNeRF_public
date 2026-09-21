#########################################################
### build fields (trilinear interpolators)
#########################################################

import logging
import numpy as np
import torch

from ...data.psi import get_psi_axes_np, get_psi_field_np

from ...fields.spherical import SphericalTrilinearField
from ...fields.regular import RegularTrilinearField

from ...tools.validate_fields import test_temp_field, test_ne_field, test_ccoef_field

from ...util.torch import np_to_torch, get_torch_dtype

def build_fields(state):
    '''
    setup temp and ccoef fields
    '''

    logger = logging.getLogger('coroNeRF.setup_aux_fields')
    #####################################
    ### build temperature interpolator
    #####################################

    logger.info('Building Temp Interpolator')
    build_temp_field(state)

    # unit testing temp field
    if state.cfg.temp_field.get('test', False):
        logger.info('Testing Temp Interpolator')
        test_temp_field(state)

    #####################################
    ### build ne interpolator
    #####################################
    logger.info('Building Ne Interpolator')
    build_ne_field(state)

    # unit testing temp field
    if state.cfg.ne_field.get('test', False):
        logger.info('Testing Ne Interpolator')
        test_ne_field(state)

    ######################################
    ### build ccoef interpolator
    ######################################

    logger.info('Building Ccoef Interpolator')
    build_ccoef_field(state)

    # unit testing ccoef field
    if state.cfg.ccoef_field.get('test', False):
        logger.info('Testing Ccoef Interpolator')
        test_ccoef_field(state)

def build_temp_field(state):
    '''
    construct a spherical trilinear interpolator for the temperature field used during training
    '''
    logger = logging.getLogger('coroNeRF.build_temp_field')

    # typing
    torch_dtype = get_torch_dtype(state.cfg.temp_field.get('dtype', 'float'))
    np_dtype = np.float64 if torch_dtype == torch.float64 else np.float32
    
    # load psi arrays
    lons, lats, rs = get_psi_axes_np(state.dataset_dir, dtype = np_dtype)
    T_lats = get_psi_field_np(
        state.dataset_dir,
        key = 'T',
        log = state.cfg.temp_field.get('log', True),
        dtype = np_dtype
    )

    # make float 64
    # if state.cfg.temp_field.get('dtype', 'float') in ['double', 'float64']:
    #     _to_double = lambda x: x.astype(np.float64)
    #     lons = _to_double(lons)
    #     lats = _to_double(lats)
    #     rs = _to_double(rs)
    #     T_lats = _to_double(T_lats)

    # uniform grid checks
    dr = np.diff(rs)
    dphi = np.diff(lats)
    dlon = np.diff(lons)
    logger.debug(f'dr min/max: {dr.min()} / {dr.max()} | uniform? {np.allclose(dr, dr[0])}')
    logger.debug(f'dphi min/max: {dphi.min()} / {dphi.max()} | uniform? {np.allclose(dphi, dphi[0])}')
    logger.debug(f'dlon min/max: {dlon.min()} / {dlon.max()} | uniform? {np.allclose(dlon, dlon[0])}')
    
    # some psi checks
    logger.debug(f'PSI T field shape: {T_lats.shape}')
    logger.debug(f'lons range: {lons[0], lons[-1]}') # [0, 2pi]
    #logger.debug(f'colats range: {colats[0], colats[-1]}') # [0, pi]
    logger.debug(f'lats range: {lats[0], lats[-1]}') # [-pi/2, pi/2]
    logger.debug(f'rs range: {rs[0], rs[-1]}') # [1, 30] R_sun

    # torch to numpy fn
    # if state.cfg.temp_field.get('dtype', 'float') in ['double', 'float64']:
    #     _np_to_torch = lambda tensor: torch.from_numpy(tensor).double()
    # else:
    #     _np_to_torch = lambda tensor: torch.from_numpy(tensor).float()

    # create spherical trilinear interpolator
    state.temp_field = SphericalTrilinearField(field    = np_to_torch(T_lats),
                                                lon_axis = np_to_torch(lons),
                                                phi_axis = np_to_torch(lats),
                                                r_axis   = np_to_torch(rs),
                                                strict_types = state.cfg.temp_field.get('strict_types', False))
    
    # send to device
    state.temp_field.to(state.device)
    logger.debug(f'sent temp_field to device: {state.temp_field.field.device}')

def build_ne_field(state):
    '''
    this is used for testing rendered with ground truth ne values
    construct a spherical trilinear interpolator for the ne field
    '''
    logger = logging.getLogger('coroNeRF.build_ne_field')

    # typing
    torch_dtype = get_torch_dtype(state.cfg.ne_field.get('dtype', 'float'))
    np_dtype = np.float64 if torch_dtype == torch.float64 else np.float32
    
    # load psi arrays
    lons, lats, rs = get_psi_axes_np(state.dataset_dir, dtype = np_dtype)
    ne_lats = get_psi_field_np(
        state.dataset_dir,
        key = 'ne',
        log = state.cfg.ne_field.get('log', True),
        dtype = np_dtype
    )

    # # make float 64
    # if state.cfg.ne_field.get('dtype', 'float') in ['double', 'float64']:
    #     _to_double = lambda x: x.astype(np.float64)
    #     lons = _to_double(lons)
    #     lats = _to_double(lats)
    #     rs = _to_double(rs)
    #     ne_lats = _to_double(ne_lats)

    # uniform grid checks
    dr = np.diff(rs)
    dphi = np.diff(lats)
    dlon = np.diff(lons)
    logger.debug(f'dr min/max: {dr.min()} / {dr.max()} | uniform? {np.allclose(dr, dr[0])}')
    logger.debug(f'dphi min/max: {dphi.min()} / {dphi.max()} | uniform? {np.allclose(dphi, dphi[0])}')
    logger.debug(f'dlon min/max: {dlon.min()} / {dlon.max()} | uniform? {np.allclose(dlon, dlon[0])}')
    
    # some psi checks
    logger.debug(f'PSI ne field shape: {ne_lats.shape}')
    logger.debug(f'lons range: {lons[0], lons[-1]}') # [0, 2pi]
    #logger.debug(f'colats range: {colats[0], colats[-1]}') # [0, pi]
    logger.debug(f'lats range: {lats[0], lats[-1]}') # [-pi/2, pi/2]
    logger.debug(f'rs range: {rs[0], rs[-1]}') # [1, 30] R_sun

    # # torch to numpy fn
    # if state.cfg.ne_field.get('dtype', 'float') in ['double', 'float64']:
    #     _np_to_torch = lambda tensor: torch.from_numpy(tensor).double()
    # else:
    #     _np_to_torch = lambda tensor: torch.from_numpy(tensor).float()

    # create spherical trilinear interpolator
    state.ne_field = SphericalTrilinearField(field    = np_to_torch(ne_lats),
                                             lon_axis = np_to_torch(lons),
                                             phi_axis = np_to_torch(lats),
                                             r_axis   = np_to_torch(rs),
                                             strict_types = state.cfg.ne_field.get('strict_types', False))
    
    # send to device
    state.ne_field.to(state.device)
    logger.debug(f'sent ne_field to device: {state.ne_field.field.device}')

def build_ccoef_field(state):
    '''
    given grid values for ccoef, build a LUT function for ccoef(ne, T, r)
    '''
    logger = logging.getLogger('coroNeRF.build_ccoef_field')

    # load ccoef LUT from CHIANTI
    ccoef_path = state.dataset_dir / 'ccoef_LUT.npy'
    grid_path = state.dataset_dir / 'grid_vals.npz'

    ccoef = np.load(ccoef_path, allow_pickle = False)
    grid_vals = np.load(grid_path, allow_pickle=False)
    ne_arr, temp_arr, r_arr = grid_vals['ne_arr'], grid_vals['temp_arr'], grid_vals['r_arr'] # in log

    # check ccoef dim
    if ccoef.ndim != 4:
        raise ValueError(
            f'Expected ccoef LUT with shape (C, Nx, Ny, Nz), got {ccoef.shape} from {ccoef_path}'
        )

    # slice active channels in ccoef
    total_channels = int(ccoef.shape[0])
    channel_indices = getattr(state, 'selected_channel_indices', None)
    if channel_indices is not None:
        bad = [int(i) for i in channel_indices if i < 0 or i >= total_channels]
        if bad:
            raise IndexError(
                f'channel indices = {channel_indices} out of bounds for cooef LUT with {total_channels} channels'
            )
        ccoef = ccoef[channel_indices, ...]

    active_channels = int(ccoef.shape[0])

    # check number of dataset channels match active ccoef channels
    dataset_channels = getattr(state, 'dataset_num_channels', None)
    if (dataset_channels is not None) and (int(dataset_channels) != active_channels):
        raise ValueError(
            f'Loaded images have {dataset_channels} channels, but ccoef LUT has {active_channels} '
            f'channels after selection. Check dataset_dir / LUT pairing in config.'
        )
    
    # do another check on all config channel-related kwargs
    # we check these when loading dataset, but check them again just in case
    def _check_cfg_channel_list(name, values):
        if values and (len(values) != active_channels):
            raise ValueError(
                f'{name} has length {len(values)}, but active ccoef LUT has {active_channels} channels'
            )
        
    _check_cfg_channel_list('renderer.wavelengths', list(state.cfg.renderer.get('wavelengths', [])))
    _check_cfg_channel_list('ion_kwargs.wavelengths', list(state.cfg.get('ion_kwargs', {}).get('wavelengths', [])))
    _check_cfg_channel_list('ion_kwargs.line_names', list(state.cfg.get('ion_kwargs', {}).get('line_names', [])))

    # add ccoef LUT bounds for renderer later
    state.cfg.renderer['log_ne_min'] = ne_arr[0]
    state.cfg.renderer['log_ne_max'] = ne_arr[-1]
    state.cfg.renderer['log_temp_min'] = temp_arr[0]
    state.cfg.renderer['log_temp_max'] = temp_arr[-1]
    state.cfg.renderer['log_r_min'] = r_arr[0]
    state.cfg.renderer['log_r_max'] = r_arr[-1]

    # some CHIANTI checks
    logger.debug(f'ccoef field shape: {ccoef.shape}')
    logger.debug(f'ne range: {ne_arr[0], ne_arr[-1]}') # log ne ~(2, 12)
    logger.debug(f'temp range: {temp_arr[0], temp_arr[-1]}') # log temp ~(5, 6)
    logger.debug(f'r range: {r_arr[0], r_arr[-1]}') # log r ~(1e-3, 1)

    # torch to numpy fn
    if state.cfg.ccoef_field.get('dtype', 'float') in ['double', 'float64']:
        _np_to_torch = lambda tensor: torch.from_numpy(tensor).double()
    else:
        _np_to_torch = lambda tensor: torch.from_numpy(tensor).float()

    # create trilinear interpolator
    state.ccoef_field = RegularTrilinearField(
        field  = _np_to_torch(ccoef),
        x_axis = _np_to_torch(ne_arr),
        y_axis = _np_to_torch(temp_arr),
        z_axis = _np_to_torch(r_arr),
        strict_types = state.cfg.ccoef_field.get('strict_types', False),
        squeeze_single_channel = False,
    )
    
    
    # send to device
    state.ccoef_field.to(state.device)

    logger.debug(f'sent ccoef_field to device: {state.ccoef_field.field.device}')



