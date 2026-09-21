################################################################
### loading datasets
################################################################

import logging
from typing import Union
import numpy as np
import torch

from ...data.dataset import CoronaDataset
from ...util.torch import to_numpy

# mismatch
from ...data.mismatch import (
    apply_observation_mismatch_images,
    combine_observation_mismatch_metadata,
    log_observation_mismatch_diagnostics,
    save_mismatch_json,
)

# noise
from ...data.noise import (
    apply_observation_noise_flat,
    log_observation_noise_diagnostics,
    save_noise_json,
)

# general checks

def check_rays(data_dict):
    '''
    Checks if rays_d are all normalized
    data_dict is tensor on GPU
    '''
    logger = logging.getLogger('coroNeRF.check_rays')

    rays_d = data_dict['rays_d']

    # only rays_d normalization check done here.
    norms = torch.linalg.norm(rays_d, dim = -1)
    logger.info(f'max deviation of rays from norm: {np.abs(to_numpy(norms) - 1).max()}')

    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)

    logger.debug('All rays are normalized to 1e-5 within 1')

# channel indices

def _normalize_channel_indices(channel_indices):
    '''
    normalizes user-provided channel indices to a list[int] or None
    '''
    # none
    if channel_indices is None:
        return None
    
    # np -> list
    if isinstance(channel_indices, np.ndarray):
        channel_indices = channel_indices.tolist()

    # otherwise
    if not isinstance(channel_indices, (list, tuple)):
        raise TypeError(f'channel_indices must be None, list or tuple, got {type(channel_indices)}')
    
    # list, tuple -> list
    out = [int(i) for i in channel_indices]

    # final checks
    if len(out) == 0:
        raise ValueError('channel_indices cannot be empty')
    if len(set(out)) != len(out):
        raise ValueError(f'channel_indices contains duplicates: {out}')
    if any(i < 0 for i in out):
        raise ValueError(f'channel_indices must be non-negative, got {out}')
    
    return out

def _validate_channel_indices_against_count(channel_indices, n_channels: int, where: str):
    '''
    IndexError checks on channel_indices,  
    where: which config value we are checking
    '''
    if channel_indices is None:
        return
    # non-valid indices
    bad = [int(i) for i in channel_indices if i < 0 or i >= n_channels]

    if bad:
        raise IndexError(
            f'{where}: requested channel_indices = {channel_indices}, '
            f'but available channels are [0, {n_channels - 1}]'
        )

def _apply_channel_selection_to_cfg(state, channel_indices):
    '''
    slice channel-related config lists in memory so renderer/plots stay consistent
    with the selected image channels loaded from disk
    '''
    if channel_indices is None:
        state.selected_wavelengths = list(state.cfg.renderer.get('wavelengths', []))
        state.selected_line_names = list(state.cfg.get('ion_kwargs', {}).get('line_names', []))
        return
    
    # renderer wavelengths
    renderer_waves = list(state.cfg.renderer.get('wavelengths', []))
    if renderer_waves:
        _validate_channel_indices_against_count(channel_indices, len(renderer_waves), 'renderer.wavelengths')
        state.cfg.renderer['wavelengths'] = [renderer_waves[i] for i in channel_indices]
    state.selected_wavelengths = list(state.cfg.renderer.get('wavelengths', []))

    # ion wavelengths
    ion_waves = list(state.cfg.get('ion_kwargs', {}).get('wavelengths', []))
    if ion_waves:
        _validate_channel_indices_against_count(channel_indices, len(ion_waves), 'ion_kwargs.wavelengths')
        state.cfg.ion_kwargs['wavelengths'] = [ion_waves[i] for i in channel_indices]

    # line names
    line_names = list(state.cfg.get('ion_kwargs', {}).get('line_names', []))
    if line_names:
        _validate_channel_indices_against_count(channel_indices, len(line_names), 'ion_kwargs.line_names')
        state.cfg.ion_kwargs['line_names'] = [line_names[i] for i in channel_indices]
    state.selected_line_names = list(state.cfg.get('ion_kwargs', {}).get('line_names', []))
    
# selecting viewpoints from datasets

def _count_viewpoints(vp_dir) -> int:
    '''
    count # viewpoints in vp_dir
    '''
    files = sorted(vp_dir.glob('vp_*.npz'))
    if not files:
        raise FileNotFoundError(f'No viewpoint files found in {vp_dir}')
    return len(files)

def _evenly_spaced_indices(n_total: int, n_select: int, offset: int = 0) -> np.ndarray:
    '''
    n_total: total amt of vp
    n_select: how many we want
    offset: offset from mod 0
    '''

    # extra cases
    if n_select <= 0:
        return np.array([], dtype = int)
    if n_select >= n_total:
        return np.arange(n_total, dtype = int)
    
    # get raw indices
    xs = np.linspace(0, n_total, num = n_select, endpoint = False)
    idx = (np.floor(xs).astype(int) + int(offset)) % n_total
    idx = np.unique(idx)

    # just in case
    if len(idx) < n_select:
        used = set(idx.tolist())
        missing = [i for i in range(n_total) if i not in used]
        idx = np.concatenate([idx, np.array(missing[:(n_select - len(idx))], dtype = int)])

    return np.sort(idx.astype(int))

def _complement_indices(n_total: int, idx: np.ndarray) -> np.ndarray:
    '''
    the indices that are not in idx
    '''
    mask = np.ones(n_total, dtype = bool) # mask for indices not used
    mask[idx] = False
    return np.arange(n_total, dtype = int)[mask]

def _pick_vp_dir(state):
    '''
    function to deal with legacy datasets (or datasets not generate in nerf pipeline)
    '''
    logger = logging.getLogger('coroNeRF.load_data')

    if state.mode in ['train', 'resume']:
        vp_dir = state.dataset_dir / 'viewpoints_gt'
        expected = 'viewpoints_gt'
    else: # regen / debug etc. -> use original viewpoints
        vp_dir = state.dataset_dir / 'viewpoints'
        expected = 'viewpoints'

    if not vp_dir.exists():
        msg = (
            f"Expected directory '{expected}' at {vp_dir}, but it does not exist.\n"
            f"If this is a NEW dataset, you must first run with mode='regen' to "
            f"regenerate viewpoints using the current forward model, e.g.\n\n"
            f"    mode: regen\n\n"
            f"in your config. After that completes, switch mode back to 'train'."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)
    
    return vp_dir
    
def _build_split_indices(state, 
                         train_type,
                         train_num_views = None,
                         train_subset_strategy = 'evenly_spaced',
                         train_subset_offset = 0,
                         holdout_type = 'fixed_count',
                         holdout_num_views = 30,
                         holdout_subset_strategy = 'evenly_spaced',
                         holdout_subset_offset = 1,
                         test_stride = 30,
                         arc = None,
                         ):
    '''
    (1) gets vp_dir
    (2) train_idx
    (3) test_idx
    '''
    logger = logging.getLogger('coroNeRF.load_data')

    # get and analyze vp dir
    vp_dir = _pick_vp_dir(state)
    n_total = _count_viewpoints(vp_dir)
    logger.info(f'found {n_total} viewpoints in {vp_dir}')

    # legacy cases 'small' and 'full
    if train_type == 'small':
        train_idx = np.arange(min(15, n_total), dtype = int)
        test_idx = np.array([0], dtype = int)
    elif train_type == 'full':
        train_idx = np.arange(n_total, dtype = int)
        test_idx = np.arange(0, n_total, test_stride, dtype = int)
    # new split cases for efficient benchmarking
    elif train_type == 'subset':

        # checks
        if train_subset_strategy != 'evenly_spaced':
            raise NotImplementedError(f'Unknown train_subset_strategy = {train_subset_strategy}')
        if holdout_subset_strategy != 'evenly_spaced':
            raise NotImplementedError(f'Unknown holdout_subset_strategy = {holdout_subset_strategy}')
        
        # get train idx
        train_idx = _evenly_spaced_indices(n_total = n_total, n_select = train_num_views, offset = train_subset_offset)

        # case on holdout type
        if holdout_type == 'fixed_count':
            raw_test_idx = _evenly_spaced_indices(
                n_total = n_total,
                n_select = holdout_num_views,
                offset = holdout_subset_offset,
            )

            # 1st safeguard
            train_set = set(train_idx.tolist())
            test_idx = np.array([i for i in raw_test_idx if i not in train_set])

            # 2nd safeguard
            if len(test_idx) < holdout_num_views:
                remaining = _complement_indices(n_total, train_idx)
                refill_local = _evenly_spaced_indices(
                    n_total = len(remaining),
                    n_select = holdout_num_views,
                    offset = 0,
                )
                test_idx = remaining[refill_local]
        elif holdout_type == 'complement':
            test_idx = _complement_indices(n_total, train_idx)
        elif holdout_type == 'arc':
            # Leave a contiguous longitude arc out for held-out views (novel-view + limited-angle).
            # Train on evenly-spaced views from the complement, optionally with a guard buffer.
            a = dict(arc or {})
            lo = float(a.get('lo_deg', 0.0)); hi = float(a.get('hi_deg', 30.0))
            buf = float(a.get('train_buffer_deg', 0.0))
            deg = np.arange(n_total) * (360.0 / n_total)          # longitude of each view index
            def _in_arc(d_lo, d_hi):
                dl, dh = d_lo % 360.0, d_hi % 360.0
                return ((deg >= dl) & (deg < dh)) if dl <= dh else ((deg >= dl) | (deg < dh))
            holdout_mask = _in_arc(lo, hi)                         # views inside the held-out arc
            train_excl   = _in_arc(lo - buf, hi + buf)            # arc (+ buffer) excluded from training
            arc_idx = np.where(holdout_mask)[0]
            if a.get('holdout_stride_deg') is not None:            # held-out sampled every N degrees
                step = max(1, int(round(float(a['holdout_stride_deg']) / (360.0 / n_total))))
                test_idx = arc_idx[::step]
            else:                                                  # ...or a fixed count across the arc
                k = int(a.get('holdout_num_views', holdout_num_views))
                test_idx = arc_idx[_evenly_spaced_indices(len(arc_idx), k, 0)]
            train_pool = np.where(~train_excl)[0]                 # evenly-spaced train from the complement
            train_idx = train_pool[_evenly_spaced_indices(len(train_pool),
                                                          int(train_num_views), int(train_subset_offset))]
        else:
            raise NotImplementedError(f'Unknown holdout_type = {holdout_type}')
    
    else:
        raise NotImplementedError(f'Unknown train_type = {train_type}')

    # check idx, log, and return
    if len(train_idx) == 0:
        raise ValueError('train_idx is empty')
    if len(test_idx) == 0:
        raise ValueError('test_idx is empty')
    
    logger.info(f'train views: {len(train_idx)} | test views: {len(test_idx)}')
    return vp_dir, train_idx, test_idx

def _load_view_dict(vp_dir, mask_path, idx_arr, channel_indices = None):
    '''
    construct data dict for views

    also does channel slicing
    '''
    # load mask and dims
    mask = np.load(mask_path)
    first = np.load(vp_dir / f'vp_{int(idx_arr[0])}.npz', allow_pickle = False)
    first_intensities = first['intensities']

    # check intensities
    raw_num_channels = int(first_intensities.shape[-1])
    _validate_channel_indices_against_count(
        channel_indices, 
        raw_num_channels,
        f'viewpoints in {vp_dir}',
    )

    # channel slicing
    if channel_indices is not None:
        first_intensities = first_intensities[..., channel_indices]
    
    H, W, C = first_intensities.shape

    # create data arrays
    num_vp = len(idx_arr)
    imgs = np.zeros((num_vp, H, W, C), dtype = np.float32)
    rays_o = np.zeros((num_vp, 3), dtype = np.float32)
    rays_d = np.zeros((num_vp, H, W, 3), dtype = np.float32)

    # load data into arrays
    for arr_i, idx in enumerate(idx_arr):
        # grab view
        vp_data = np.load(vp_dir / f'vp_{int(idx)}.npz', allow_pickle = False)
        
        # get sliced image
        vp_intensities = vp_data['intensities']
        if channel_indices is not None:
            vp_intensities = vp_intensities[..., channel_indices]
        imgs[arr_i,:,:,:] = vp_intensities

        # get rays
        rays_o[arr_i,:] = vp_data['rays_o']
        rays_d[arr_i,:,:,:] = vp_data['rays_d']

    # create saving dict (generic float dtype, will match with model when its loaded)
    return {
        'imgs': torch.from_numpy(imgs).float(),
        'mask': torch.from_numpy(mask).bool(),
        'rays_o': torch.from_numpy(rays_o).float(),
        'rays_d': torch.from_numpy(rays_d).float(),
    }

# actual load function

def load_data(state, 
              train_type: str = 'full',
              train_num_views: Union[int, None] = None,
              train_subset_strategy: str = 'evenly_spaced',
              train_subset_offset: int = 0,
              holdout_type: str = 'fixed_count',
              holdout_num_views: int = 30,
              holdout_subset_strategy: str = 'evenly_spaced',
              holdout_subset_offset: int = 1,
              test_stride: int = 30,
              arc = None,
              channel_indices = None,
              ):
    '''
    load in train and test split of viewpoint data, in torch tensors in FLOAT32, no dtype checks yet, done when model is loaded in

    INPUTS
    [str] train_type: 'small' for a small fixed hardcoded set of viewpoints

    OUTPUTS
    [dict] train_data: {imgs, mask, rays_o, rays_d}
    [dict] test_data:  {imgs, mask, rays_o, rays_d}
    '''
    logger = logging.getLogger('coroNeRF.load_data')
    logger.debug('loading data...')

    ###############################
    # channel normalization
    ###############################

    # normalize channel indices
    channel_indices = _normalize_channel_indices(channel_indices)
    state.selected_channel_indices = channel_indices

    # apply selected channels to rest of config
    _apply_channel_selection_to_cfg(state, channel_indices)

    ###############################
    # train test split
    ###############################

    # train and test idx
    vp_dir, train_idx, test_idx = _build_split_indices(
        state, 
        train_type = train_type,
        train_num_views = train_num_views,
        train_subset_strategy = train_subset_strategy,
        train_subset_offset = train_subset_offset,
        holdout_type = holdout_type,
        holdout_num_views = holdout_num_views,
        holdout_subset_strategy = holdout_subset_strategy,
        holdout_subset_offset = holdout_subset_offset,
        test_stride = test_stride,
        arc = arc,
    )
    logger.debug(f'first 10 train idx: {train_idx[:min(10, len(train_idx))]}')
    logger.debug(f'first 5 test idx: {test_idx[:min(5, len(test_idx))]}')

    ###############################
    # create raw train test data
    ###############################

    # train and test data
    mask_path = state.dataset_dir / 'mask.npy'
    state.train_data = _load_view_dict(vp_dir, mask_path, train_idx, channel_indices = channel_indices)
    state.test_data = _load_view_dict(vp_dir, mask_path, test_idx, channel_indices = channel_indices)
    logger.debug('done loading data!')

    # apply model mismatch before dataset flattening
    # right now its just abundance mismatch, applied to ALL views
    mismatch_split_meta = {} # per-dataset
    for split_name, data_dict in [('train', state.train_data), ('test', state.test_data)]:
        mismatched_imgs, split_meta = apply_observation_mismatch_images(
            imgs = data_dict['imgs'],
            cfg = state.cfg,
            split = split_name,
            global_channel_ids = state.selected_channel_indices,
            line_names = state.selected_line_names,
            mask_2d = data_dict['mask'],
        )
        if split_meta is not None:
            data_dict['imgs'] = mismatched_imgs
            mismatch_split_meta[split_name] = split_meta

    ###############################
    # add model mismatch effects
    ###############################

    # compute mismatch diagnostics, combined train and test
    mismatch_meta = combine_observation_mismatch_metadata(mismatch_split_meta, cfg = state.cfg)

    # log mismatch
    if mismatch_meta is not None:
        state.observation_mismatch_meta = mismatch_meta
        log_observation_mismatch_diagnostics(mismatch_meta, logger = logger)

        if not state.no_save:
            mismatch_json_path = state.dirs.artifacts / 'data' / 'observation_mismatch.json'
            save_mismatch_json(mismatch_meta, mismatch_json_path)
            logger.info(f'Saved observation-mismatch diagnostics to {mismatch_json_path}')

    ################################
    # raw train test metadata
    ################################

    # save some metadata
    train_imgs = state.train_data['imgs']

    state.dataset_num_viewpoints = _count_viewpoints(vp_dir)
    state.train_num_viewpoints_loaded = len(train_idx)
    state.test_num_viewpoints_loaded = len(test_idx)

    state.dataset_image_H = int(train_imgs.shape[1])
    state.dataset_image_W = int(train_imgs.shape[2])
    state.dataset_num_channels = int(train_imgs.shape[3])

    # some checks

    # printing shapes
    for key in state.train_data:
        logger.debug(f'{key}: train shape {state.train_data[key].shape}')
        logger.debug(f'{key}: test shape {state.test_data[key].shape}')

    if state.debug:
        logger.debug('Checking normalization of rays')
        check_rays(state.train_data)
        check_rays(state.test_data)

    ###############################
    # load into dataset
    ###############################

    # load into datasetclass
    logger.info('Building Training Dataset')
    state.ds_train = CoronaDataset(**state.train_data)
    state.ds_test = CoronaDataset(**state.test_data)

    ###############################
    # appply observation noise
    ###############################

    # inject observation noise ONLY into the flattened train targets, NOT test targets
    # do this AFTER model mismatch has been applied
    
    #seed_offset = int(state.cfg.get('observation_noise', {}).get('seed_offset', 0xFACADE))
    #noise_seed = int(state.cfg.get('seed', 0)) + seed_offset

    _obs_noise = state.cfg.get('observation_noise', {}) or {}
    _fixed_noise_seed = _obs_noise.get('fixed_noise_seed', None)
    if _fixed_noise_seed is not None:
        # Deployment-faithful epistemic ensemble: every model seed trains on the SAME
        # noisy observation, so cross-seed disagreement reflects model/optimization
        # uncertainty only (the quantity reproducible on a single real observation).
        noise_seed = int(_fixed_noise_seed)
    else:
        seed_offset = int(_obs_noise.get('seed_offset', 0xFACADE))
        noise_seed = int(state.cfg.get('seed', 0)) + seed_offset


    noisy_flat, sigma_flat, noise_meta = apply_observation_noise_flat(
        clean_flat = state.ds_train.imgs_flat,
        cfg = state.cfg,
        global_channel_ids = state.selected_channel_indices,
        line_names = state.selected_line_names,
        seed = noise_seed,
        mask_2d=state.train_data['mask'],
        pix_idx=state.ds_train.pix_idx,
        image_hw=(
            int(state.train_data['imgs'].shape[1]),
            int(state.train_data['imgs'].shape[2]),
        ),
    )

    if noise_meta is not None:
        # replace with poisson noise
        state.ds_train.replace_targets(noisy_flat)

        # replace with gaussina noise
        if sigma_flat is not None:
            state.ds_train.replace_target_sigma(sigma_flat)

        state.observation_noise_meta = noise_meta
        log_observation_noise_diagnostics(noise_meta, logger = logger)

        if not state.no_save:
            noise_json_path = state.dirs.artifacts / 'train' / 'observation_noise.json'
            save_noise_json(noise_meta, noise_json_path)
            logger.info(f'Saved observation-noise diagnostics to {noise_json_path}')




