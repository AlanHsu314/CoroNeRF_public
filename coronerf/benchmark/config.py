#############################################
# GENERAL UTILITY OF CONFIGS FOR BENCHMARKING
#############################################

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Union
import json
import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open('r') as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f'Expected YAML dict at {path}, got {type(data)}')
    return data

def save_yaml(data: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok = True, parents = True)
    with path.open('w') as f:
        yaml.safe_dump(data, f, sort_keys = False)

def save_json(data: Union[dict, list], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok = True, parents = True)
    with path.open('w') as f:
        json.dump(data, f, indent = 2)

def load_json(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open('r') as f:
        return json.load(f)

def deep_copy_cfg(cfg: dict) -> dict:
    return deepcopy(cfg)

def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    '''
    set value to a key somewhere in the nested cfg

    Example: set_nested(Cfg, 'train.max_steps', 100)
    '''

    parts = dotted_key.split('.')
    if not parts:
        raise ValueError(f'Invalid dotted key: {dotted_key}')
    
    cur = cfg # current dict level, start at top
    # descend up until second to last layer
    for part in parts[:-1]: 
        if part not in cur or cur[part] is None:
            cur[part] = {} # add key, and value = {}
        if not isinstance(cur[part], dict): 
            raise TypeError(
                f'Cannot descend into non-dict key "{part}" while setting "{dotted_key}"'
            )
        cur = cur[part] # descend

    # assign value
    cur[parts[-1]] = value
    

def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    '''
    generalized function to apply overrides to config
    '''
    out = deep_copy_cfg(cfg)
    for key, value in overrides.items():
        set_nested(out, key, value)
    return out

def normalize_config(cfg: dict) -> dict:
    '''
    normalize entries across the config
    (for example aabb_scale)

    and ensure some benchmark entries exist
    '''
    
    out = deep_copy_cfg(cfg)
    
    # canonicalize aabb_scale
    scene = out.get('scene', {})

    out.setdefault('shared', {})
    if 'aabb_scale' in scene:
        out['shared']['aabb_scale'] = scene['aabb_scale']
    if 'aabb_min' in scene:
        out['shared']['aabb_min'] = scene['aabb_min']
    if 'aabb_max' in scene:
        out['shared']['aabb_max'] = scene['aabb_max']
    
    # reconstruction task defaults
    out.setdefault('reconstruction', {})
    out['reconstruction'].setdefault('target', 'ne') # ne | ne_t
    
    # add target to shared context
    out['shared']['reconstruction_target'] = out['reconstruction']['target']

    # ensure benchmark metadata exists
    out.setdefault('benchmark_meta', {})
    out['benchmark_meta'].setdefault('benchmark_name', None)
    out['benchmark_meta'].setdefault('experiment_name', None)
    out['benchmark_meta'].setdefault('run_name', None)

    # ensure train dict exists and has safe default for nerfacc
    out.setdefault('train', {})
    out['train'].setdefault('occ_update_every', 16)

    # image loss
    out['train'].setdefault('image_loss', {})
    out['train']['image_loss'].setdefault('name', 'log_ratio_cosh')     # default to log cosh ratio loss
    out['train']['image_loss'].setdefault('lambda_gauss', 1.0)          # hetero-gaussian + ashin loss
    out['train']['image_loss'].setdefault('lambda_asinh', 0.1)
    out['train']['image_loss'].setdefault('use_huber_asinh', False)
    out['train']['image_loss'].setdefault('huber_delta', 0.05)
    out['train']['image_loss'].setdefault('asinh_scale_mode', 'auto')   # auto | target_sigma | fixed
    out['train']['image_loss'].setdefault('asinh_scale', 1.0)
    out['train']['image_loss'].setdefault('asinh_scale_by_channel', None)

    # val loss
    out['train'].setdefault('validation_loss', {})
    out['train']['validation_loss'].setdefault('enabled', False)
    out['train']['validation_loss'].setdefault('every', None)          # None -> eval_every/loss_every fallback
    out['train']['validation_loss'].setdefault('batch_rays', 4096)
    out['train']['validation_loss'].setdefault('max_batches', None)    # None -> full heldout set
    out['train']['validation_loss'].setdefault('num_workers', 0)
    out['train']['validation_loss'].setdefault('image_loss', None)     # optional override of train.image_loss

    # observation/model mismatch defaults
    out.setdefault('observation_mismatch', {})
    out['observation_mismatch'].setdefault('enabled', False)
    out['observation_mismatch'].setdefault('apply_to', 'all')        # all | train | test
    out['observation_mismatch'].setdefault('model', 'channel_scale') # channel_scale (effective abundance mismatch)

    out['observation_mismatch'].setdefault('diagnostics', {})
    out['observation_mismatch']['diagnostics'].setdefault('enabled', True)
    out['observation_mismatch']['diagnostics'].setdefault('positive_eps', 1.0e-12)
    out['observation_mismatch']['diagnostics'].setdefault('percentiles', [10.0, 50.0, 90.0])
    out['observation_mismatch']['diagnostics'].setdefault('max_quantile_samples', 200000)
    out['observation_mismatch']['diagnostics'].setdefault('radial_bins', [
        {'name': 'low_corona', 'r_min': 1.01, 'r_max': 1.5},
        {'name': 'inner', 'r_min': 1.01, 'r_max': 2.0},
        {'name': 'mid_outer_inner', 'r_min': 1.5, 'r_max': 2.0},
    ])

    out['observation_mismatch'].setdefault('channel_scale', {})
    out['observation_mismatch']['channel_scale'].setdefault('scale', None)
    out['observation_mismatch']['channel_scale'].setdefault('scale_by_global_channel', None)


    # observation noise defaults
    out.setdefault('observation_noise', {})
    out['observation_noise'].setdefault('enabled', False)
    out['observation_noise'].setdefault('apply_to', 'train')
    out['observation_noise'].setdefault('model', 'effective_shot')
    out['observation_noise'].setdefault('seed_offset', 0xFACADE)

    out['observation_noise'].setdefault('diagnostics', {})
    out['observation_noise']['diagnostics'].setdefault('enabled', True)
    out['observation_noise']['diagnostics'].setdefault('positive_eps', 1.0e-12)
    out['observation_noise']['diagnostics'].setdefault('percentiles', [10.0, 50.0, 90.0])
    out['observation_noise']['diagnostics'].setdefault('max_quantile_samples', 200000)
    out['observation_noise']['diagnostics'].setdefault('radial_bins', [
        {'name': 'low_corona', 'r_min': 1.01, 'r_max': 1.5},
        {'name': 'inner', 'r_min': 1.01, 'r_max': 2.0},
        {'name': 'mid_outer_inner', 'r_min': 1.5, 'r_max': 2.0},
    ])

    # effective shot noise: Poisson(lambda), lambda = alpha * I
    out['observation_noise'].setdefault('effective_shot', {})
    out['observation_noise']['effective_shot'].setdefault('alpha', None)
    out['observation_noise']['effective_shot'].setdefault('alpha_by_global_channel', None)

    # hetero gaussian noise: N(0, sigma(I)), sigma(I) = sqrt aI + b^2
    out['observation_noise'].setdefault('hetero_gaussian', {})
    out['observation_noise']['hetero_gaussian'].setdefault('shot_coeff', None)                      # a
    out['observation_noise']['hetero_gaussian'].setdefault('shot_coeff_by_global_channel', None)   
    out['observation_noise']['hetero_gaussian'].setdefault('sigma_floor', 0.0)                      # b
    out['observation_noise']['hetero_gaussian'].setdefault('sigma_floor_by_global_channel', None)

    # ensure io exists
    out.setdefault('io', {})
    out['io'].setdefault('no_save', False)

    # generic plasma field kwargs
    out.setdefault('PlasmaField_kwargs', {})
    out['PlasmaField_kwargs'].setdefault('backend', 'torch')
    out['PlasmaField_kwargs'].setdefault('backend_fallback', True)
    out['PlasmaField_kwargs'].setdefault('in_dim', 3)                 # field spatial dimension
    out['PlasmaField_kwargs'].setdefault('n_levels', 16)              # hashgrid resolution levels
    out['PlasmaField_kwargs'].setdefault('n_features_per_level', 2)   
    out['PlasmaField_kwargs'].setdefault('log2_hashmap_size', 19)     # hashgrid size in log2
    out['PlasmaField_kwargs'].setdefault('base_resolution', 16)       # coarsest
    out['PlasmaField_kwargs'].setdefault('finest_resolution', 512)    # finest
    out['PlasmaField_kwargs'].setdefault('use_raw_xyz', False)        # append raw pos to features
    out['PlasmaField_kwargs'].setdefault('use_sh_encoding', False)    # append spherical harmonics encoding
    out['PlasmaField_kwargs'].setdefault('sh_degree', 2)            
    out['PlasmaField_kwargs'].setdefault('shared_hidden_dim', 64)     # shared trunk
    out['PlasmaField_kwargs'].setdefault('shared_num_hidden', 0) 
    out['PlasmaField_kwargs'].setdefault('head_hidden_dim', 64)       # decoder head
    out['PlasmaField_kwargs'].setdefault('head_num_hidden', 2)

    return out
















