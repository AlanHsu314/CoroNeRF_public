#############################################
### Plasma field output abstraction
#############################################

from __future__ import annotations
from typing import Union

from dataclasses import dataclass
import torch


@dataclass
class PlasmaFieldOutputs:
    log_ne: Union[torch.Tensor, None] = None
    log_temp: Union[torch.Tensor, None] = None

def unpack_plasma_outputs(raw: torch.Tensor, target: str) -> PlasmaFieldOutputs:
    '''
    input is raw torch
    (N,) or (N,1) for target = 'ne'
    (N,2)         for target = 'ne_t'
    '''

    if raw.ndim == 1:
        raw = raw[:, None]
    
    if raw.ndim != 2:
        raise ValueError(f'Expected raw plasma outputs to be 2D, got shape {tuple(raw.shape)}')
    
    if target == 'ne':
        if raw.shape[-1] != 1:
            raise ValueError(f"target = 'ne' expects 1 output channel, got {raw.shape[-1]}")
        return PlasmaFieldOutputs(log_ne = raw[:,0], 
                                  log_temp = None)
    
    if target == 'ne_t':
        if raw.shape[-1] != 2:
            raise ValueError(f"target = 'ne_t' expects 2 output channels, got {raw.shape[-1]}")
        return PlasmaFieldOutputs(log_ne = raw[:,0], 
                                  log_temp = raw[:,1])
    
    raise ValueError(f'Unknown reconstruction target: {target}')

def get_log_ne_from_raw(raw: torch.Tensor, target: str, restore_shape: Union[tuple[int, ...], None] = None) -> torch.Tensor:
    """
    Extract predicted log_ne from raw model output.
    raw can be:
      (N,)
      (N,1)
      (N,2)
      (...,1)
      (...,2)
    """
    if raw.ndim == 1:
        raw = raw[:, None]

    orig_shape = restore_shape if restore_shape is not None else raw.shape[:-1]
    raw2d = raw.reshape(-1, raw.shape[-1])
    out = unpack_plasma_outputs(raw2d, target)
    return out.log_ne.reshape(*orig_shape)


def get_log_temp_from_raw(raw: torch.Tensor, target: str, restore_shape: Union[tuple[int, ...], None] = None) -> Union[torch.Tensor, None]:
    """
    Extract predicted log_temp from raw model output.
    Returns None for target='ne'.
    """
    if raw.ndim == 1:
        raw = raw[:, None]

    orig_shape = restore_shape if restore_shape is not None else raw.shape[:-1]
    raw2d = raw.reshape(-1, raw.shape[-1])
    out = unpack_plasma_outputs(raw2d, target)

    if out.log_temp is None:
        return None
    return out.log_temp.reshape(*orig_shape)







