####################################################################
### generic torch helper functions
####################################################################

import torch
import numpy as np

from ..fields.interp_util import match_field

# wrapper for match_field, when the obj is just a reference and not really a "tensor"
def match_ref(ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return match_field(ref, x)

DTYPE_MAP = {
	'float': torch.float32,
	'float32': torch.float32,
	'double': torch.float64,
	'float64': torch.float64,
}

def to_numpy(x) -> np.ndarray:
	if isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy()
	return np.asarray(x)

def get_torch_dtype(name: str) -> torch.dtype:
	try:
		return DTYPE_MAP[name]
	except KeyError:
		raise ValueError(f"Unknown dtype string: {name}")
	
def np_to_torch(x, torch_dtype = torch.float32) -> torch.Tensor:
	'''
	numpy array to torch tensor with type
	'''
	if torch_dtype == torch.float64:
		return torch.from_numpy(x).double()
	elif torch_dtype == torch.float32:
		return torch.from_numpy(x).float()
	else:
		return torch.from_numpy(x).to(dtype = torch_dtype)
	
