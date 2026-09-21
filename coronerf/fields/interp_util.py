###################################################
### interpolation util
###################################################

import torch
import numpy as np

def match_field(field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
	"""
	Ensure x matches dtype and device of the field and is memory contiguous.
	Efficient when no conversion is needed.
	"""
	if x.dtype != field.dtype or x.device != field.device:
		x = x.to(dtype=field.dtype, device=field.device)
	return x.contiguous()

# to [-1, +1] for uniform grid
def _axis_to_m1p1(v, lo, hi):
	g = 2 * (torch.clamp(v, lo, hi) - lo) / (hi - lo) - 1
	return g.clamp(-1.0, 1.0)

# to [-1, +1] for nonuniform grid
def _axis_to_m1p1_nonuniform(v, axis_vals):
	# axis_vals: (N,) increasing
	# v: (...,) physical coordinate(s)
	N = axis_vals.numel()
	# clamp to bounds to avoid out-of-range
	v = torch.clamp(v, axis_vals[0], axis_vals[-1])
	# find segment k with axis[k] <= v <= axis[k+1]
	k = torch.searchsorted(axis_vals, v, right=True).clamp(1, N-1) - 1  # (...,)
	v0 = axis_vals[k]
	v1 = axis_vals[k+1]
	t  = torch.where((v1 > v0), (v - v0) / (v1 - v0), torch.zeros_like(v))  # guard zero span
	i  = k.to(v.dtype) + t  # fractional index in [0, N-1]
	g  = (2.0 * i / (N - 1)) - 1.0        # to [-1, 1] with align_corners=True
	return g.clamp(-1.0, 1.0)


def compute_ccoef(log_ne, log_temp, log_r, ion, line):
	'''
	Given (ne, T, r) triplet, compute ccoef for a given emission line for a given ion
	'''
	thetab = np.degrees(np.arccos(1/np.sqrt(3))) # dummy placeholder, update when doing full stokes

	ne = 10**log_ne
	temp = 10**log_temp
	r = 10**log_r
	height = r - 1
	ion.calc_rho_sym(ne,temp,height,thetab)

	# recompute line C_coeff and upper_level_alignment
	rho00_u = ion.rho[line.upper_level_index, 0]
	#rho20_u = ion.rho[line.upper_level_index, 2]
	#upper_level_alignment = rho20_u / rho00_u if rho00_u != 0 else 0.0

	upper_level_pop_frac = np.sqrt(2.0 * line.Jupp + 1.0) * rho00_u
	C_coeff = (line.hnu / (4.0 * np.pi)) * line.Einstein_A * upper_level_pop_frac * ion.totn

	return C_coeff