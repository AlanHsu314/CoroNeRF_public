#######################################################################################
### density spherical grid (similar to spherical interpolator, but with nn.Parameters)
#######################################################################################

'''
Baseline model, just a voxel grid to learn the density inversion

a combination of 
(a) density MLP (.density_mlp.DensityMLP)
	-- completely in x_AABB
(b) spherical trilinear field (...fields.spherical.SphericalTrilinearField)
	-- does not use AABB coords (indep of nerf)
	-- takes x_xyz, interpolates in x_llr
'''

import logging
from typing import Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fields.interp_util import match_field, _axis_to_m1p1_nonuniform

class DensitySphericalGrid(nn.Module):
	'''
	trainable spherical trilinear density field

	model:
	input:  x_AABB    (..., 3) in [-1, 1]^3
	output: log10 n_e (...,)

	internally:
	- maps x_AABB -> x_xyz (AABB_scale)
	- maps x_xyz -> x_llr
	- samples grid via trilinear interpolation

	notes:
	- as usual, we send to device outside of the model
	- just take care of typing inside
	'''

	def __init__(
			self,
			lon_axis: torch.Tensor,
			phi_axis: torch.Tensor,
			r_axis: torch.Tensor,
			shared_context: dict[str, Any],
			init_value: float = 7.5,
			dtype: torch.dtype = torch.float32,
			strict_types: bool = False,
	):
		super().__init__()

		# init
		self.dtype = dtype
		self.strict_types = strict_types
		if shared_context is None:
			raise ValueError(f'Renderer expects non-None shared_context dictionary')
		self.shared_context = shared_context

		self.logger = logging.getLogger('coroNeRF.DensitySphericalGrid')
		
		# load shared context values
		if 'aabb_scale' not in shared_context:
			raise KeyError('DensitySphericalGrid requires shared_context["aabb_scale"]')
		self.aabb_scale = float(shared_context['aabb_scale'])

		# check input type
		self.logger.debug(f'input lon_axis type: {lon_axis.dtype}')
		self.logger.debug(f'input phi_axis type: {phi_axis.dtype}')
		self.logger.debug(f'input r_axis type: {r_axis.dtype}')
		self.logger.debug(f'strict_types: {self.strict_types}')

		# register axes as buffers
		self.register_buffer('lon_axis', lon_axis.to(dtype = self.dtype).contiguous())
		self.register_buffer('phi_axis', phi_axis.to(dtype = self.dtype).contiguous())
		self.register_buffer('r_axis', r_axis.to(dtype = self.dtype).contiguous())

		# do the usual padding longitude axis for periodicity correctness
		if self.lon_axis.numel() > 1: # check > 1 length ...
			dlon = self.lon_axis[1] - self.lon_axis[0]
		else:
			dlon = torch.tensor(2 * math.pi, dtype = self.dtype)

		self.register_buffer('lon_min', self.lon_axis[0])
		self.register_buffer('lon_max_padded', self.lon_axis[-1] + dlon)
		self.register_buffer('lon_axis_pad', torch.cat([self.lon_axis, self.lon_axis[-1:] + dlon], dim = 0))

		# check
		self.logger.debug(f'lon bounds: [{lon_axis[0]:.4f}, {lon_axis[-1]:.4f}], dlon padding: {dlon:.4f}')

		# create trainable field
		L = self.lon_axis.numel()
		P = self.phi_axis.numel()
		R = self.r_axis.numel()

		grid = torch.full((L, P, R), fill_value = float(init_value), dtype = self.dtype)
		self.log_ne_grid = nn.Parameter(grid)

		# check
		self.logger.debug(f'Log10 ne grid created, size=({L, P, R}), value={init_value:.04f}')

		# at the end of init, set logger from DEBUG to WARNING
		self.logger.setLevel(logging.WARNING)

	def initialize_density_bias(self, value: float):
		'''
		init density value in grid (useful to call in renderer)
		'''
		with torch.no_grad():
			self.log_ne_grid.fill_(float(value))
	
	def _wrap_lon(self, lon):
		'''
		compute the lon angle with padded max

		- same function as in spherical.py
		'''
		Lspan = self.lon_max_padded - self.lon_min
		return (lon - self.lon_min) % Lspan + self.lon_min

	def forward(self, x_AABB:  torch.Tensor) -> torch.Tensor:
		'''
		[INPUTS]
		x_AABB: (..., 3) tensor, cartesian coordinates in normalized AABB space

		[OUTPUTS]
		(...,) tensor, predict log10 electron density
		'''
		ref = self.log_ne_grid # reference to match types and device

		if self.strict_types:
			assert x_AABB.dtype == ref.dtype, f"DensitySphericalGrid dtype mismatch: x={x_AABB.dtype}, grid={ref.dtype}"
			assert x_AABB.device == ref.device, f"DensitySphericalGrid device mismatch: x={x_AABB.device}, grid={ref.device}"
			x_AABB = x_AABB.contiguous()
		else:
			x_AABB = match_field(ref, x_AABB)

		################################################
		### x_AABB -> x_llr, mostly from spherical.py
		################################################
		
		# x_AABB -> x_xyz
		x_xyz = x_AABB * self.aabb_scale

		# split coordinates
		x, y, z = x_xyz.unbind(-1)

		# convert (x, y, z) to (lon, phi, r)
		r = torch.linalg.vector_norm(x_xyz, dim = -1).clamp_min(1e-12)
		
		lon = torch.atan2(y, x) # (-pi, pi]
		lon = (lon + 2*math.pi) % (2*math.pi) # change to [0, 2pi)

		# if we are at pol, force a canonical lon (at lat = +/-pi/2, a cartesian grid learn values at all lons)
		rho = torch.linalg.vector_norm(torch.stack([x, y], dim=-1), dim=-1)
		lon = torch.where(rho < (1e-6 * r), torch.zeros_like(lon), lon)

		# wrap based on padded bounds
		lon = self._wrap_lon(lon) 

		phi = torch.asin(torch.clamp(z / r, -1.0, 1.0)) # [-pi/2, pi/2] latitude 

		# normalize to AABB [-1, 1]^3
		# gx = self._to_m1p1(lon, self.lon_min, self.lon_max_padded)   # W
		# gy = self._to_m1p1(phi, self.phi_axis[0], self.phi_axis[-1]) # H
		# gz = self._to_m1p1(r, self.r_axis[0], self.r_axis[-1])       # D

		gx = _axis_to_m1p1_nonuniform(lon, self.lon_axis_pad)   # W+1
		gy = _axis_to_m1p1_nonuniform(phi, self.phi_axis)       # H
		gz = _axis_to_m1p1_nonuniform(r, self.r_axis)           # D

		# remember to pad the field too!
		# we did this in init for spherical, bu now field changes!
		# (L, P, R) -> (L+1, P, R)
		grid_pad = torch.cat([self.log_ne_grid, self.log_ne_grid[:1]], dim = 0)

		# organize dimensions
		# grid sample wants (N, C, D, H, W) = (1, 1, R, P, Lpad)
		field_5d = grid_pad.permute(2, 1, 0)[None, None, ...]

		# now we sample points: (1, D_out, H_out, W_out, 3)
		# use W_out = number of query points
		sample_grid = torch.stack([gx, gy, gz], dim = -1).view(1, 1, 1, -1, 3)

		# do the sample using grid_sample
		vals = F.grid_sample(
			field_5d,
			sample_grid,
			mode = 'bilinear',
			padding_mode = 'border',
			align_corners = True,
		)

		return vals.view(-1)

	def smoothness_loss(self,
					 lambda_lon: float = 1.0,
					 lambda_lat: float = 1.0,
					 lambda_r: float = 1.0,
					 radial_weight_power: float = 0.0,
					 squared: bool = True,
					 ) -> torch.Tensor:
		'''
		First-order smoothness penalty on grid
		Grid is (L, P, R) = (lon, lat, r)

		[INPUTS]
		lambdas: relative weights for directional differences
		radial_weight_power: if power > 0, outer radii get upweighted by r**power
		squared: True = L2, False = L1 regularizer
		'''

		g = self.log_ne_grid

		# get diffs
		d_lon = torch.roll(g, shifts = -1, dims = 0) - g # (L, P,   R  ), gets periodicity!
		d_lat = g[:,1:,:] - g[:,:-1,:]                   # (L, P-1, R  )
		d_r = g[:,:,1:] - g[:,:,:-1]                     # (L, P,   R-1)

		# weight radially
		if radial_weight_power != 0.0:
			raise NotImplementedError('radial weighting in regualrizer not tested yet...')
			r = self.r_axis
			
			# compute weights
			w_r_full = r.pow(radial_weight_power)
			w_r_mid = 0.5*(r[1:] + r[:-1]) # for dr
			w_r_mid = w_r_mid.pow(radial_weight_power)

			# apply to diffs
			d_lon = d_lon * w_r_full.view(1, 1, -1) # broadcast
			d_lat = d_lat * w_r_full.view(1, 1, -1)
			d_r = d_r * w_r_mid.view(1, 1, -1)

		# L2 or L1
		if squared:
			loss_lon = d_lon.pow(2).mean()
			loss_lat = d_lat.pow(2).mean()
			loss_r = d_r.pow(2).mean()
		else:
			loss_lon = d_lon.abs().mean()
			loss_lat = d_lat.abs().mean()
			loss_r = d_r.abs().mean()

		total_loss = float(lambda_lon) * loss_lon + float(lambda_lat) * loss_lat + float(lambda_r) * loss_r

		return total_loss





