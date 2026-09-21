###########################################################
### spherical interpolators
###########################################################

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interp_util import match_field, _axis_to_m1p1_nonuniform

class SphericalTrilinearField(nn.Module):
	'''
	Generic spherical trilinear interpolator that is fast and differentiable

	[INPUTS]
	field: (L, P, R) tensor
	lon_axis: (L,) tensor with longitude coordinates in radians [0, 2pi]
	phi_axis: (P,) tensor with latitude coordinates in radians [-pi/2, pi/2]
	r_axis: (R,) tensor with radius coordinates in R_sun [1.0, 30.0]
	

	Notes
	-- We pad lon_axis such that lon wraps periodically
	-- grid sample needs (N, C, D, H, W). We map R --> D, P --> H, and Lpad --> W
	'''
	def __init__(self, field: torch.Tensor, 
					   lon_axis: torch.Tensor,
					   phi_axis: torch.Tensor,
					   r_axis: torch.Tensor,
					   strict_types: bool = False):
		super().__init__()
		self.strict_types = strict_types
		assert field.ndim == 3 # has to be (lon, lat, r)

		# logging params
		self._step = 0 # count forward calls
		self.debug_every = 0 # debug every x forward calls
		self.logger = logging.getLogger('coroNeRF.SphericalTrilinearField')

		# check input type
		self.logger.debug(f'input field type: {field.dtype}')
		self.logger.debug(f'input lon_axis type: {lon_axis.dtype}')
		self.logger.debug(f'input phi_axis type: {phi_axis.dtype}')
		self.logger.debug(f'input r_axis type: {r_axis.dtype}')
		self.logger.debug(f'strict_types: {self.strict_types}')

		# first pad the actual field in lon axis
		L, P, R = field.shape
		field_pad = torch.cat([field, field[:1]], dim = 0) # [:1] preserves extra dimension for concatenation
		field_5d = field_pad.permute(2, 1, 0)[None, None, ...] # make dim (1, 1, R, P, L)

		# logging
		self.logger.debug(f'field shape: {field.shape}')
		self.logger.debug(f'padded field shape: {field_pad.shape}')
		self.logger.debug(f'padded 5d field shape: {field_5d.shape}')

		# register field
		self.register_buffer('field', field_5d)

		# store axes
		self.register_buffer('lon_axis', lon_axis)
		self.register_buffer('phi_axis', phi_axis)
		self.register_buffer('r_axis', r_axis)

		# define the dlon used for padding later
		'''
		grid_sample does not interpolate cyclically. Thus, to numerically make the interpolator periodic,
		we add a synthetic dlon entry past the max so the values can smoothly transition to the modulo min
		'''
		dlon = (lon_axis[1] - lon_axis[0]) if lon_axis.numel() > 1 else torch.tensor(2*math.pi, dtype = lon_axis.dtype)
		self.register_buffer('lon_min', lon_axis[0])
		self.register_buffer('lon_max_padded', lon_axis[-1] + dlon)

		# build a padded lon axis for m1p1 mapping later in forward
		lon_axis_pad = torch.cat([self.lon_axis, self.lon_axis[-1:] + dlon])
		self.register_buffer('lon_axis_pad', lon_axis_pad)

		# check
		self.logger.debug(f'lon bounds: [{lon_axis[0]:.4f}, {lon_axis[-1]:.4f}], dlon padding: {dlon:.4f}')

		# at the end of init, set logger from DEBUG to WARNING
		self.logger.setLevel(logging.WARNING)

	def _wrap_lon(self, lon):
		'''
		compute the lon angle with padded max
		'''
		Lspan = self.lon_max_padded - self.lon_min
		return (lon - self.lon_min) % Lspan + self.lon_min

	def forward(self, x_xyz: torch.Tensor) -> torch.Tensor:
		'''
		Given a spatial coordinate in Cartesian, grid_sample the T_field

		[INPUTS]
		x_xyz: (N,3) physical Cartesian in R_sun
		'''
		if self.strict_types:
			field = self.field
			assert x_xyz.dtype == field.dtype, "dtype mismatch in temp_field.forward"
			assert x_xyz.device == field.device, "device mismatch in temp_field.forward"
			x_xyz = x_xyz.contiguous()
		else:
			x_xyz = match_field(self.field, x_xyz)

		# count step
		self._step += 1

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

		# create grid and interpolate
		'''
		recall grid_sample expects grid to be (N, D_out, H_out, W_out, 3)
		we have M samples that we want to sample at, so we do (1, 1, 1, M, 3)
		'''
		grid = torch.stack([gx, gy, gz], dim = -1).view(1, 1, 1, -1, 3)
		vals = F.grid_sample(self.field, 
					   		 grid, 
							 mode = 'bilinear', 
							 padding_mode = 'border', 
							 align_corners = True) # (1, 1, 1, 1, N)
		
		# logging calls
		# if self.debug_every and (self._step % self.debug_every == 0) and \
		# 	  self.logger.isEnabledFor(logging.DEBUG):
			
		# 	self.logger.debug(f'forward coord: {lon, phi, r}')
		# 	self.logger.debug(f'grid shape: {grid.shape}')
		# 	self.logger.debug(f'vals shape: {vals.shape}')

		return vals.view(-1) # (N,)