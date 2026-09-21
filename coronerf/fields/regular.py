###########################################################
### cartesian interpolators
###########################################################

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interp_util import match_field, _axis_to_m1p1_nonuniform

class RegularTrilinearField(nn.Module):
	'''
	Generic cartesian trilinear interpolator for f(x, y, z) on nonuniform grid

	[INPUTS]
	field: (C, Nx, Ny, Nz) OR (Nx, Ny, Nz) tensor
	x_axis: (Nx,) tensor
	y_axis: (Ny,) tensor
	z_axis: (Nz,) tensor
	
	Notes
	-- for us, x = log_ne, y = log_temp, z = log_r 
	-- grid sample needs (N, C, D, H, W). We map Nz --> D, Ny --> H, and Nx --> W
	'''
	def __init__(self, field: torch.Tensor, 
					   x_axis: torch.Tensor,
					   y_axis: torch.Tensor,
					   z_axis: torch.Tensor,
					   strict_types: bool = False,
					   squeeze_single_channel: bool = True):
		super().__init__()
		self.strict_types = strict_types

		# logging params
		self.squeeze_single_channel = squeeze_single_channel # for single channel inputs
		self.logger = logging.getLogger('coroNeRF.RegularTrilinearField')

		# ensure we always have a channel dim
		if field.ndim == 3:
			field = field.unsqueeze(0)  # (1, Nx, Ny, Nz)
		assert field.ndim == 4, f'values must be (C,Nx,Ny,Nz), got {field.shape}'

		# axes should match these lengths
		C, Nx, Ny, Nz = field.shape
		assert x_axis.shape[0] == Nx
		assert y_axis.shape[0] == Ny
		assert z_axis.shape[0] == Nz

		# check input type
		self.logger.debug(f'input field type: {field.dtype}')
		self.logger.debug(f'input x_axis (ne) type: {x_axis.dtype}')
		self.logger.debug(f'input y_axis (temp) type: {y_axis.dtype}')
		self.logger.debug(f'input z_axis (r) type: {z_axis.dtype}')
		self.logger.debug(f'strict_types: {self.strict_types}')

		# register field (remember that grid sample saves axes backwards, so need to permute)
		field_5d = field.permute(0, 3, 2, 1)[None, ...] # make dim (1, C, Nz, Ny, Nx)
		self.register_buffer('field', field_5d)

		# register axes
		self.register_buffer('x_axis', x_axis)
		self.register_buffer('y_axis', y_axis)
		self.register_buffer('z_axis', z_axis)

		# logging
		self.logger.debug(f'saved field shape: {field_5d.shape}')
		self.logger.info(
			f'RegularTrilinearField initialized: C={C}, Nx={Nx}, Ny={Ny}, Nz={Nz}, '
			f'dtype={field.dtype}, device={field.device}'
		)

		# at the end of init, set logger from DEBUG to WARNING
		self.logger.setLevel(logging.WARNING)

	def forward(self, x_xyz: torch.Tensor) -> torch.Tensor:
		'''
		Given a spatial coordinate in Cartesian, grid_sample the field

		[INPUTS]
		x_xyz: (...,3)

		Notes:
		-- in our case, coordinates are (x, y, z) = (log_ne, log_temp, log_r)
		'''
		if self.strict_types:
			field = self.field
			assert x_xyz.dtype == field.dtype, "dtype mismatch in ccoef_field.forward"
			assert x_xyz.device == field.device, "device mismatch in ccoef_field.forward"
			x_xyz = x_xyz.contiguous()
		else:
			x_xyz = match_field(self.field, x_xyz)

		# split coordinates
		x, y, z = x_xyz.unbind(-1)

		# map each coordinate to m1p1
		gx = _axis_to_m1p1_nonuniform(x, self.x_axis)
		gy = _axis_to_m1p1_nonuniform(y, self.y_axis)
		gz = _axis_to_m1p1_nonuniform(z, self.z_axis)

		orig_shape = gx.shape # reshape after sampling, in case input (...,3) is complex

		# stack and reshape 
		grid = torch.stack([gx, gy, gz], dim = -1).view(1, 1, 1, -1, 3)

		# sample
		vals = F.grid_sample(self.field, 
					   		 grid, 
							 mode = 'bilinear', 
							 padding_mode = 'border', 
							 align_corners = True) # (1, C, 1, 1, N)

		C = self.field.shape[1]
		vals = vals.view(C, *orig_shape) # reshape

		# squeeze if necessary
		if C == 1 and self.squeeze_single_channel:
			vals = vals.squeeze(0)        # -> (...)
		return vals




