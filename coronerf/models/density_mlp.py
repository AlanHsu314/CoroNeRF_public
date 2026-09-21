###############################################################
### density mlp
###############################################################

import torch
import torch.nn as nn
from .encoding import PosEnc, spherical_harmonics_encoding
from ..util.torch import match_ref

class DensityMLP(nn.Module):
	'''
	basic MLP for density nerf model
	'''
	def __init__(self, hidden_dim: int = 128, 
			  		   num_hidden: int = 3,
					   L: int = 6, 
					   in_dim: int = 3,
					   dtype = torch.float32,
					   strict_types: bool = False,
					   use_sh_encoding: bool = False,
					   sh_degree: int = 2):
		super().__init__()
		self.dtype = dtype
		self.strict_types = strict_types
		self.use_sh_encoding = use_sh_encoding
		self.sh_degree = sh_degree
		
		# define pos enc
		self.pe = PosEnc(in_dim = in_dim, L = L, dtype = dtype, strict_types = strict_types)
		pe_dim = self.pe.out_dim

		# angular (SH) feature dimension
		if self.use_sh_encoding:
			self.sh_dim = (self.sh_degree + 1)**2 # l = 0..L
		else:
			self.sh_dim = 0

		# build layers
		in_features = pe_dim + self.sh_dim
		layers = []
		dims = [in_features] + [hidden_dim]*num_hidden + [1]
		# everything except last layer to 1
		for i in range(len(dims) - 2): 
			layers.append(nn.Linear(dims[i], dims[i+1]))
			layers.append(nn.SiLU())
		# last layer
		layers.append(nn.Linear(dims[-2], dims[-1]))

		# define net 
		self.net = nn.Sequential(*layers)

		# add bias to last layer
		self.to(dtype)

	def initialize_density_bias(self, value: float):
		'''
		used by renderer 
		'''
		with torch.no_grad():
			last = self.net[-1]
			if hasattr(last, 'bias') and last.bias is not None:
				last.bias.fill_(float(value))

	def forward(self, x_AABB: torch.Tensor) -> torch.Tensor:
		'''
		Docstring for forward
		
		:param x_AABB: (..., 3) position in AABB [-1,1]^3
		:return: (...,) log10 n_e
		'''

		# match dtype and device if needed
		ref_param = next(self.net.parameters())

		if self.strict_types:
			assert x_AABB.dtype == ref_param.dtype, f"DensityMLP dtype mismatch: x={x_AABB.dtype}, params={ref_param.dtype}"
			assert x_AABB.device == ref_param.device, f"DensityMLP device mismatch: x={x_AABB.device}, params={ref_param.device}"
			x_AABB = x_AABB.contiguous()
		else:
			x_AABB = match_ref(ref_param, x_AABB)

		# positional encoding
		x_enc = self.pe(x_AABB)

		# optional angular encoding
		if self.use_sh_encoding:
			r = torch.linalg.norm(x_AABB, dim = -1, keepdim=True)
			u = x_AABB / (r + 1e-9)

			if self.strict_types:
				assert u.dtype == ref_param.dtype and u.device == ref_param.device
				u = u.contiguous()
			else:
				u = match_ref(ref_param, u)

			# get sh encoding
			dir_enc = spherical_harmonics_encoding(u, degree = self.sh_degree)

			# concatenat
			x_in = torch.cat([x_enc, dir_enc], dim = -1)
		else:
			x_in = x_enc # only positional encoding

		ne_out = self.net(x_in)
		return ne_out.squeeze(-1)

