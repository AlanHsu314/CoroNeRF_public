###############################################################
### positional and angular encoding
###############################################################

import math
import torch
import torch.nn as nn
from ..util.torch import match_ref

# chatGPT function
def spherical_harmonics_encoding(u: torch.Tensor, degree: int = 2) -> torch.Tensor:
	"""
	Simple real spherical-harmonic encoding of a unit vector u.

	Inputs
	-------
	u: (..., 3) tensor of unit vectors

	Returns
	-------
	enc: (..., (degree+1)^2) SH features up to given degree
	"""
	# u: (...,3)
	x = u[..., 0]
	y = u[..., 1]
	z = u[..., 2]

	enc = []

	# l = 0 (1 term)
	enc.append(0.282095 * torch.ones_like(x))  # Y_0^0

	if degree >= 1:
		# l = 1 (3 terms)
		enc.extend([
			-0.488603 * y,          # Y_1^-1
			 0.488603 * z,          # Y_1^0
			-0.488603 * x,          # Y_1^1
		])

	if degree >= 2:
		# l = 2 (5 terms)
		enc.extend([
			 1.092548 * x * y,                     # Y_2^-2
			-1.092548 * y * z,                     # Y_2^-1
			 0.315392 * (3.0 * z * z - 1.0),       # Y_2^0
			-1.092548 * x * z,                     # Y_2^1
			 0.546274 * (x * x - y * y),           # Y_2^2
		])

	# extend to degree >= 3 later if needed
	return torch.stack(enc, dim=-1)

class PosEnc(nn.Module):
	'''
	small positional encoding module for x_AABB for high-frequency rentention during training
	'''
	def __init__(self, in_dim: int = 3, L: int = 6, dtype = torch.float32, strict_types: bool = False):
		super().__init__()
		self.in_dim = in_dim
		self.L = L
		self.strict_types = strict_types

		# register buffer to it goes with .to(device)
		freqs = 2.0 ** torch.arange(L, dtype=dtype)
		self.register_buffer("freqs", freqs)

	@property
	def out_dim(self) -> int:
		# original x + 2 * in_dim * L
		return self.in_dim + 2 * self.in_dim * self.L
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (..., in_dim), assumed already in [-1, 1] AABB space

		# cast type and device if needed
		if self.strict_types:
			assert x.dtype == self.freqs.dtype,  f"PosEnc dtype mismatch: x={x.dtype}, freqs={self.freqs.dtype}"
			assert x.device == self.freqs.device, f"PosEnc device mismatch: x={x.device}, freqs={self.freqs.device}"
			x = x.contiguous()
		else:
			x = match_ref(self.freqs, x)

		pe = [x]
		# self.freqs: (L,), will broadcast over x
		for f in self.freqs:
			pe.append(torch.sin(f * math.pi * x))
			pe.append(torch.cos(f * math.pi * x))
		return torch.cat(pe, dim=-1)