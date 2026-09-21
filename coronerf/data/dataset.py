#################################################
### dataset classes
#################################################


import logging
from torch.utils.data import Dataset
import torch


class CoronaDataset(Dataset):
	def __init__(self, imgs: torch.Tensor, 
				 mask: torch.Tensor, 
				 rays_o: torch.Tensor, 
				 rays_d: torch.Tensor):
		'''
		imgs: (V,H,W,C) float32 target intensities per channel (C=2)
		mask: (H,W) boolean; True where corona
		rays_o: (V,3) origin per view in normalized coords
		rays_d: (V,H,W,3) direction per pixel in normalized coords
		'''
		logger = logging.getLogger('coroNeRF.CoronaDataset')
		# expects torch inputs

		# get V, H, W, C
		assert imgs.ndim == 4
		V, H, W, C = imgs.shape

		# save num channels
		self.num_channels = int(C)
		self.target_sigma_flat = None 

		# our dataset class does not need pixels inside photosphere (aka do the masking for efficient sampling later)
		keep = mask.view(1, H, W, 1).expand(V, H, W, 1).contiguous() # broadcast mask to V viewpoints, since expand makes it noncontiguous, convert back

		# indices of pixels to keep
		self.pix_idx = keep.view(-1).nonzero().squeeze(-1) # flatten, filter nonzero, then flatten again since .nonzero() create 2D array

		# filter images
		self.imgs_flat = imgs.view(-1, C)[self.pix_idx] # flatten imgs, then extract idx

		# filter rays_d
		self.rays_d_flat = rays_d.view(-1, 3)[self.pix_idx]

		# "filter" rays_o while also making it batchable with imgs and rays_d
		rays_o_aug = rays_o[:,None,None,:].expand(V, H, W, 3).contiguous() # add 2 new axes for broadcasting, then broadcast, then make contiguous
		self.rays_o_flat = rays_o_aug.view(-1, 3)[self.pix_idx]

		# check
		logger.debug(f'pix_idx shape: {self.pix_idx.shape}')
		logger.debug(f'imgs_flat shape: {self.imgs_flat.shape}')
		logger.debug(f'rays_d_flat shape: {self.rays_d_flat.shape}')
		logger.debug(f'rays_o_flat shape: {self.rays_o_flat.shape}')

	def replace_targets(self, imgs_flat_new: torch.Tensor):
		'''
		helper for later on: when we add observation noise we will call this method
		--> replaces training data with noisy data (or technically any other data)
		'''
		if not isinstance(imgs_flat_new, torch.Tensor):
			raise TypeError(f'imgs_flat_new must be torch.Tensor, got {type(imgs_flat_new)}')
		if imgs_flat_new.shape != self.imgs_flat.shape:
			raise ValueError(
				f'imgs_flat_new shape {tuple(imgs_flat_new.shape)} does not match existing '
				f'imgs_flat shape {tuple(self.imgs_flat.shape)}'
			)
		if imgs_flat_new.dtype != self.imgs_flat.dtype:
			raise TypeError(
				f'imgs_flat_new dtype {imgs_flat_new.dtype} does not match existing dtype {self.imgs_flat.dtype}'
			)

		self.imgs_flat = imgs_flat_new.contiguous()

	def replace_target_sigma(self, sigma_flat_new: torch.Tensor):
		'''
		currently very similar to replace_targets, but used for hetero_gaussian case
		'''
		if not isinstance(sigma_flat_new, torch.Tensor):
			raise TypeError(f'sigma_flat_new must be torch.Tensor, got {type(sigma_flat_new)}')
		if sigma_flat_new.shape != self.imgs_flat.shape:
			raise ValueError(
				f'sigma_flat_new shape {tuple(sigma_flat_new.shape)} does not match target shape {tuple(self.imgs_flat.shape)}'
			)
		if sigma_flat_new.dtype != self.imgs_flat.dtype:
			raise TypeError(
				f'sigma_flat_new dtype {sigma_flat_new.dtype} does not match existing dtype {self.imgs_flat.dtype}'
			)

		self.target_sigma_flat = sigma_flat_new.contiguous()

	def __len__(self):
		return self.pix_idx.numel()
	
	def __getitem__(self, i):
		out = {
			"target": self.imgs_flat[i], # 2-channel
			"ray_o": self.rays_o_flat[i], # 3-vec
			"ray_d": self.rays_d_flat[i], # 3-vec
		}

		# if we added hetero-gaussian noise, replace targets
		if self.target_sigma_flat is not None:
			out["target_sigma"] = self.target_sigma_flat[i]
		return out	