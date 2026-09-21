######################################################
### emissivity renderer class
######################################################

from typing import Tuple, Optional, Dict, Union, Any
import logging

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# helpers
from ..models.density_mlp import DensityMLP
from ..fields.spherical import SphericalTrilinearField
from ..fields.regular import RegularTrilinearField

# model helpers
from ..models.field_outputs import unpack_plasma_outputs

# nerfacc
try:
	import nerfacc
except:
	nerfacc = None

# guard function to use for nerfacc
def nerfacc_is_available():
	if nerfacc is None:
		raise ImportError("renderer_model='nerfacc' requires the nerfacc package.")

# for visualization, radial mask for samples
def radial_sample_mask(x_xyz: torch.Tensor,
					   r_min_rsun: Union[float, None] = None,
					   r_max_rsun: Union[float, None] = None) -> torch.Tensor:
	'''
	Return boolean mask selecting samples within a physical radial range.

	x_xyz: (..., 3) sample positions in physical R_sun coordinates
	'''
	r = torch.linalg.vector_norm(x_xyz, dim = -1)

	keep = torch.ones_like(r, dtype = torch.bool)
	if r_min_rsun is not None:
		keep = keep & (r >= float(r_min_rsun))
	if r_max_rsun is not None:
		keep = keep & (r <= float(r_max_rsun))

	return keep

# main class
class EmissivityRenderer(nn.Module):
	'''
	renderer class for ray marching and LOS integration

	For a given ray
	[1] sample points in x_AABB
	[2] computes T(x_llr) 
	[3] computes r_xyz from x_AABB
	[4] computes ne from DensityMLP
	[5] computes ccoef(ne, T, r)
	[6] LOS integral of ccoef
	'''
	def __init__(self, model: DensityMLP, 
					   temp_field: SphericalTrilinearField, 
					   ccoef_field: RegularTrilinearField, 
					   cfg: Dict[str, Any],
					   ne_field: Optional[SphericalTrilinearField] = None,
					   shared_context: Dict[str, Any] = None):
		
		# logging
		self._step = 0 # count forward calls
		self.debug_every = 0 # debug every x forward calls
		self.forward_debug_on = False # turns on inside forward loop if step matches condition
		self.logger = logging.getLogger('coroNeRF.EmissivityRenderer')

		super().__init__()
		self.cfg = cfg
		if shared_context is None:
			raise ValueError(f'Renderer expects non-None shared_context dictionary')
		self.shared_context = shared_context
		self.strict_types = cfg.get('strict_types', False)
		self.clamp = self.cfg.get('clamp', True)

		# cross-module type and device consistency
		model_ref = next(model.parameters())  # model net reference parameter (for dtype and params)

		# canonical dtype and device for the renderer
		self.dtype = model_ref.dtype
		self.device = model_ref.device

		self.logger.debug(f'before strict types')
		self.logger.debug(f'model dtype: {model_ref.dtype}, device = {model_ref.device}')
		self.logger.debug(f'temp_field dtype: {temp_field.field.dtype}, device = {temp_field.field.device}')
		self.logger.debug(f'ccoef_field dtype: {ccoef_field.field.dtype}, device = {ccoef_field.field.device}')
		if ne_field is not None:
			self.logger.debug(f'ne_field dtype: {ne_field.field.dtype}, device = {ne_field.field.device}')

		if self.strict_types: 
			assert model_ref.dtype == temp_field.field.dtype, \
				   f"model and temp_field dtype mismatch: model={model_ref.dtype}, field={temp_field.field.dtype}"
			assert model_ref.device == temp_field.field.device, \
				   f"model and temp_field device mismatch: model={model_ref.device}, field={temp_field.field.device}"
			assert model_ref.dtype == ccoef_field.field.dtype, \
				   f"model and ccoef_field dtype mismatch: model={model_ref.dtype}, field={ccoef_field.field.dtype}"
			assert model_ref.device == ccoef_field.field.device, \
				   f"model and ccoef_field device mismatch: model={model_ref.device}, field={ccoef_field.field.device}"
			if ne_field is not None:
				assert model_ref.dtype == ne_field.field.dtype, \
					f"model and ne_field dtype mismatch: model={model_ref.dtype}, field={ne_field.field.dtype}"
				assert model_ref.device == ne_field.field.device, \
					f"model and ne_field device mismatch: model={model_ref.device}, field={ne_field.field.device}"
		else:
			# canonically match fields to model
			temp_field = temp_field.to(device=model_ref.device, dtype=model_ref.dtype)
			ccoef_field = ccoef_field.to(device=model_ref.device, dtype=model_ref.dtype)
			if ne_field is not None:
				ne_field = ne_field.to(device=model_ref.device, dtype=model_ref.dtype)

		self.logger.debug(f'after strict types') # should all match
		self.logger.debug(f'model dtype: {model_ref.dtype}, device = {model_ref.device}')
		self.logger.debug(f'temp_field dtype: {temp_field.field.dtype}, device = {temp_field.field.device}')
		self.logger.debug(f'ccoef_field dtype: {ccoef_field.field.dtype}, device = {ccoef_field.field.device}')
		if ne_field is not None:
			self.logger.debug(f'ne_field dtype: {ne_field.field.dtype}, device = {ne_field.field.device}')

		# save model and fields
		self.model = model
		self.temp_field = temp_field
		self.ccoef_field = ccoef_field
		self.ne_field = ne_field
		
		# ground-truth switch
		self.use_gt_ne = self.cfg.get('use_gt_ne', False)

		# dtype and device checking for renderer hyperparameters
		# integers
		self.occ_warmup = int(self.cfg.get('occ_warmup', 256))
		self.occ_res = int(self.cfg.get('occ_res', 128))
		if self.strict_types:
			assert self.occ_warmup >= 0, 'occ_warmup must be non-negative'
			assert self.occ_res > 0, 'occ_res must be positive'
		
		# float, shared context
		if 'aabb_scale' not in shared_context:
			raise KeyError('EmissivityRenderer requires shared_context["aabb_scale"]')
		self.aabb_scale = float(shared_context['aabb_scale'])
		if 'aabb_min' not in shared_context:
			raise KeyError('EmissivityRenderer requires shared_context["aabb_min"]')
		self.aabb_min = float(shared_context['aabb_min'])
		if 'aabb_max' not in shared_context:
			raise KeyError('EmissivityRenderer requires shared_context["aabb_max"]')
		self.aabb_max = float(shared_context['aabb_max'])

		# other, shared context
		if 'reconstruction_target' not in shared_context:
			raise KeyError('EmissivityRenderer requires shared_context["reconstruction_target"]')
		self.reconstruction_target = shared_context['reconstruction_target']

		# float, renderer cfg
		self.step_size = float(self.cfg.get('step_size', 1 / 256))
		self.occ_decay = float(self.cfg.get('occ_decay', 0.01))
		self.occ_alpha = float(self.cfg.get('occ_alpha', 1.5))
		self.sigma_scale = float(self.cfg.get('sigma_scale', 1))
		self.occ_eval_fn_ne_width = float(self.cfg.get('occ_eval_fn_ne_width', 0.25))
		self.occ_eval_fn_ne_mid = float(self.cfg.get('occ_eval_fn_ne_mid', 7.25))
		self.min_samples_per_ray = int(self.cfg.get("min_samples_per_ray", 64))
		self.enable_fallback = bool(self.cfg.get("enable_min_samples_fallback", True))
		self.fallback_when_samples = int(self.cfg.get('fallback_when_samples', 8))

		# checks
		if self.strict_types:
			assert self.aabb_min < self.aabb_max, 'aabb_min must be < aabb_max'
			assert self.aabb_scale > 0, 'aabb_scale must be positive'
			assert self.step_size > 0, 'step_size must be positive'
			assert 0 < self.occ_decay < 1, 'occ_decay must be between (0, 1)'
			assert 1 <= self.occ_alpha <= 2, 'occ_alpha must be between [1, 2]'

		self.logger.debug(f'loaded and checked renderer hyperparameters')

		# define nerfacc estimator (occupancy grid of importance sampling weights)
		if self.cfg['renderer_model'] == 'nerfacc':
			# check
			nerfacc_is_available()

			# build estimate
			self.nerfacc_estimator = nerfacc.OccGridEstimator(
								roi_aabb = torch.tensor([self.aabb_min]*3 + [self.aabb_max]*3,
														dtype=self.dtype,
														device=self.device),
								resolution = self.occ_res)
			
			self.logger.debug(f'occgrid roi_aabb = {self.nerfacc_estimator.aabbs}, res = {self.nerfacc_estimator.resolution}')

			# send to device
			self.nerfacc_estimator.to(self.device)
			self.logger.debug(f'sent renderer estimate to {self.device}')
		else:
			self.nerfacc_estimator = None
		
		# assume nerf sigma is proportional to ne^alpha via some scale for importance sampling
		self.sigma_scale = nn.Parameter(torch.tensor(self.sigma_scale, 
											   		 dtype=self.dtype, 
													 device=self.device))
		
		# register ne clamp bounds
		if self.clamp:
			log_ne_min = torch.tensor(self.cfg["log_ne_min"], dtype=self.dtype, device=self.device)
			log_ne_max = torch.tensor(self.cfg["log_ne_max"], dtype=self.dtype, device=self.device)

			# strict typing, technically not need since we load it in with dtype and device already
			if self.strict_types:
				assert log_ne_min.dtype == self.dtype, \
					   f"log_ne_min and model dtype mismatch: log_ne_min={log_ne_min.dtype}, model={self.dtype}"
				assert log_ne_min.device == self.device, \
					   f"log_ne_min and model device mismatch: log_ne_min={log_ne_min.device}, model={self.device}"
				assert log_ne_max.dtype == self.dtype, \
					   f"log_ne_max and model dtype mismatch: log_ne_max={log_ne_max.dtype}, model={self.dtype}"
				assert log_ne_max.device == self.device, \
					   f"log_ne_max and model device mismatch: log_ne_max={log_ne_max.device}, model={self.device}"

			# store as buffers
			self.register_buffer("log_ne_min", log_ne_min)
			self.register_buffer("log_ne_max", log_ne_max)

			# store temp and r bounds
			self.register_buffer("log_temp_min", torch.tensor(self.cfg["log_temp_min"], dtype=self.dtype, device=self.device))
			self.register_buffer("log_temp_max", torch.tensor(self.cfg["log_temp_max"], dtype=self.dtype, device=self.device))
			self.register_buffer("log_r_min", torch.tensor(self.cfg["log_r_min"], dtype=self.dtype, device=self.device))
			self.register_buffer("log_r_max", torch.tensor(self.cfg["log_r_max"], dtype=self.dtype, device=self.device))

			# logger
			self.logger.debug('renderer will clamp (log_ne, log_temp, log_r) before ccoef')
			self.logger.debug(f'registered log_ne bounds: [{self.cfg["log_ne_min"]}, {self.cfg["log_ne_max"]}]')
			self.logger.debug(f'registered log_temp bounds: [{self.cfg["log_temp_min"]}, {self.cfg["log_temp_max"]}]')
			self.logger.debug(f'registered log_r bounds: [{self.cfg["log_r_min"]}, {self.cfg["log_r_max"]}]')

		else:
			self.log_ne_min = None
			self.log_ne_max = None
			self.log_temp_min = None
			self.log_temp_max = None
			self.log_r_min = None
			self.log_r_max = None

			# logger
			self.logger.debug('renderer will NOT clamp (log_ne, log_temp, log_r) before ccoef, may give bad inteprolants')


		self.logger.debug(f'loaded and checked log_ne, log_temp, log_r bounds')

		# add bias to last layer of model (so it inputs avg of ne bounds at start)
		if self.clamp:
			init_val = float((self.log_ne_min + self.log_ne_max) / 2.0)

			if hasattr(self.model, 'initialize_density_bias'):
				self.model.initialize_density_bias(init_val)
			elif hasattr(self.model, 'net'):
				# backward compatibility path for DensityMLP, we'll update that too
				with torch.no_grad():
					last = self.model.net[-1]  # final nn.Linear
					if hasattr(last, 'bias') and last.bias is not None:
						last.bias.fill_(init_val)
			else:
				self.logger.warning(
					'Model has no initialize_density_bias() hook and no .net[-1].bias; '
					'skipping density midpoint initialization, proceed with caution...'
				)

		# initialize temperature midpoint bias when model predicts temperature
		if self.clamp and self.reconstruction_target == 'ne_t':
			init_temp_val = float((self.log_temp_min + self.log_temp_max) / 2.0)

			if hasattr(self.model, 'initialize_temperature_bias'):
				self.model.initialize_temperature_bias(init_temp_val)
			elif hasattr(self.model, 'initialize_temp_bias'):
				self.model.initialize_temp_bias(init_temp_val)
			else:
				self.logger.warning(
					'Model predicts temperature but has no initialize_temperature_bias() hook; '
					'skipping temperature midpoint initialization.'
				)

			self.logger.debug(f'initialized temperature bias to {init_temp_val}')

		# at the end of init, set logger from DEBUG to WARNING
		self.logger.setLevel(logging.WARNING)

	# legacy
	# def _predict_ne(self, x_AABB: torch.Tensor):
	# 	"""
	# 	given AABB, [1] sends through MLP, [2] clamps out log ne, [3] returns ne
	# 	"""
	# 	log10_ne = self.model(x_AABB)

	# 	if self.clamp:
	# 		log10_ne = log10_ne.clamp(min=self.log_ne_min, max=self.log_ne_max)

	# 	ne = 10.0 ** log10_ne
	# 	return log10_ne, ne

	# def _predict_ne(self, x_AABB: torch.Tensor):
	# 	"""
	# 	given AABB, [1] sends through MLP, [2] clamps out log ne, [3] returns ne
	# 	"""
	# 	log10_ne = self._predict_log_ne(x_AABB)
	# 	ne = 10.0 ** log10_ne
	# 	return log10_ne, ne
	
	# generic field forward
	def _predict_field_outputs(self, x_AABB: torch.Tensor):
		raw = self.model(x_AABB)

		# backward compatibility: old density-only models may return (...,)
		if raw.ndim == x_AABB.ndim - 1:
			raw = raw[..., None]

		raw_flat = raw.reshape(-1, raw.shape[-1])
		return unpack_plasma_outputs(raw_flat, self.reconstruction_target)

	# post-process the forward output a bit
	def _predict_learned_fields(self, x_AABB: torch.Tensor):
		outs = self._predict_field_outputs(x_AABB)

		log_ne = outs.log_ne
		if self.clamp:
			log_ne = log_ne.clamp(min=self.log_ne_min, max=self.log_ne_max)

		log_temp = outs.log_temp
		if log_temp is not None and self.clamp:
			log_temp = log_temp.clamp(min=self.log_temp_min, max=self.log_temp_max)

		return outs, log_ne, log_temp
	
	# fn that evaluates importance weight at position in occupancy grid
	def occ_eval_fn(self, x_AABB):
		# choose density source (match _forward_lut)
		if self.use_gt_ne and (self.ne_field is not None): # why? for vpgen
			# map to physical coords for LUT queries
			x_xyz = x_AABB * self.aabb_scale

			# call interpolant
			log10_ne = self.ne_field(x_xyz)
			if self.clamp:
				log10_ne = log10_ne.clamp(self.log_ne_min, self.log_ne_max)
			ne = 10.0 ** log10_ne
		else: # for training
			_, log10_ne, _ = self._predict_learned_fields(x_AABB)  # already clamped if clamp=True
			#log10_ne = torch.log10(ne)

		# map density to sigma (right now sigma = scale * ne**alpha)
		#sigma = F.softplus(self.sigma_scale) * ne ** self.occ_alpha

		# finally map sigma to occupancy via 1 - e^-sigma*step_size
		#occ = 1.0 - torch.exp(-sigma * self.step_size)

		occ = torch.sigmoid((log10_ne - self.occ_eval_fn_ne_mid) / self.occ_eval_fn_ne_width)
		
		return occ

	@torch.no_grad()
	def update_occupancy(self, step: int):
		'''
		Given a timestep, updates occupancy grid for importance sampling during ray marching
		'''

		# # fn that evaluates importance weight at position in occupancy grid
		# def occ_eval_fn(x_AABB):
		# 	# choose density source (match _forward_lut)
		# 	if self.use_gt_ne and (self.ne_field is not None): # why? for vpgen
		# 		# map to physical coords for LUT queries
		# 		x_xyz = x_AABB * self.aabb_scale

		# 		# call interpolant
		# 		log10_ne = self.ne_field(x_xyz)
		# 		if self.clamp:
		# 			log10_ne = log10_ne.clamp(self.log_ne_min, self.log_ne_max)
		# 		ne = 10.0 ** log10_ne
		# 	else: # for training
		# 		_, ne = self._predict_ne(x_AABB)  # already clamped if clamp=True
		# 		log10_ne = torch.log10(ne)

		# 	# map density to sigma (right now sigma = scale * ne**alpha)
		# 	#sigma = F.softplus(self.sigma_scale) * ne ** self.occ_alpha

		# 	# finally map sigma to occupancy via 1 - e^-sigma*step_size
		# 	#occ = 1.0 - torch.exp(-sigma * self.step_size)

		# 	occ = torch.sigmoid((log10_ne - self.occ_eval_fn_ne_mid) / self.occ_eval_fn_ne_width)
			
		# 	return occ
		
		# compute ema decay (exponential moving average decay)
		# we factored this into built-in warmup for the estimator
		# if step > self.occ_warmup:
		# 	ema = 1.0 - self.occ_decay
		# else:
		# 	ema = 0.0

		# define n step update for occ grid
		if self.cfg['renderer_model'] == 'nerfacc':
			nerfacc_is_available()
			self.nerfacc_estimator.update_every_n_steps(step=step, 
											   			occ_eval_fn=self.occ_eval_fn, 
														ema_decay=1.0 - self.occ_decay, # compute ema decay (exponential moving average decay)
														warmup_steps = self.occ_warmup,
														n = 1)

	def ray_aabb_intersect(self, rays_o: torch.Tensor, rays_d: torch.Tensor, aabb_min=-1.0, aabb_max=1.0):
		'''
		compute near/far t for AABB [-1,1]^3.
		rays_o: (N,3)  
		rays_d: (N,3)
		returns t_near, t_far with shape (N,)
		'''
		inv_d = 1.0 / torch.clamp(rays_d, min=-1e10, max=1e10) # reciprocal clamping t = (x - o) / d
		t0s = (aabb_min - rays_o) * inv_d
		t1s = (aabb_max - rays_o) * inv_d
		tmin = torch.minimum(t0s, t1s).amax(dim=-1) # min for negative rays, max for first time when all 3 comp inside box
		tmax = torch.maximum(t0s, t1s).amin(dim=-1) # max for negative rays, min for first time 1 comp leaves box
		return tmin, tmax

	def uniform_ray_marching(self, rays_o: torch.Tensor, 
						  		   rays_d: torch.Tensor, 
								   t_min: torch.Tensor,
								   t_max: torch.Tensor,
								   render_step_size):
		'''
		for machines that do not support nerfacc, just do simple uniform ray marching

		[INPUTS]
		rays_o: (N, 3), origin of rays
		rays_d: (N, 3), direction of rays
		t_min:  (N,),   t_near for each ray
		t_max:  (N,),   t_far for each ray
		render_step_size: step size between pts on ray  

		[OUTPUTS]
		t_s: (M,), start t for each point for each ray, flattened
		t_e: (M,), end t for each point
		ray_idx: (M,), ray_index (0, N) for each t pt
		'''
		# sanity checks
		assert torch.all(t_max > t_min), 't_max must be greater than t_min for all rays'

		# compute how many steps per ray (ceil)
		n_steps = torch.clamp(((t_max - t_min) / render_step_size).ceil().long(), min=1)
		max_steps = int(n_steps.max().item())

		if self.forward_debug_on:
			self.logger.debug('performing uniform ray marching... ')
			self.logger.debug(f'n_steps: {n_steps}')
			self.logger.debug(f'max steps: {max_steps}')

		# for each ray, find valid # steps to take: build [0, 1, ..., max_steps-1] and mask per ray
		idx = torch.arange(max_steps, device=rays_o.device)[None, ...]  # (1, max_steps)
		n_steps_exp = n_steps[..., None]                                # (Nv, 1)
		mask = idx < n_steps_exp                                        # (Nv, max_steps)

		if self.forward_debug_on:
			self.logger.debug(f'idx shape: {idx.shape}')
			self.logger.debug(f'n_steps_exp shape: {n_steps_exp.shape}')
			self.logger.debug(f'mask shape: {mask.shape}')
			#self.logger.debug(f'mask min, max: {mask.min(), mask.max()}')
			#mask_numpy = mask.detach().cpu().numpy().astype(np.uint8)
			#plt.plot(mask_numpy[0,:])
			#plt.plot(mask_numpy[1,:])
			#plt.show()

		# start time for each "point": a pt is really a small interval [t_s, t_e] with delta step_size
		t0 = t_min[..., None]                                            # (Nv, 1)
		t_s = t0 + idx * render_step_size                                # (Nv, max_steps)
		t_s = t_s[mask]                                                  # flatten valid only

		# end time (clamped) for each "point"
		tf = t_max[..., None]
		t_e = t0 + (idx+1) * render_step_size     
		t_e = torch.minimum(t_e, tf) # clip by max, just in case last step goes over the t_max for that ray                           
		t_e = t_e[mask]

		# if self.forward_debug_on:
		# 	self.logger.debug(f't shape: {t.shape}')

		# construct ray index for each t value in flattened array (tag each pt for LOS integration)
		ray_idx = torch.arange(rays_o.shape[0], device=rays_o.device)
		ray_idx = ray_idx[..., None].expand(-1, max_steps)[mask]      

		# if self.forward_debug_on:
		# 	self.logger.debug(f'ray_idx shape: {ray_idx.shape}')
		# 	# ray_idx_np = ray_idx.detach().cpu().numpy()
		# 	# plt.plot(ray_idx_np)
		# 	# plt.show()

		# return t (pt samples) and ray_idx (labels for each pt)
		return ray_idx, t_s, t_e

	def _sample_points_along_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
		'''
		given rays_o and rays_d, computes 
		[1] t_near, t_far
		[2] compute [t_s, t_e] for each "interval" on ray to be integrated
		'''

		# only call this from forward, so strict types is already checked

		# compute t_near, t_far for all rays
		t_near, t_far = self.ray_aabb_intersect(rays_o, rays_d, self.aabb_min, self.aabb_max)
		valid = t_far > t_near # sanity check, although most rays should be valid

		if self.forward_debug_on: # can use in helpers too, such as uniform_ray_marching
			self.logger.debug(f'rays_o: {rays_o}')
			self.logger.debug(f'rays_d: {rays_d}')
			self.logger.debug(f't_near, t_far: {t_near, t_far}')

		# add check for no valid (t_near, t_far) rays
		if not valid.any(): # make sure to return same as other cases
			t_s = torch.zeros((0,), dtype=self.dtype, device=self.device)
			t_e = torch.zeros((0,), dtype=self.dtype, device=self.device)
			ray_idx = torch.zeros((0,), dtype = torch.long, device=self.device) # need torch.long for index_add_ later in forward
			return t_s, t_e, ray_idx, valid
		
		# ray marching for points, grab [t_s, t_e] for each point along each ray
		if self.cfg['renderer_model'] == 'nerfacc': # adaptive sampling along rays
			# check
			nerfacc_is_available()
	
			# run adaptive sampling
			ray_idx, t_s, t_e = self.nerfacc_estimator.sampling(
									rays_o=rays_o[valid], 
									rays_d=rays_d[valid],
									t_min=t_near[valid],  
									t_max=t_far[valid], 
									render_step_size=self.step_size)
	
			ray_idx = ray_idx.to(dtype = torch.long) # typing convert

			# uniform fallback check
			if self.enable_fallback:
				Nv = int(valid.sum().item())
				counts = torch.bincount(ray_idx, minlength = Nv) # get number of samples of all Nv rays
				low = counts < self.fallback_when_samples # low mask

				if low.any():
					low_ids = torch.nonzero(low, as_tuple=False).squeeze(-1) # get nonzero indices
					Nl = low_ids.numel()

					# build uniform with min_samples_per_ray on [t_near, t_far]
					t0 = t_near[valid][low_ids] # (Nl,)
					t1 = t_far[valid][low_ids] # (Nl,)
					K = self.min_samples_per_ray

					# (Nl, K) starts
					k = torch.arange(K, device = self.device, dtype = self.dtype)[None, :] # (1, K)
					dt = ((t1 - t0) / K).to(self.dtype)[:, None]                           # (Nl, 1)
					t_s_fb = t0[:, None] + k * dt                                          # (Nl, K)
					t_e_fb = t_s_fb + dt                                                   # (Nl, K)
					# numerical safety: clamp final to t1
					t_e_fb[:,-1] = t1

					# packed flatten
					t_s_fb = t_s_fb.reshape(-1)
					t_e_fb = t_e_fb.reshape(-1)

					# get ids
					ray_idx_fb = low_ids[:, None].expand(Nl, K).reshape(-1).to(torch.long)

					# concat with nerfacc samples, discard samples of rays with low counts
					ok_mask = ~low[ray_idx] # broadcast fill ray_idx shape (M,) with 0's or 1's depending on low mask, where low[ray_idx][i] = low[ ray_idx[i] ]
					ray_idx_ok = ray_idx[ok_mask] # apply mask
					t_s_ok = t_s[ok_mask]
					t_e_ok = t_e[ok_mask]

					ray_idx = torch.cat([ray_idx_ok, ray_idx_fb], dim = 0)
					t_s = torch.cat([t_s_ok, t_s_fb], dim = 0)
					t_e = torch.cat([t_e_ok, t_e_fb], dim = 0)
			
		elif self.cfg['renderer_model'] == 'uniform': # uniform sampling along rays
			ray_idx, t_s, t_e = self.uniform_ray_marching(
									rays_o=rays_o[valid], 
									rays_d=rays_d[valid],
									t_min=t_near[valid],  
									t_max=t_far[valid], 
									render_step_size=self.step_size)
		else:
			raise ValueError("Undefined cfg renderer_model")
		
		if self.forward_debug_on:
			self.logger.debug(f't_s shape: {t_s.shape}')
			self.logger.debug(f't_e shape: {t_e.shape}')
			self.logger.debug(f'ray_idx shape: {ray_idx.shape}')

		return t_s, t_e, ray_idx, valid

	@torch.no_grad()
	def _sample_uniform_shell(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
							  n_samples: int, r_max_rsun: Union[float, None] = None):
		'''
		Dense uniform sampler for the VISUALIZATION LOS integral. Clips each ray to the box and,
		if r_max_rsun is given, to the outer sphere r <= r_max_rsun, then lays n_samples equal
		sub-segments across that interval. This replaces the coarse nerfacc step
		(step_size*aabb_scale ~ 0.117 R_sun) that aliases the steep base into concentric
		"sampling rings" once the r>=r_min cut is applied.
		Returns (t_s, t_e, ray_idx, valid) in the SAME format as _sample_points_along_rays.
		'''
		# box interval [t_near, t_far]
		t_near, t_far = self.ray_aabb_intersect(rays_o, rays_d, self.aabb_min, self.aabb_max)
		ts0 = t_near.clone(); ts1 = t_far.clone()
		ok = t_far > t_near

		# clip to outer sphere r <= r_max (AABB units): |o + t d|^2 = (r_max/aabb_scale)^2
		if r_max_rsun is not None:
			rho = float(r_max_rsun) / self.aabb_scale
			A = (rays_d * rays_d).sum(-1)
			B = 2.0 * (rays_o * rays_d).sum(-1)
			C = (rays_o * rays_o).sum(-1) - rho * rho
			disc = B * B - 4.0 * A * C
			hit = disc > 0
			sq = torch.sqrt(disc.clamp_min(0.0))
			A_safe = A.clamp_min(1e-20)
			t_in  = (-B - sq) / (2.0 * A_safe)
			t_out = (-B + sq) / (2.0 * A_safe)
			ts0 = torch.maximum(ts0, t_in)
			ts1 = torch.minimum(ts1, t_out)
			ok = ok & hit

		valid = ok & (ts1 > ts0)

		# uniform sub-segments for valid rays only (ray_idx indexes into rays[valid])
		a = ts0[valid]; b = ts1[valid]                                   # (Nv,)
		Nv = int(a.shape[0])
		if Nv == 0:
			z = torch.zeros((0,), dtype=self.dtype, device=self.device)
			return z, z, torch.zeros((0,), dtype=torch.long, device=self.device), valid
		u = torch.linspace(0.0, 1.0, int(n_samples) + 1, dtype=self.dtype, device=self.device)  # (n+1,)
		edges = a[:, None] + (b - a)[:, None] * u[None, :]               # (Nv, n+1)
		t_s = edges[:, :-1].reshape(-1)                                  # (Nv*n,)
		t_e = edges[:, 1:].reshape(-1)
		ray_idx = torch.arange(Nv, device=self.device, dtype=torch.long).repeat_interleave(int(n_samples))
		return t_s, t_e, ray_idx, valid
	
	def _forward_lut(self, rays_o: torch.Tensor, 
						   rays_d: torch.Tensor,
						   t_s: torch.Tensor, 
						   t_e: torch.Tensor, 
						   ray_idx: torch.Tensor, 
						   valid: torch.Tensor,
						   viz_radial_cutoff: bool = False,
				 		   viz_r_min_render_rsun: Union[float, None] = None,
				 		   viz_r_max_render_rsun: Union[float, None] = None) -> torch.Tensor:
		'''
		Does the actual LOS integration

		inputs:
		rays_o: (N, 3), origin coordinate
		rays_d: (N, 3), normalized direction
		t_s: (M,), startpt of segment
		t_e: (M,), endpt of segment
		ray_idx: (M,), which ray segment belongs to
		valid: (N,) which rays are valid rays

		outputs: 
		full: (N, C), LOS integrated emissivity
		'''
		##############################################
		# compute coordinates to sample from
		##############################################
		t_mid = 0.5 * (t_s + t_e)
		# [valid] gets valid rays, [ray_idx] gets copies of each ray for each pt
		x_AABB = rays_o[valid][ray_idx] + t_mid[...,None] * rays_d[valid][ray_idx]
		x_xyz = x_AABB * self.aabb_scale

		# optional visualize-only radial cutoff in 3D render volume
		if viz_radial_cutoff:
			sample_keep = radial_sample_mask(
				x_xyz,
				r_min_rsun = viz_r_min_render_rsun,
				r_max_rsun = viz_r_max_render_rsun,
			)

			if self.forward_debug_on:
				n_keep = int(sample_keep.sum().item())
				n_tot = int(sample_keep.numel())
				self.logger.debug(
					f'viz radial cutoff active: keeping {n_keep}/{n_tot} samples '
					f'in [{viz_r_min_render_rsun}, {viz_r_max_render_rsun}] R_sun'
				)
		else:
			sample_keep = None

		if self.forward_debug_on:
			self.logger.debug(f'x_AABB shape: {x_AABB.shape}')

		##########################################
		### forward pass to get (ne, T)
		##########################################

		# guard for case when we use gt ne but try to reconstruct ne and t
		if self.use_gt_ne and self.reconstruction_target == 'ne_t':
			self.logger.warning(
				'use_gt_ne=True with reconstruction_target="ne_t": '
				'density comes from GT while temperature comes from model/GT depending on branch.'
			)

		# choose learned vs gt density/temperature sources
		if self.use_gt_ne and (self.ne_field is not None):
			# use ground truth ne
			log_ne = self.ne_field(x_xyz)
			if self.clamp:
				log_ne = log_ne.clamp(self.log_ne_min, self.log_ne_max)

			# cant predict only temp
			if self.reconstruction_target == 'ne_t':
				raise ValueError('use_gt_ne with reconstruction_target="ne_t" is not supported cleanly yet')

			# use ground truth temp
			log_temp = self.temp_field(x_xyz)
			if self.clamp:
				log_temp = log_temp.clamp(self.log_temp_min, self.log_temp_max)

		else:
			# get predicted ne and temp
			_, log_ne, log_temp_pred = self._predict_learned_fields(x_AABB)

			if self.reconstruction_target == 'ne_t':
				# temp_pred should be used
				if log_temp_pred is None:
					raise ValueError('Model did not return log_temp for reconstruction_target="ne_t"')
				log_temp = log_temp_pred
			elif self.reconstruction_target == 'ne':
				# we use gt temp, only predict density
				log_temp = self.temp_field(x_xyz)
				if self.clamp:
					log_temp = log_temp.clamp(self.log_temp_min, self.log_temp_max)
			else:
				raise ValueError(f'Invalid reconstruction target: {self.reconstruction_target}')

		# compute log_r
		r = torch.linalg.vector_norm(x_xyz, dim=-1)
		log_r = torch.log10(r)
		
		# clamp log_r
		if self.clamp:
			#log_temp = log_temp.clamp(self.log_temp_min, self.log_temp_max)
			log_r    = log_r.clamp(self.log_r_min, self.log_r_max)

		if self.forward_debug_on:
			self.logger.debug(f'log_temp bounds: {log_temp.min(), log_temp.max()}')
			self.logger.debug(f'log_r bounds: {log_r.min(), log_r.max()}')

		##############################################
		### compute emissivities
		##############################################

		# compute ccoef
		ntr_in = torch.stack((log_ne, log_temp, log_r), axis = 0)
		ntr_in = torch.transpose(ntr_in, 0, 1) # make rows samples

		if self.forward_debug_on:
			self.logger.debug(f'ntr_in shape: {ntr_in.shape}')

		# in ergs s^-1 cm^-3 sr^-1
		ccoef = self.ccoef_field(ntr_in)
		# transpose back
		ccoef = torch.transpose(ccoef, 0, 1)

		if self.forward_debug_on:
			self.logger.debug(f'ccoef shape: {ccoef.shape}')

		#############################################
		### LOS integrate
		#############################################

		# next, need to map everything to physically-consistent units

		# compute dI = eps * ds at each pt
		seg_len_phys = (t_e - t_s)[..., None] * self.aabb_scale * self.cfg.get('rsun_to_cm', 6.957e10)  # (M, 1)

		# sr to arcsec^2
		sr2arcsec = (180.0/np.pi)**2 * 3600.0**2 # scalar

		# ph to ergs
		ph2ergs = 6.62607015e-27 * 3e10 / (np.array(self.cfg.get('wavelengths', [])) * 1e-8)
		ph2ergs = torch.from_numpy(ph2ergs).to(device = self.device, dtype = self.dtype)

		# scale, and broadcast the photons to ergs for each channel
		epsI = ccoef / ph2ergs[None,:] / sr2arcsec

		# get physically consistent integrand
		dI = epsI * seg_len_phys # (M, 2), broadcast to each intensity channel

		# visualization radial mask: zero out contributions from samples outside the selected radial range
		if sample_keep is not None:
			dI = dI * sample_keep[..., None].to(dI.dtype)

		if self.forward_debug_on:
			self.logger.debug(f'dI shape: {dI.shape}')

		# integrate dI
		# out (# valid rays to integrate, # intensity channels)
		out = torch.zeros((valid.sum(), dI.shape[-1]), dtype = self.dtype, device=self.device)

		if self.forward_debug_on:
			self.logger.debug(f'out shape before integration: {out.shape}')

		# integrate each ray using ray_idx to select pts to add from dI, along first dimension
		out.index_add_(0, ray_idx, dI)

		if self.forward_debug_on:
			self.logger.debug(f'out shape after integration: {out.shape}')

		# fill into full output
		full = torch.zeros((rays_o.shape[0], dI.shape[-1]), dtype = self.dtype, device=self.device)
		full[valid] = out

		if self.forward_debug_on:
			self.logger.debug(f'full shape: {full.shape}')

		return full

	def forward(self, 
				rays_o: torch.Tensor, 
				rays_d: torch.Tensor,
				viz_radial_cutoff: bool = False,
				viz_r_min_render_rsun: Union[float, None] = None,
				viz_r_max_render_rsun: Union[float, None] = None) -> torch.Tensor:
		'''
		Given ray, do ray marching and LOS integration of emissivity
		'''
		# type check for rays_o and rays_d
		if self.strict_types:
			assert self.dtype == rays_o.dtype, \
				   f"model and rays_o dtype mismatch: model={self.dtype}, rays_o={rays_o.dtype}"
			assert self.dtype == rays_d.dtype, \
				   f"model and rays_d dtype mismatch: model={self.dtype}, rays_d={rays_d.dtype}"
			assert self.device == rays_o.device, \
				   f"model and rays_o device mismatch: model={self.device}, rays_o={rays_o.device}"
			assert self.device == rays_d.device, \
				   f"model and rays_d device mismatch: model={self.device}, rays_d={rays_d.device}"
		else:
			rays_o = rays_o.to(dtype = self.dtype, device = self.device)
			rays_d = rays_d.to(dtype = self.dtype, device = self.device)

		# count step
		self._step += 1
		self.forward_debug_on = self.debug_every and (self._step % self.debug_every == 0) and \
			  self.logger.isEnabledFor(logging.DEBUG)
		
		# compute points to integrate
		t_s, t_e, ray_idx, valid = self._sample_points_along_rays(rays_o, rays_d)

		# actual LOS integration
		return self._forward_lut(
			rays_o, 
			rays_d, 
			t_s, 
			t_e, 
			ray_idx, 
			valid,
			viz_radial_cutoff = viz_radial_cutoff,
			viz_r_min_render_rsun = viz_r_min_render_rsun,
			viz_r_max_render_rsun = viz_r_max_render_rsun,
		)

	def render(self, 
			   rays_o: torch.Tensor, 
			   rays_d: torch.Tensor, 
			   backend: str = 'lut',
			   viz_radial_cutoff: bool = False,
			   viz_r_min_render_rsun: Union[float, None] = None,
			   viz_r_max_render_rsun: Union[float, None] = None
			) -> torch.Tensor:
		'''
		A more generic render function that allows us to switch the backend forward model
		'''

		# type check for rays_o and rays_d
		if self.strict_types:
			assert self.dtype == rays_o.dtype, \
				   f"model and rays_o dtype mismatch: model={self.dtype}, rays_o={rays_o.dtype}"
			assert self.dtype == rays_d.dtype, \
				   f"model and rays_d dtype mismatch: model={self.dtype}, rays_d={rays_d.dtype}"
			assert self.device == rays_o.device, \
				   f"model and rays_o device mismatch: model={self.device}, rays_o={rays_o.device}"
			assert self.device == rays_d.device, \
				   f"model and rays_d device mismatch: model={self.device}, rays_d={rays_d.device}"
		else:
			rays_o = rays_o.to(dtype = self.dtype, device = self.device)
			rays_d = rays_d.to(dtype = self.dtype, device = self.device)

		# backend switch
		if backend == 'lut':
			return self.forward(
				rays_o, 
				rays_d,
				viz_radial_cutoff = viz_radial_cutoff,
				viz_r_min_render_rsun = viz_r_min_render_rsun,
				viz_r_max_render_rsun = viz_r_max_render_rsun,
			)
		else:
			raise ValueError(f'Unkown backend {backend}')


	@torch.no_grad()
	def los_integral(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
					 quantity: str = "column_ne", use_gt: bool = False,
					 r_min_rsun: Union[float, None] = 1.0,
					 r_max_rsun: Union[float, None] = None,
					 uniform_samples: int = 0,
					 channels: Union[list, tuple, int, None] = None) -> torch.Tensor:
		'''
		LOS-integrate a scalar along rays for VISUALIZATION. Reuses the intensity path
		(same sampling, fields, LUT, and ds convention as _forward_lut).
		quantity: 'column_ne' = ∫ n_e ds ; 'emission' = ∫ eps ds (summed channels) ;
		          'ewt_temp' = ∫ T eps ds / ∫ eps ds (emission-weighted temperature).
		Returns (N,) per input ray; invalid rays -> NaN.
		'''
		rays_o = rays_o.to(dtype=self.dtype, device=self.device)
		rays_d = rays_d.to(dtype=self.dtype, device=self.device)

		# generate uniform samples if asked
		if uniform_samples and int(uniform_samples) > 0:
			t_s, t_e, ray_idx, valid = self._sample_uniform_shell(
				rays_o, rays_d, int(uniform_samples), r_max_rsun=r_max_rsun)
		else:
			t_s, t_e, ray_idx, valid = self._sample_points_along_rays(rays_o, rays_d)

		
		t_mid = 0.5 * (t_s + t_e)
		x_AABB = rays_o[valid][ray_idx] + t_mid[..., None] * rays_d[valid][ray_idx]
		x_xyz = x_AABB * self.aabb_scale

		# --- restrict the integral to the physically-valid corona shell ---
		# GT/emissivity fields are defined for r >= 1 R_sun (grid inner boundary). Below r=1 the
		# spherical interpolant border-pads the dense chromospheric base, so integrating the full
		# box manufactures a saturated filled disk. Zero out-of-shell samples' contribution.
		r_samp = torch.linalg.vector_norm(x_xyz, dim=-1)
		keep_w = torch.ones_like(r_samp)
		if r_min_rsun is not None:
			keep_w = keep_w * (r_samp >= float(r_min_rsun)).to(self.dtype)
		if r_max_rsun is not None:
			keep_w = keep_w * (r_samp <= float(r_max_rsun)).to(self.dtype)

		# fields at sample points (GT or learned) — same sources as _forward_lut
		if use_gt and (self.ne_field is not None):
			log_ne = self.ne_field(x_xyz)
			log_temp = self.temp_field(x_xyz) if self.temp_field is not None else None
		else:
			_, log_ne, log_temp = self._predict_learned_fields(x_AABB)
			if log_temp is None and self.temp_field is not None:
				log_temp = self.temp_field(x_xyz)                      # target='ne' -> GT temp
		if self.clamp:
			log_ne = log_ne.clamp(self.log_ne_min, self.log_ne_max)
			if log_temp is not None:
				log_temp = log_temp.clamp(self.log_temp_min, self.log_temp_max)

		# physical segment length ds (cm) — identical convention to _forward_lut
		seg_len = (t_e - t_s) * self.aabb_scale * self.cfg.get('rsun_to_cm', 6.957e10)   # (M,)
		Nvalid = int(valid.sum().item())
		# integral helper
		def integ(vals):
			out = torch.zeros(Nvalid, dtype=self.dtype, device=self.device)
			out.index_add_(0, ray_idx, vals * seg_len * keep_w) # keep_w mask on valid pixels
			return out

		# case on integrand
		if quantity == "column_ne":
			per_ray = integ(torch.pow(10.0, log_ne))
		elif quantity in ("emission", "ewt_temp"):
			r = torch.linalg.vector_norm(x_xyz, dim=-1); log_r = torch.log10(r)
			if self.clamp: log_r = log_r.clamp(self.log_r_min, self.log_r_max)
			ntr = torch.stack((log_ne, log_temp, log_r), axis=0).transpose(0, 1)
			ccoef_all = self.ccoef_field(ntr).transpose(0, 1)         # (M, C) per-channel emissivity
			if channels is not None:                                  # select spectral line(s); None -> all (total)
				ci = channels if isinstance(channels, (list, tuple)) else [channels]
				ci = torch.as_tensor([int(c) for c in ci], device=ccoef_all.device)
				ccoef_all = ccoef_all.index_select(-1, ci)
			eps = ccoef_all.sum(-1)                                   # (M,) summed over selected channels
			if quantity == "emission":
				per_ray = integ(eps)
			else:
				num = integ(torch.pow(10.0, log_temp) * eps); den = integ(eps)
				per_ray = num / den.clamp_min(1e-30)                  # ds cancels in the ratio                  # ds cancels in the ratio
		else:
			raise ValueError(f"unknown los_integral quantity: {quantity}")

		full = torch.full((rays_o.shape[0],), float('nan'), dtype=self.dtype, device=self.device)
		full[valid] = per_ray
		return full