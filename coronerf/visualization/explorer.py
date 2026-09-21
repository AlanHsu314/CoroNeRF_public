##########################################
### POST-TRAINING VIDEO RENDERING
##########################################


from __future__ import annotations
from typing import Union

import logging
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import torch

from ..data.camera import make_camera_rays_from_xyz

from ..artifacts.scalar_fields import get_scalar_field_spec, build_scalar_shell_product
from ..artifacts.plots import save_scalar_shell_triptych

from .geometry import xyz_to_llr, llr_to_xyz_np, apply_rmax_mask, make_coronal_mask_simple

# helpers
def create_video(frames: np.ndarray, file_path: Path, fps: int = 20):
	num_frames = frames.shape[0]
	H, W = frames.shape[1:3]

	out = cv2.VideoWriter(str(file_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
	for i in range(num_frames):
		out.write(frames[i])
	out.release()

def fig_to_numpy(fig):
	# Render a Matplotlib figure to a HxWx3 uint8 NumPy array (RGB).
	fig.canvas.draw()

	# RGBA uint8 view, shape (H, W, 4)
	buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)

	# Drop alpha -> RGB, then convert to BGR for OpenCV
	return buf[:, :, :3][:, :, ::-1].copy()

class CoroNeRFExplorer:
	def __init__(self, state, ckpt_path: Path, out_dir_name: str = 'visualize', viz_cfg: Union[dict, None] = None):
		self.logger = logging.getLogger('coroNeRF.visualization.explorer')
		self.state = state
		self.ckpt_path = Path(ckpt_path)

		# visualization config
		self.viz_cfg = viz_cfg or {}

		vol_cfg = self.viz_cfg.get('render_volume', {})
		if vol_cfg.get('enable_radial_cutoff', False):
			r_min_render = vol_cfg.get('r_min_render_rsun', None)
			r_max_render = vol_cfg.get('r_max_render_rsun', None)

			self.logger.info(
				f'Using renderer-side visualize radial cutoff: '
				f'r_min={r_min_render}, r_max={r_max_render} R_sun'
			)


		# create dirs
		self.run_dir = Path(self.state.dirs.run_dir)
		self.out_dir = self.run_dir / out_dir_name
		self.out_dir.mkdir(parents=True, exist_ok=True)

		self.video_dir = self.out_dir / 'videos'
		self.video_dir.mkdir(parents=True, exist_ok=True)

		with open(self.out_dir / 'visualize_meta.json', 'w') as f:
			json.dump({
				'run_dir': str(self.run_dir),
				'ckpt_path': str(self.ckpt_path),
			}, f, indent=2)

	#########################################################
	### intensity rendering
	#########################################################
	def render_viewpoint(self,
						 lon: float = 0.0,
						 lat: float = 0.0,
						 r: float = 215.0,
						 H: int = 256,
						 W: int = 256,
						 fov_rsun: float = 3.0,
						 chunk_rays: int = 8192):
		'''
		render one synthetic viewpoint from selected observer llr
		'''
		obs_xyz = llr_to_xyz_np(lon, lat, r=r)

		fov_rad = 2.0 * np.arctan2(fov_rsun, r)

		rays_o_np, rays_d_np = make_camera_rays_from_xyz(
			obs_xyz = obs_xyz,
			H = H,
			W = W,
			fov = fov_rad,
			aabb_scale = self.state.renderer.aabb_scale,
		)

		rays_o = torch.from_numpy(rays_o_np).to(
			device = self.state.renderer.device,
			dtype = self.state.renderer.dtype,
		)
		rays_d = torch.from_numpy(rays_d_np).to(
			device = self.state.renderer.device,
			dtype = self.state.renderer.dtype,
		)

		N = H * W
		outs = []

		# renderer volume configs
		vol_cfg = self.viz_cfg.get('render_volume', {})
		viz_radial_cutoff = vol_cfg.get('enable_radial_cutoff', False)
		viz_r_min_render_rsun = vol_cfg.get('r_min_render_rsun', None)
		viz_r_max_render_rsun = vol_cfg.get('r_max_render_rsun', None)

		with torch.no_grad():
			for start in range(0, N, chunk_rays):
				end = min(start + chunk_rays, N)
				I_chunk = self.state.renderer(
					rays_o[start:end], 
					rays_d[start:end],
					viz_radial_cutoff = viz_radial_cutoff,
					viz_r_min_render_rsun = viz_r_min_render_rsun,
					viz_r_max_render_rsun = viz_r_max_render_rsun,
				)
				outs.append(I_chunk.detach().cpu())

		I_flat = torch.cat(outs, dim=0)
		I = I_flat.view(H, W, -1).numpy()

		return I

	def _save_orbit_video(self,
						  video_path: Path,
						  raw_frames: np.ndarray,
						  fov_rsun: float,
						  rmax_vis_rsun = None,
						  apply_coronal_mask: bool = True,
						  also_save_unmasked: bool = False,
						  r_min_mask_rsun: float = 1.0):
		'''
		raw_frames: (T, H, W), single channel
		'''
		def _process_frames(frames_in: np.ndarray, do_mask: bool):
			processed = []

			H, W = frames_in.shape[1:3]
			if do_mask:
				mask2d = make_coronal_mask_simple(
					H = H,
					W = W,
					fov_rsun = fov_rsun,
					r_min_rsun = r_min_mask_rsun,
					r_max_rsun = rmax_vis_rsun,
				)
			else:
				mask2d = None

			for frame in frames_in:
				frame_proc = frame.copy()

				# optional projected-radius clipping
				if rmax_vis_rsun is not None and not do_mask:
					frame_proc = apply_rmax_mask(frame_proc, rmax_vis_rsun, fov_rsun)

				# optional annular mask
				if mask2d is not None:
					floor_val = np.nanmin(frame_proc)
					frame_proc[~mask2d] = floor_val

				processed.append(frame_proc)

			return np.stack(processed)

		def _write_one(frames_in: np.ndarray, out_path: Path):
			frames_rgb = np.zeros((*frames_in.shape, 3), dtype=np.uint8)

			all_vals = frames_in[np.isfinite(frames_in) & (frames_in > 0)]
			if all_vals.size == 0:
				raise ValueError("No positive finite values found for orbit video scaling.")

			vmin = np.percentile(all_vals, 1)
			vmax = np.percentile(all_vals, 99)

			for i, frame in enumerate(frames_in):
				fig, ax = plt.subplots(figsize=(5, 5))
				ax_img = ax.imshow(frame, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno')
				ax.set_axis_off()
				frame_rgb = ax_img.make_image(renderer=None, unsampled=True)[0][:, :, :-1]
				plt.close(fig)

				frames_rgb[i] = frame_rgb[:, :, ::-1]

			create_video(frames_rgb, out_path)

		# default masked output
		if apply_coronal_mask:
			processed_masked = _process_frames(raw_frames, do_mask=True)
			_write_one(processed_masked, video_path)
		else:
			processed_unmasked = _process_frames(raw_frames, do_mask=False)
			_write_one(processed_unmasked, video_path)

		# optional extra unmasked output
		if apply_coronal_mask and also_save_unmasked:
			video_path_unmasked = video_path.with_name(video_path.stem + '_unmasked' + video_path.suffix)
			processed_unmasked = _process_frames(raw_frames, do_mask=False)
			_write_one(processed_unmasked, video_path_unmasked)

	def make_microwave_orbit(self,
							 lat: float = 0.0,
							 obs_r: float = 215.0,
							 n_frames: int = 120,
							 H: int = 512,
							 W: int = 512,
							 fov_rsun: float = 3.0,
							 rmax_vis_rsun = None,
							 chunk_rays: int = 8192,
							 apply_coronal_mask: bool = True,
							 also_save_unmasked: bool = False,
							 r_min_mask_rsun: float = 1.0):
		out_dir = self.video_dir / 'microwave_orbit'
		out_dir.mkdir(parents=True, exist_ok=True)

		lons = np.linspace(0, 2*np.pi, n_frames, endpoint=False, dtype=np.float64)
		lats = np.full_like(lons, lat, dtype=np.float64)
		rs = np.full_like(lons, obs_r, dtype=np.float64)

		llrs = np.stack([lons, lats, rs], axis=-1)

		line_names = self.state.cfg.ion_kwargs.get('line_names', [])
		C = len(line_names)
		I_frames = np.zeros((n_frames, H, W, C), dtype=np.float32)

		for i, (lon, lat, r) in enumerate(llrs):
			self.logger.info(f'rendering microwave orbit frame {i+1}/{n_frames}')
			I_frames[i] = self.render_viewpoint(
				lon = float(lon),
				lat = float(lat),
				r = float(r),
				H = H,
				W = W,
				fov_rsun = fov_rsun,
				chunk_rays = chunk_rays,
			)

		for c, line_name in enumerate(line_names):
			video_path = out_dir / f'microwave_orbit_{line_name}.avi'
			self._save_orbit_video(
				video_path = video_path,
				raw_frames = I_frames[:, :, :, c],
				fov_rsun = fov_rsun,
				rmax_vis_rsun = rmax_vis_rsun,
				apply_coronal_mask = apply_coronal_mask,
				also_save_unmasked = also_save_unmasked,
				r_min_mask_rsun = r_min_mask_rsun,
			)

	def make_tilted_orbit(self,
						  lon_c: float = 0.0,
						  lat_c: float = 0.0,
						  alpha_deg: float = 10.0,
						  obs_r: float = 215.0,
						  n_frames: int = 120,
						  H: int = 512,
						  W: int = 512,
						  fov_rsun: float = 2.5,
						  rmax_vis_rsun = 2.5,
						  chunk_rays: int = 8192,
						  apply_coronal_mask: bool = True,
						  also_save_unmasked: bool = False,
						  r_min_mask_rsun: float = 1.0):
		out_dir = self.video_dir / 'tilted_orbit'
		out_dir.mkdir(parents=True, exist_ok=True)

		alpha = np.deg2rad(alpha_deg)

		c = llr_to_xyz_np(lon_c, lat_c, r=1.0)
		c = c / np.linalg.norm(c)

		ref = np.array([0.0, 0.0, 1.0])
		if np.allclose(np.cross(ref, c), 0.0):
			ref = np.array([0.0, 1.0, 0.0])

		u = np.cross(ref, c)
		u = u / np.linalg.norm(u)
		v = np.cross(c, u)

		thetas = np.linspace(0, 2*np.pi, n_frames, endpoint=False)
		llrs = np.zeros((n_frames, 3), dtype=np.float64)

		for i, th in enumerate(thetas):
			d = np.cos(alpha) * c + np.sin(alpha) * (np.cos(th) * u + np.sin(th) * v)
			llrs[i] = xyz_to_llr(*d)

		llrs[:, -1] = obs_r

		line_names = self.state.cfg.ion_kwargs.get('line_names', [])
		C = len(line_names)
		I_frames = np.zeros((n_frames, H, W, C), dtype=np.float32)

		for i, (lon, lat, r) in enumerate(llrs):
			self.logger.info(f'rendering tilted orbit frame {i+1}/{n_frames}')
			I_frames[i] = self.render_viewpoint(
				lon = float(lon),
				lat = float(lat),
				r = float(r),
				H = H,
				W = W,
				fov_rsun = fov_rsun,
				chunk_rays = chunk_rays,
			)

		for c, line_name in enumerate(line_names):
			video_path = out_dir / f'tilted_orbit_{line_name}.avi'
			self._save_orbit_video(
				video_path = video_path,
				raw_frames = I_frames[:, :, :, c],
				fov_rsun = fov_rsun,
				rmax_vis_rsun = rmax_vis_rsun,
				apply_coronal_mask = apply_coronal_mask,
				also_save_unmasked = also_save_unmasked,
				r_min_mask_rsun = r_min_mask_rsun,
			)

	#########################################################
	### shell rendering
	#########################################################
	def render_shell(self, quantity: str, r_target: float = 1.5):
		'''
		returns a rendered comparison frame (GT / pred / residual)
		'''
		if (quantity == 'ne') and (self.state.ne_field is None):
			raise ValueError("ne_field not available; cannot render shell comparison")

		if (quantity == 'temp'):
			if (self.state.temp_field is None):
				raise ValueError("temp_field not available; cannot render shell comparison")
			
			if (self.state.shared.get('reconstruction_target', 'ne') != 'ne_t'):
				raise ValueError('temperature render_shell requires reconstruction_target = ne_t')

		product = build_scalar_shell_product(self.state, quantity = quantity, r_target = r_target)
		frame = save_scalar_shell_triptych(product, out_path = None, save = False)
		return frame

	def make_shell_video(self, quantity: str,
					           r_min: float = 1.1, 
							   r_max: float = 3.0, 
							   n_radii: int = 120):
		'''
		generic shell video maker
		'''
		spec = get_scalar_field_spec(quantity)
		out_dir = self.video_dir / spec.video_dirname
		out_dir.mkdir(parents=True, exist_ok=True)

		r_arr = np.linspace(r_min, r_max, n_radii)
		frames = []

		for i, r in enumerate(r_arr):
			if i % 10 == 0:
				self.logger.info(f'rendering {spec.eval_prefix} frame {i+1}/{n_radii}')
			frames.append(self.render_shell(quantity, r_target=float(r)))

		frames = np.stack(frames, axis=0).astype(np.uint8)
		create_video(frames, out_dir / f'{spec.eval_prefix}_movie.avi')
	
	
	##########################################
	### dispatch
	##########################################
	def run_from_config(self, viz_cfg: dict):
		if viz_cfg.get('make_density_shell_video', False):
			cfg = viz_cfg.get('density_shell_video', {})
			self.make_shell_video(
				quantity = 'ne',
				r_min = cfg.get('r_min', 1.1),
				r_max = cfg.get('r_max', 3.0),
				n_radii = cfg.get('n_radii', 120),
			)

		if viz_cfg.get('make_temperature_shell_video', False):
			cfg = viz_cfg.get('temperature_shell_video', {})
			self.make_shell_video(
				quantity = 'temp',
				r_min = cfg.get('r_min', 1.1),
				r_max = cfg.get('r_max', 3.0),
				n_radii = cfg.get('n_radii', 120),
			)

		if viz_cfg.get('make_microwave_orbit', False):
			cfg = viz_cfg.get('microwave_orbit', {})
			mask_cfg = viz_cfg.get('image_mask', {})

			self.make_microwave_orbit(
				lat = cfg.get('lat', 0.0),
				obs_r = cfg.get('obs_r', 215.0),
				n_frames = cfg.get('n_frames', 120),
				H = cfg.get('H', 512),
				W = cfg.get('W', 512),
				fov_rsun = cfg.get('fov_rsun', 3.0),
				rmax_vis_rsun = cfg.get('rmax_vis_rsun', None),
				chunk_rays = cfg.get('chunk_rays', 8192),
				apply_coronal_mask = mask_cfg.get('apply_coronal_mask', True),
				also_save_unmasked = mask_cfg.get('also_save_unmasked', False),
				r_min_mask_rsun = mask_cfg.get('r_min_mask_rsun', 1.0),
			)

		if viz_cfg.get('make_tilted_orbit', False):
			cfg = viz_cfg.get('tilted_orbit', {})
			mask_cfg = viz_cfg.get('image_mask', {})

			self.make_tilted_orbit(
				lon_c = cfg.get('lon_c', 0.0),
				lat_c = cfg.get('lat_c', 0.0),
				alpha_deg = cfg.get('alpha_deg', 10.0),
				obs_r = cfg.get('obs_r', 215.0),
				n_frames = cfg.get('n_frames', 120),
				H = cfg.get('H', 512),
				W = cfg.get('W', 512),
				fov_rsun = cfg.get('fov_rsun', 2.5),
				rmax_vis_rsun = cfg.get('rmax_vis_rsun', 2.5),
				chunk_rays = cfg.get('chunk_rays', 8192),
				apply_coronal_mask = mask_cfg.get('apply_coronal_mask', True),
				also_save_unmasked = mask_cfg.get('also_save_unmasked', False),
				r_min_mask_rsun = mask_cfg.get('r_min_mask_rsun', 1.0),
			)
