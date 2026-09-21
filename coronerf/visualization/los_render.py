###################################
## generic LOS renderer wrapper
###################################

from __future__ import annotations
import numpy as np, torch
from ..data.camera import make_camera_rays_from_xyz
from .geometry import llr_to_xyz_np

@torch.no_grad()
def render_los_map(state, quantity: str, *, lon=0.0, lat=0.0, obs_r=215.0,
				   H=256, W=256, fov_rsun=3.0, chunk_rays=8192, use_gt=False,
				   r_min_rsun=1.0, r_max_rsun=None,
				   uniform_samples=0,
				   channels = None) -> np.ndarray:
	'''One observer viewpoint -> (H, W) LOS-integrated map of `quantity`.'''
	obs_xyz = llr_to_xyz_np(lon, lat, r=obs_r)
	fov_rad = 2.0 * np.arctan2(fov_rsun, obs_r)
	ro_np, rd_np = make_camera_rays_from_xyz(obs_xyz=obs_xyz, H=H, W=W, fov=fov_rad,
											 aabb_scale=state.renderer.aabb_scale)
	dev, dt = state.renderer.device, state.renderer.dtype
	ro = torch.from_numpy(ro_np).to(device=dev, dtype=dt)
	rd = torch.from_numpy(rd_np).to(device=dev, dtype=dt)
	vals = [state.renderer.los_integral(ro[i:i+chunk_rays], rd[i:i+chunk_rays], quantity=quantity, use_gt=use_gt, r_min_rsun=r_min_rsun, r_max_rsun=r_max_rsun, uniform_samples=uniform_samples, channels=channels)
			for i in range(0, H*W, chunk_rays)]
	return torch.cat(vals).view(H, W).float().cpu().numpy()
