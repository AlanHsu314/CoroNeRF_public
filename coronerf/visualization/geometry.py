####################################
## generic helpers for visualization
####################################

import numpy as np


# helpers
def xyz_to_llr(x, y, z):
	r = np.sqrt(x**2 + y**2 + z**2)
	lon = np.arctan2(y, x)
	lat = np.arcsin(z / r)
	return lon, lat, r

def llr_to_xyz_np(lon, lat, r=1.0):
	x = r * np.cos(lat) * np.cos(lon)
	y = r * np.cos(lat) * np.sin(lon)
	z = r * np.sin(lat)
	return np.array([x, y, z], dtype=np.float64)

# courtesy of chatgpt
def apply_rmax_mask(frame, rmax_vis_rsun, fov_rsun):
	"""
	frame: (H, W) intensity image
	rmax_vis_rsun: radius in R_sun you want to keep (e.g. 5.0)
	fov_rsun: max radius your FOV covers (e.g. 3.0 for training; whatever you used)

	Returns a masked copy of frame, with pixels beyond rmax_vis set to min(frame).
	"""
	H, W = frame.shape
	cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

	ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
	r_pix = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

	r_pix_max = np.min([cx, cy])
	r_pix_cut = (rmax_vis_rsun / fov_rsun) * r_pix_max

	mask_keep = r_pix <= r_pix_cut

	masked = frame.copy()
	floor_val = np.nanmin(frame)
	masked[~mask_keep] = floor_val
	return masked

def make_coronal_mask_simple(H: int, W: int, fov_rsun: float, r_min_rsun: float = 1.0, r_max_rsun: float | None = None):
	'''
	Simple 2D annular mask in image plane.

	Assumes image spans roughly [-fov_rsun, +fov_rsun] in both axes.
	Keeps projected radii between r_min_rsun and r_max_rsun.
	'''
	cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
	ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

	x = (xs - cx) / min(cx, cy) * fov_rsun
	y = (ys - cy) / min(cx, cy) * fov_rsun
	r_proj = np.sqrt(x**2 + y**2)

	mask = r_proj >= float(r_min_rsun)
	if r_max_rsun is not None:
		mask = mask & (r_proj <= float(r_max_rsun))

	return mask