#######################################################
## camera geometry
#######################################################

import logging
import numpy as np

# generate rays from camera pos
def make_camera_rays_from_xyz(obs_xyz: np.ndarray,
							  H: int,
							  W: int,
							  fov: float,
							  aabb_scale: float,
							  up_world: np.ndarray = np.array([0.0, 0.0, 1.0])):
	'''
	Given camera position, pointing at sun
	[1] generate rays that construct pixels at image plane
	[2] normalize to AABB [-1,1]^3

	assumptions:
	image plane @ z = 1 in camera space

	inputs:
	obs_xyz: (3,) observer's position in world space
	H: # vertical pixels in image plane
	W: # horizontal pixels in image plane
	fov: VERTICAL field-of-view, in radians
	aabb_scale: scale for AAABB
	up_world: canonical "up" in world space
	'''

	# define camera space basis
	o = obs_xyz.astype(np.float64)
	r = np.linalg.norm(o)

	# z-axis (camera "forward")
	z_cam = -o / r
	up = up_world.astype(np.float64)
	if np.allclose(np.cross(up, z_cam), 0):
		up = np.array([0.0, 1.0, 0.0]) # set another canonical up if close to z

	# construct x and y camera axes
	x_cam = np.cross(up, z_cam)
	x_cam /= np.linalg.norm(x_cam)
	y_cam = np.cross(z_cam, x_cam)

	# create normalized device coordinates (NDC) [-1,1]^2 @ image plane z = + 1
	ys, xs = np.meshgrid(np.linspace(-1.0, 1.0, H),
					  	 np.linspace(-1.0, 1.0, W),
						 indexing = 'ij') # y is ROWS, x is COLS

	# map NDC to camera space via vertical fov (don't forget aspect ratio)
	aspect = W / H
	xs = xs * np.tan(fov / 2.0) * aspect
	ys = ys * np.tan(fov / 2.0)

	# finally map camera space to world space: d = x*x_cam + y*y_cam + 1*z_cam
	rays_d = (xs[..., None] * x_cam[None, None, :] + 
		      ys[..., None] * y_cam[None, None, :] + 
			  1             * z_cam[None, None, :])
	rays_d_norms = np.linalg.norm(rays_d, axis = -1, keepdims = True)
	rays_d /= rays_d_norms
	rays_d = rays_d.reshape(-1, 3) # flatten

	# create rays_o (same origin for all flattened pixels)
	rays_o = (o / aabb_scale)[None, :].repeat(H * W, axis = 0)

	return rays_o.astype(np.float32), rays_d.astype(np.float32)

# simple masking (on image plane)
def make_coronal_mask_simple(H, W, fov, obs_r, r_sun = 1.0):
	'''
	Simple geometric masking of photosphere on image plane

	returns: mask where corona = true
	'''
	logger = logging.getLogger('coroNeRF.make_coronal_mask_simple')
	# pixel NDC centers on image plane
	ys, xs = np.meshgrid(np.linspace(-1.0, 1.0, H),
					  	 np.linspace(-1.0, 1.0, W),
						 indexing = 'ij') # y is ROWS, x is COLS
	
	# map NDC to camera space via vertical fov (don't forget aspect ratio)
	aspect = W / H
	x = xs * np.tan(fov / 2.0) * aspect
	y = ys * np.tan(fov / 2.0)
	r = np.sqrt(x**2 + y**2)

	logger.debug(f'r max: {r.max()}, min: {r.min()}')

	# project radius onto image plane (tan(arcsin(R_sun / d_obs)))
	# large d, so we approximate via series
	r_disk = r_sun / obs_r

	logger.debug(f'r disk: {r_disk}')

	mask = r > r_disk

	return mask







