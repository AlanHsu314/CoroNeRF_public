##################################################################
### code to generate and preprocess viewpoints
##################################################################

import logging
import numpy as np
import torch

from .camera import make_camera_rays_from_xyz, make_coronal_mask_simple
from ..geom.coords import llr_to_xyz
from ..experiments.dirs import save_pickle

# chatGPT function
'''
if you use the old psi viewpoint generation code (pycelp-style LOS integration), 
use this to match vp with NeRF forward model
'''
def regenerate_psi_viewpoints(pipeline):
    """
    Recompute intensities for all vp_*.npz using the current renderer
    with use_gt_ne=True, and save to a new 'viewpoints_gt' directory.

    Requires:
        - self.renderer already constructed
        - self.ne_field set and used by renderer (use_gt_ne=True)
    """
    
    # only things needed from pipeline
    dataset_dir = pipeline.dataset_dir
    renderer = pipeline.renderer
    

    logger = logging.getLogger("coroNeRF.regen_dataset")

    vp_dir = dataset_dir / "viewpoints"
    out_dir = dataset_dir / "viewpoints_gt"
    out_dir.mkdir(exist_ok=True)

    # find all vp_*.npz
    vp_files = sorted(vp_dir.glob("vp_*.npz"))
    logger.info(f"Regenerating {len(vp_files)} viewpoints into {out_dir}")

    for vp_path in vp_files:
        vp_idx_str = vp_path.stem.split("_")[-1]  # 'vp_12' -> '12'
        logger.info(f"Rendering viewpoint {vp_idx_str} from {vp_path.name}")

        vp_data = np.load(vp_path, allow_pickle=False)
        rays_o_np = vp_data["rays_o"]          # (3,)
        rays_d_np = vp_data["rays_d"]          # (H,W,3)

        H, W, _ = rays_d_np.shape

        # flatten rays
        rays_d = torch.from_numpy(rays_d_np).to(
            device=renderer.device,
            dtype=renderer.dtype,
        ).view(-1, 3)                           # (H*W,3)
        rays_o = torch.from_numpy(rays_o_np).to(
            device=renderer.device,
            dtype=renderer.dtype,
        ).view(1, 3).expand(H * W, 3)          # (H*W,3)

        # render with ground-truth ne
        use_gt_old = getattr(renderer, "use_gt_ne", False)
        renderer.use_gt_ne = True

        with torch.no_grad():
            intens_flat = renderer(rays_o, rays_d)   # (H*W,C)

        # restore flag just in case
        renderer.use_gt_ne = use_gt_old

        intens = intens_flat.view(H, W, -1).cpu().numpy()

        # write new file: same rays, new intensities
        out_path = out_dir / f"vp_{vp_idx_str}.npz"
        np.savez(
            out_path,
            intensities=intens,
            rays_o=rays_o_np,
            rays_d=rays_d_np,
        )

        logger.info(f"Saved regenerated view to {out_path}")

def generate_psi_viewpoints(pipeline):
    '''
    generates synthetic viewpoints based on requested emission lines
    '''

    # no save
    no_save = bool(getattr(pipeline, 'no_save', False))

    # only things needed from pipeline
    dirs = pipeline.dirs
    cfg = pipeline.cfg
    renderer = pipeline.renderer

    logger = logging.getLogger('coroNeRF.generate_psi_viewpoints')

    logger.info('generating new viewpoints...')

    # create viewpoints dir
    if no_save:
        out_dir = None
        logger.info('[vpgen] no_save: not creating viewpoints_gt/ or writing mask/vp files.')
    else:
        out_dir = dirs.datasets / 'viewpoints_gt'
        out_dir.mkdir(parents = True, exist_ok = True)

    # set parameters
    H = int(cfg.vpgen.get('spatial_H', 200))
    W = int(cfg.vpgen.get('spatial_W', 200))
    fov_rsun = float(cfg.vpgen.get('fov_rsun', 3))
    n_lon = int(cfg.vpgen.get('n_lon', 100))
    obs_r = float(cfg.vpgen.get('obs_r', 215))
    
    vp_dtype_str = cfg.vpgen.get('dtype', 'double')
    vp_dtype_np = np.float64 if vp_dtype_str in ['double', 'float64'] else np.float32
    vp_dtype_torch = torch.float64 if vp_dtype_str in ['double', 'float64'] else torch.float32

    assert (H > 0) and (W > 0)
    assert fov_rsun > 0
    assert obs_r > 0
    assert n_lon > 0

    logger.info(f'vpgen config, (H, W) = {H, W}, fov = {fov_rsun} rsun, n_lon = {n_lon}')

    # create list of angles
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint = False, dtype = vp_dtype_np)
    lats = np.array([0.0]) # currently latitude always at 0

    # set renderer to correct settings
    use_gt_ne_old = getattr(renderer, "use_gt_ne", False)
    renderer.use_gt_ne = True

    # compute fov in radians: FOV / 2 = arctan(R_max / d)
    fov_rad = 2*np.arctan2(fov_rsun,  obs_r)

    # create (and save) corona mask
    mask = make_coronal_mask_simple(H, W, fov_rad, obs_r)
    logger.debug(f'mask min: {mask.min()}, mask max: {mask.max()}')
    if not no_save:
        np.save(dirs.datasets / 'mask.npy', mask)

    if renderer.cfg["renderer_model"] == "nerfacc":
        logger.info("[vpgen] warming up nerfacc occupancy grid...")

        occ_mask = renderer.nerfacc_estimator.binaries
        logger.info(f"[vpgen] occ binaries sum before={int(occ_mask.sum().item())} / {occ_mask.numel()}")

        for s in range(10):
            renderer.update_occupancy(step=s)

        occ_mask = renderer.nerfacc_estimator.binaries
        logger.info(f"[vpgen] occ binaries sum after={int(occ_mask.sum().item())} / {occ_mask.numel()}")

    # viewpoint gen loop
    vp_idx = 0

    for lat in lats:
        for lon in lons:
            
            # get observer position (r_sun)
            obs_llr = torch.tensor([lon, lat, obs_r], dtype = vp_dtype_torch)
            obs_xyz = llr_to_xyz(obs_llr[None, :])[0].cpu().numpy()

            # get rays
            rays_o_np, rays_d_np = make_camera_rays_from_xyz(
                obs_xyz = obs_xyz,
                H = H,
                W = W,
                fov = fov_rad,
                aabb_scale = renderer.aabb_scale
            )

            # typing
            rays_o = torch.from_numpy(rays_o_np).to(device = renderer.device, dtype = renderer.dtype)
            rays_d = torch.from_numpy(rays_d_np).to(device = renderer.device, dtype = renderer.dtype)

            # --- optional: dump nerfacc sample locations for debugging

            dump_samples = bool(cfg.vpgen.get("dump_samples", False))
            dump_only_first = bool(cfg.vpgen.get("dump_samples_only_first", True))  # only first vp
            dump_bins = int(cfg.vpgen.get("dump_samples_bins", 200))

            if dump_samples and (not dump_only_first or vp_idx == 0):
                # Sample only (no rendering); works for uniform or nerfacc
                with torch.no_grad():
                    t_s, t_e, ray_idx, valid = renderer._sample_points_along_rays(rays_o, rays_d)

                M = t_s.numel()
                Nv = int(valid.sum().item())
                logger.debug(f'[vpgen] dump_samples: valid rays {Nv}/{valid.numel()}, packed samples M = {M}')

                counts = torch.bincount(ray_idx, minlength=Nv)  # ray_idx from packed samples
                print("fraction rays with >0 samples:", (counts>0).float().mean().item())
                print("min samples per ray:", counts.min().item())
                print("median samples per hit-ray:", counts[counts>0].median().item())
                print("max samples per ray:", counts.max().item())

                if M > 0:
                    # get midpt positions (in AABB coords)
                    t_mid = 0.5 * (t_s + t_e)   # (M,)
                    o = rays_o[valid][ray_idx]  # (M, 3)
                    d = rays_d[valid][ray_idx]  # (M, 3)
                    x = o + d*t_mid[:, None]    # (M, 3)

                    # AABB -> radius
                    r_rsun = torch.linalg.norm(x, dim = -1) * renderer.aabb_scale # (M,)

                    # subsample for stats 
                    stats_max = int(cfg.vpgen.get("dump_samples_stats_max", 2_000_000)) # for quantile saving
                    save_max = int(cfg.vpgen.get("dump_samples_max", 200_000))          # for saving points

                    # summary stats
                    if M > stats_max:
                        stats_idx = torch.randperm(M, device = r_rsun.device)[:stats_max]
                    else:
                        stats_idx = torch.arange(M, device = r_rsun.device)
                    
                    r_stats = r_rsun[stats_idx]
                    q = torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], device = r_stats.device, dtype = r_stats.dtype)
                    r_quant = torch.quantile(r_stats, q).detach().cpu().numpy()

                    # histogram 
                    r_cpu = r_stats.detach().cpu().numpy()
                    hist, edges = np.histogram(r_cpu, bins = dump_bins)

                    #----------- start test ---------------------------
                    '''
                    basically, i was concerned that nerfacc wasnt really changing the sampling distribution
                    this test shows otherwise, uncomment to see the debug results
                    '''

                    # # after warmup, on GPU
                    # K = 200000
                    # x_test = (torch.rand(K, 3, device=renderer.device) * 2 - 1)  # uniform in [-1,1]^3 AABB

                    # # 1) your intended occupancy score (float in [0,1])
                    # with torch.no_grad():
                    #     occ_float = renderer.occ_eval_fn(x_test)  # sigmoid(log_ne-mid/width)
                    #     # also compute log_ne used by occ_eval_fn (make sure you can return it)
                    #     x_test_xyz = x_test * renderer.aabb_scale
                    #     log_ne = renderer.ne_field(x_test_xyz)

                    # print("log_ne stats:", log_ne.min().item(), log_ne.median().item(), log_ne.max().item())
                    # print("occ_float stats:", occ_float.min().item(), occ_float.median().item(), occ_float.max().item())
                    # print("frac occ_float>0.5:", (occ_float>0.5).float().mean().item())
                    # print("frac occ_float>0.1:", (occ_float>0.1).float().mean().item())

                    # # 2) what nerfacc thinks is occupied: sample points along rays and check their occ_float
                    # # simplest: reuse your sampled points from nerfacc
                    # # if you have x_AABB_samples from nerfacc:
                    # with torch.no_grad():
                    #     log_ne_s = renderer.ne_field((x * renderer.aabb_scale))
                    #     occ_s = torch.sigmoid((log_ne_s - 8.25)/0.25)
                    # print("nerfacc samples: log_ne min/med/max:",
                    #     log_ne_s.min().item(), log_ne_s.median().item(), log_ne_s.max().item())
                    # print("nerfacc samples: occ_s min/med/max:",
                    #     occ_s.min().item(), occ_s.median().item(), occ_s.max().item())
                    # print("nerfacc samples: frac occ_s>0.1:", (occ_s>0.1).float().mean().item())

                    #--------------- end test -----------------------------

                    # random subset so file does not explode
                    if M > save_max:
                        save_idx = torch.randperm(M, device = r_rsun.device)[:save_max]
                    else:
                        save_idx = torch.arange(M, device = r_rsun.device)
                    
                    # save this in debug file with the vp
                    if not no_save:
                        debug_path = out_dir / f'vp_{vp_idx}_samples_debug.npz'
                        np.savez(debug_path,
                                    # summary
                                    valid_rays = Nv,
                                    total_rays = int(valid.numel()),
                                    total_samples = int(M),
                                    aabb_scale = float(renderer.aabb_scale),
                                    quantiles = q.detach().cpu().numpy(),
                                    r_quantiles = r_quant,
                                    r_hist = hist,
                                    r_hist_edges = edges,
                                    # subset (packed)
                                    ray_idx = ray_idx[save_idx].detach().cpu().numpy().astype(np.int32),
                                    t_s = t_s[save_idx].detach().cpu().numpy().astype(np.float32),
                                    t_e = t_e[save_idx].detach().cpu().numpy().astype(np.float32),
                                    x = x[save_idx].detach().cpu().numpy().astype(np.float32),
                                    r_rsun = r_rsun[save_idx].detach().cpu().numpy().astype(np.float32),
                                    )
                    
                        logger.debug(f'[vpgen] saved sample debug to {debug_path}')
                else:
                    logger.debug(f'[vpgen] no samples to dump (M=0)')

            # -- end dump

            # render
            with torch.no_grad():
                eps_I_flat = renderer(rays_o, rays_d)

            # reshape
            eps_I = eps_I_flat.view(H, W, -1).cpu().numpy()

            # mask
            eps_I = eps_I * mask[..., None]

            # save
            if not no_save:
                np.savez(out_dir / f'vp_{vp_idx}.npz',
                            intensities = eps_I,
                            rays_o = rays_o_np[0],
                            rays_d= rays_d_np.reshape(H, W, 3))


            vp_idx += 1


    # set renderer back to old state
    renderer.use_gt_ne = use_gt_ne_old

    logger.info(f'Generated {vp_idx} viewpoints into directory {out_dir}')

def preprocess_viewpoints(pipeline):
    '''
    After generating viewpoints, preprocess them 
    [1] computes states
    [2] normalizes images
    [3] Note rays_o already normalized to AABB in the config
    '''
    # pipeline attributes needed
    dirs = pipeline.dirs
    cfg = pipeline.cfg

    no_save = bool(getattr(pipeline, 'no_save', False))

    logger = logging.getLogger('coroNeRF.preprocess_viewpoints')

    if no_save:
        logger.info('[vpgen] no_save: skipping preprocess_viewpoints (it expects on-disk vp files).')
        return

    logger.info('computing viewpoint statistics...')

    # load corona mask
    mask = np.load(f'{dirs.datasets}/mask.npy') # load mask
    rows_valid, cols_valid = np.where(mask) # get indices where it is not photosphere

    # load dimensions
    first_vp = np.load(f"{dirs.datasets / 'viewpoints_gt'}/vp_0.npz", allow_pickle = False)
    num_channels = int(first_vp['intensities'].shape[-1])

    num_rows = int(cfg.vpgen.get('spatial_H', 200))
    num_cols = int(cfg.vpgen.get('spatial_W', 200))
    num_vp = int(cfg.vpgen.get('n_lon', 100))

    line_names = list(cfg.ion_kwargs.get('line_names', []))
    wavelengths = list(cfg.ion_kwargs.get('wavelengths', []))

    # create channel dict mapping channels to dictionaries
    channel_dict = dict()
    for i in range(num_channels):
        # if line_name exists, use that
        if i < len(line_names):
            name = line_names[i]
        # otherwise use wvl as name
        elif i < len(wavelengths):
            name = f'ch{i} ({wavelengths[i]})'
        # otherwise just use index
        else:
            name = f'ch{i}'
        channel_dict[i] = {'name': name}
    
    # load data and compute state
    for i_ch in range(num_channels):

        # reset cube for the channel
        data_cube = np.zeros((num_vp, num_rows, num_cols))
        cd = channel_dict[i_ch]

        # loop through vp and populate cube, honestly, not the most efficient, but whatever
        for i_vp in range(num_vp):
            data_cube[i_vp,:,:] = np.load(f'{dirs.datasets / "viewpoints_gt"}/vp_{i_vp}.npz')['intensities'][:,:,i_ch]

        # compute stats
        pixels = data_cube[:, rows_valid, cols_valid]
        cd['min']      = pixels.min()
        cd['-3_sigma'] = np.percentile(pixels, 0.1)
        cd['-2_sigma'] = np.percentile(pixels, 2.3)
        cd['-1_sigma'] = np.percentile(pixels, 15.9)
        cd['0_sigma']  = np.percentile(pixels, 50)
        cd['+1_sigma'] = np.percentile(pixels, 84.1)
        cd['+2_sigma'] = np.percentile(pixels, 97.7)
        cd['+3_sigma'] = np.percentile(pixels, 99.9)
        cd['max']      = pixels.max()

        # print them
        logger.debug(f'-------------- {cd["name"]}-----------------')
        logger.debug(f'min: {cd["min"]}')
        logger.debug(f'-3_sigma: {cd["-3_sigma"]}')
        logger.debug(f'-2_sigma: {cd["-2_sigma"]}')
        logger.debug(f'-1_sigma: {cd["-1_sigma"]}')
        logger.debug(f'0_sigma: {cd["0_sigma"]}')
        logger.debug(f'+1_sigma: {cd["+1_sigma"]}')
        logger.debug(f'+2_sigma: {cd["+2_sigma"]}')
        logger.debug(f'+3_sigma: {cd["+3_sigma"]}')
        logger.debug(f'max: {cd["max"]}')

    # save the dictionary
    save_pickle(channel_dict, f'{dirs.datasets}/stats_dict.pkl')

    logger.info(f'saved stats file to {dirs.datasets}/stats_dict.pkl')

    # determine global bounds
    img_lo = np.array([channel_dict[key]['-3_sigma'] for key in channel_dict]).min()
    img_hi = np.array([channel_dict[key]['+3_sigma'] for key in channel_dict]).max()
    logger.debug(f'global lo: {img_lo}, global hi: {img_hi}')
