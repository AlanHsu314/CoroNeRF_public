#############################################################
## rendering tools for debugging
#############################################################

import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union

import pickle # for stats dict

from ..artifacts.scalar_fields import get_scalar_field_spec, build_scalar_shell_product
from ..artifacts.plots import save_scalar_shell_triptych

from ..artifacts.intensity import render_rays_chunked
from ..artifacts.scalar_fields import scalar_field_available

def load_fixed_log_limits(dataset_dir: Union[Path, str], channel_indices = None):
    '''
    load stats dict from dataset_dir, use that as color bounds in log space
    '''
    stats_path = Path(dataset_dir) / 'stats_dict.pkl' # we'll change this json eventually
    with open(stats_path, 'rb') as file:
        stats = pickle.load(file)

    limits = []
    # loop through global indices
    for gidx in channel_indices:
        if gidx < 0 or gidx >= len(stats.keys()):
            limits.append((None, None))
            continue
        limits.append((float(np.log10(stats[gidx]['-3_sigma'])), float(np.log10(stats[gidx]['+3_sigma']))))

    return limits
    


def render_view_debug(pipeline, step: int, vp_idx: int = 0) -> None:
    """
    Render GT vs prediction for viewpoint vp_idx using:
    - log10 intensity scaling
    - positive-only clipping
    - consistent color scale
    - saved in figs/debug_views/
    """
    logger = logging.getLogger("coroNeRF.render_debug")

    if getattr(pipeline, "no_save", False):
        logger.debug("io.no_save = True -> skipping debug render save")
        return

    # Subdirectory
    debug_dir = pipeline.dirs.figs / "debug_views"
    debug_dir.mkdir(exist_ok=True)

    # Data
    imgs  = pipeline.train_data["imgs"]        # (V,H,W,C)
    mask  = pipeline.train_data["mask"]        # (H,W)
    rays_o_all = pipeline.train_data["rays_o"] # (V,3)
    rays_d_all = pipeline.train_data["rays_d"] # (V,H,W,3)

    V, H, W, C = imgs.shape
    assert vp_idx < V

    # channel line names
    line_names = list(
        getattr(pipeline, 'sselected_line_names', None)
        or pipeline.cfg.get('ion_kwargs', {}).get('line_names', [])
        or []
    )
    if len(line_names) < C:
        line_names = line_names + [f'ch {i}' for i in range(len(line_names), C)]
    else:
        line_names = line_names[:C]


    # collect rays
    gt = imgs[vp_idx]                # (H,W,C)
    rays_o_v = rays_o_all[vp_idx]    # (3,)
    rays_d_v = rays_d_all[vp_idx]    # (H,W,3)

    # Flatten rays
    rays_d_flat = rays_d_v.reshape(-1, 3)
    rays_o_flat = rays_o_v.reshape(1, 3).expand(H * W, 3)

    device = pipeline.renderer.device
    dtype  = pipeline.renderer.dtype

    # match device and type
    rays_o=rays_o_flat.to(device=device, dtype=dtype)
    rays_d=rays_d_flat.to(device=device, dtype=dtype)
    
    # render call
    N = H * W
    chunk_rays = pipeline.cfg.train['batch_rays']
    pred_flat = render_rays_chunked(
        pipeline,
        rays_o=rays_o,
        rays_d=rays_d,
        chunk_rays=chunk_rays,
    )
    pred = pred_flat.view(H, W, -1)

    # Prepare masked copies
    mask_cpu = mask.cpu()
    gt_plot   = gt.clone()
    pred_plot = pred.clone()
    gt_plot[~mask_cpu]   = float("nan")
    pred_plot[~mask_cpu] = float("nan")

    # ---- LOG SCALE ----
    # clamp to positive before log10
    eps = 1e-12
    log_gt   = torch.log10(gt_plot.clamp(min=eps))
    log_pred = torch.log10(pred_plot.clamp(min=eps))

    # old vmin/vmax
    # color scale from **positive-only log values**
    # finite_vals = torch.cat([
    #     log_gt[torch.isfinite(log_gt)].flatten(),
    #     log_pred[torch.isfinite(log_pred)].flatten()
    # ])

    # if finite_vals.numel() == 0:
    #     vmin, vmax = -6, -1   # fallback
    # else:
    #     vmin = finite_vals.min().item()
    #     vmax = finite_vals.max().item()

    # new vmin/vmax
    active_channels = pipeline.cfg.get('load_data_kwargs', {}).get('channel_indices')
    fixed_limits = load_fixed_log_limits(pipeline.cfg['dataset_dir'], active_channels)

    # ---- PLOTTING ----
    fig, axes = plt.subplots(2, C, figsize=(4 * C, 8), constrained_layout=True)
    if C == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for c in range(C):
        # get color bounds
        if fixed_limits is not None:
            vmin, vmax = fixed_limits[c]
        else:
            vmin,vmax = None

        im0 = axes[0, c].imshow(log_gt[:, :, c].numpy(),
                                origin="lower", vmin=vmin, vmax=vmax,
                                cmap="inferno")
        axes[0, c].set_title(f"GT log10 {line_names[c]}")
        fig.colorbar(im0, ax=axes[0, c])

        im1 = axes[1, c].imshow(log_pred[:, :, c].numpy(),
                                origin="lower", vmin=vmin, vmax=vmax,
                                cmap="inferno")
        axes[1, c].set_title(f"Pred log10 {line_names[c]}")
        fig.colorbar(im1, ax=axes[1, c])

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    out_path = debug_dir / f"vp{vp_idx:02d}_step{step:06d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Record view-level scalar error (before log)
    rel = (pred + eps) / (gt + eps)
    view_loss = torch.log(torch.cosh(torch.log(rel).abs())).mean().item()

    logger.info(f"[render_debug] step {step} vp {vp_idx} "
                f"view_loss={view_loss:.4e} saved={out_path}")
    
def render_scalar_shell_debug(pipeline, quantity: str, step: int, r_target: float = 1.5):
    logger = logging.getLogger(f"coroNeRF.{quantity}_shell_debug")

    if getattr(pipeline, "no_save", False):
        logger.debug("io.no_save = True -> skipping scalar shell debug save")
        return

    if not scalar_field_available(pipeline, quantity):
        logger.warning(f"{quantity} field not available; skipping scalar shell debug.")
        return

    product = build_scalar_shell_product(pipeline, quantity=quantity, r_target=r_target)
    spec = get_scalar_field_spec(quantity)

    shell_dir = pipeline.dirs.figs / spec.shell_dirname
    shell_dir.mkdir(exist_ok=True, parents=True)

    out_path = shell_dir / f"{spec.shell_dirname[:-1]}_step{step:08d}_r{product.r:.2f}.png"
    save_scalar_shell_triptych(product, out_path)

    logger.info(
        f"[{quantity}_shell_debug] step {step}: r_target={float(r_target):.3f}, "
        f"using r={product.r:.3f}, saved={out_path}"
    )
