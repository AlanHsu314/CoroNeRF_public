from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import torch

from .uncertainty_quantification import _cfg, _outdir
from ..util.figure import _resolve_path, _savefig
from ..benchmark.config import save_json
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.select import _collect_condition_seed_rows
from ..benchmark.condition_compare import _resolve_run_dir, _build_state_for_analysis, _release_state
from ..data.camera import make_camera_rays_from_xyz
from ..visualization.geometry import llr_to_xyz_np, make_coronal_mask_simple

logger = logging.getLogger("coroNeRF.paper.convergence")

_RSUN_PER_AABB = 30.0   # one normalized AABB unit = 30 R_sun (matches the PSI cube normalization)


@torch.no_grad()
def _render_intensity(renderer, lon, lat, obs_r, H, W, fov_rsun):
    """One GT observation: (H, W, C) multichannel intensity, same path as vpgen."""
    obs_xyz = llr_to_xyz_np(lon, lat, r=obs_r)
    fov_rad = 2.0 * np.arctan2(fov_rsun, obs_r)
    ro_np, rd_np = make_camera_rays_from_xyz(obs_xyz=obs_xyz, H=H, W=W, fov=fov_rad, aabb_scale=renderer.aabb_scale)
    ro = torch.from_numpy(ro_np).to(device=renderer.device, dtype=renderer.dtype)
    rd = torch.from_numpy(rd_np).to(device=renderer.device, dtype=renderer.dtype)
    I = renderer(ro, rd)                                   # (H*W, C)
    return I.view(H, W, -1).float().cpu().numpy()


def generate_stepsize_convergence(spec_path, device_override=None, output_dir_override=None):
    """Render the GT forward model at several LOS integration step sizes and report image differences vs the
    finest step. Tests forward-renderer convergence / the no-inverse-crime concern. No dataset is written."""
    cfg, root = _cfg(spec_path, "convergence")
    out = _outdir(cfg, root, output_dir_override)
    device = device_override if device_override is not None else cfg.get("device")

    # pick any run just to build a GT-capable renderer (we render GT; the trained model is unused)
    bdir = _resolve_path(cfg["benchmark_dir"], root)
    sel = {k: cfg[k] for k in ("experiment", "where", "run_names", "filters", "min_seeds") if k in cfg}
    sel.setdefault("min_seeds", 1)
    seed_rows = _collect_condition_seed_rows(collect_run_summaries(bdir), sel)
    if not seed_rows:
        raise ValueError(f"convergence: no runs matched {sel} in {bdir}")
    run_dir = _resolve_run_dir(seed_rows[0], bdir)
    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=cfg.get("path_remap"))

    r = state.renderer
    saved = (r.use_gt_ne, r.reconstruction_target, r.step_size,
             getattr(r, "min_samples_per_ray", None), getattr(r, "enable_fallback", None))
    try:
        # --- GT-both forward: both ne and temp come from the GT fields (bypasses the ne_t model/GT guard) ---
        r.use_gt_ne = True
        r.reconstruction_target = "ne"
        if "min_samples_per_ray" in cfg:        r.min_samples_per_ray = int(cfg["min_samples_per_ray"])
        if "enable_min_samples_fallback" in cfg: r.enable_fallback = bool(cfg["enable_min_samples_fallback"])

        vc = dict(cfg.get("views", {}) or {})
        H = int(vc.get("H", 200)); W = int(vc.get("W", 200)); fov = float(vc.get("fov_rsun", 3.0))
        lat = float(vc.get("lat", 0.0)); obs_r = float(vc.get("obs_r", 215.0))
        n_views = int(vc.get("n_views", 8))
        lons = [float(x) for x in (vc.get("lons") or np.linspace(0.0, 2 * np.pi, n_views, endpoint=False))]

        steps = sorted((float(s) for s in cfg.get("step_sizes", [1/128, 1/256, 1/512, 1/1024])), reverse=True)
        s_c = np.asarray(cfg.get("asinh_scale_by_channel", [1.3e-2, 4.5e-4, 1.0e-2, 5.8e-2]), float)
        keep = np.asarray(make_coronal_mask_simple(H, W, fov_rsun=fov,
                                                   r_min_rsun=float(cfg.get("r_min_rsun", 1.0))), bool).reshape(-1)

        # render every (step x view); keep in memory
        imgs = {}
        for s in steps:
            r.step_size = float(s)
            imgs[s] = np.stack([np.nan_to_num(_render_intensity(r, L, lat, obs_r, H, W, fov)) for L in lons], 0)
            logger.info("rendered %d views at step=%.6g  (ds=%.4f R_sun)", len(lons), s, _RSUN_PER_AABB * s)

        ref = imgs[steps[-1]]                                       # finest = reference
        C = ref.shape[-1]
        recs = []
        for s in steps:
            X = imgs[s]; rel = np.zeros((len(lons), C)); asi = np.zeros((len(lons), C))
            for vi in range(len(lons)):
                for c in range(C):
                    a = X[vi, :, :, c].reshape(-1)[keep]; b = ref[vi, :, :, c].reshape(-1)[keep]
                    rel[vi, c] = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
                    asi[vi, c] = float(np.mean(np.abs(np.arcsinh(a / s_c[c]) - np.arcsinh(b / s_c[c]))))
            recs.append({"step": float(s), "ds_rsun": float(_RSUN_PER_AABB * s),
                         "rel_l2_by_channel": rel.mean(0).tolist(), "rel_l2_mean": float(rel.mean()),
                         "asinh_by_channel": asi.mean(0).tolist(), "asinh_mean": float(asi.mean())})
            logger.info("  step=%.6g: rel_L2=%.3e  asinhErr=%.3e (vs finest)", s, recs[-1]["rel_l2_mean"], recs[-1]["asinh_mean"])

        name = str(cfg.get("name", "stepsize_convergence"))
        npz = {"steps": np.array(steps), "ds_rsun": np.array([_RSUN_PER_AABB * s for s in steps]),
               "lons": np.array(lons), "asinh_scale_by_channel": s_c,
               "rel_l2": np.array([x["rel_l2_by_channel"] for x in recs]),
               "asinh":  np.array([x["asinh_by_channel"] for x in recs])}
        if bool(cfg.get("save_images", False)):
            npz.update({f"imgs_{i}": imgs[s] for i, s in enumerate(steps)})
        np.savez_compressed(out / f"{name}.npz", **npz)

        man = {"name": name, "run_dir": str(run_dir), "reference_step": float(steps[-1]),
               "rsun_per_aabb": _RSUN_PER_AABB, "n_channels": int(C),
               "views": {"n": len(lons), "lat": lat, "obs_r": obs_r, "H": H, "W": W, "fov_rsun": fov,
                         "lons": lons}, "asinh_scale_by_channel": s_c.tolist(), "records": recs}
        save_json(man, out / f"{name}_manifest.json")
        _fig_convergence(recs, name, out, cfg)
        logger.info("convergence -> %s", out / f"{name}_manifest.json")
        return man
    finally:
        (r.use_gt_ne, r.reconstruction_target, r.step_size) = saved[0], saved[1], saved[2]
        if saved[3] is not None: r.min_samples_per_ray = saved[3]
        if saved[4] is not None: r.enable_fallback = saved[4]
        _release_state(state)


def _fig_convergence(recs, name, out, cfg):
    import matplotlib.pyplot as plt
    ds = [x["ds_rsun"] for x in recs]; rl = [x["rel_l2_mean"] for x in recs]; ae = [x["asinh_mean"] for x in recs]
    fig, ax = plt.subplots(figsize=tuple(cfg.get("plot", {}).get("figsize", [5.0, 3.6])))
    ax.loglog(ds, rl, "o-", label="relative $L_2$ vs finest")
    ax.loglog(ds, ae, "s--", label="asinh image error vs finest")
    ax.set_xlabel(r"integration step $\Delta s$ [$R_\odot$]"); ax.set_ylabel("image difference vs finest step")
    ax.invert_xaxis(); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Forward-renderer step-size convergence")
    fig.tight_layout()
    return _savefig(fig, out / name, cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 200)))