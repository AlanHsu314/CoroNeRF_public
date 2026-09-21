##########################################################
### identifiability.py — linearized (Jacobian/Fisher) identifiability analysis
###
### THE MATH (what the code computes):
###   Observation model:   d = F(m) + η,   η ~ N(0, Σ),  Σ = diag(σ_i²)
###     m : log-field VALUES on a coarse 3-D control grid  (log10 n_e [, log10 T])
###         -> we analyze the *physical field*, not the network weights, because
###            identifiability is a property of the observation setup, not the prior.
###     F : control grid --(spherical trilinear)--> renderer LOS integral --> intensities
###     Σ : per-observation noise variance from the run's heteroscedastic noise model.
###
###   Linearize F about the GROUND-TRUTH field m0:
###       F(m0 + δm) ≈ F(m0) + J δm ,   J = ∂F/∂m  ∈ R^{N×M}
###   Noise-whiten (put rows in units of "sigmas"):   J̃ = Σ^{-1/2} J
###   Gauss–Newton Hessian == Fisher information (identical for Gaussian Σ):
###       H = Jᵀ Σ⁻¹ J = J̃ᵀ J̃  ∈ R^{M×M}
###   SVD:  J̃ = U S Vᵀ ;  s_k = singular values ;  v_k = field-space modes
###       eig(H) = s_k² ,  eigenvectors = v_k
###   READOUTS:
###     identifiable  : s_k ≳ 1  (mode imprints on data above the noise)
###     null space    : s_k → 0  (data can't see it; prior decides it)
###     effective rank: #{s_k > τ}, plus threshold-free participation-ratio & spectral-entropy ranks
###     condition #   : s_max / s_min  (and s_max / s_τ)
###     information map (diag of Fisher):  diag(H)_j = Σ_i J̃_{ij}²  -> per-control-point "constrainedness"
###     null-space 3-D map: smallest-s_k right-singular vectors v_k reshaped onto the grid
###     resolution matrix:  R = (H + λI)⁻¹ H ; diag(R)≈1 well-resolved, off-diag = smearing
###
###   VERIFICATION (not closure):
###     finite_difference_check() compares autograd J to (F(m0+εe_j) − F(m0))/ε.
##########################################################

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Union
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..fields.interp_util import _axis_to_m1p1_nonuniform
from ..geom.coords import llr_to_xyz
from ..benchmark.condition_compare import _build_state_for_analysis, _release_state, _resolve_run_dir
from ..util.figure import _resolve_path, _savefig, _cmap_with_bad, _qty_tex
from ..benchmark.select import _collect_condition_seed_rows

logger = logging.getLogger("coroNeRF.identifiability._core")


def _maybe_suptitle(fig, text):
    if text:
        fig.suptitle(str(text))


# ----------------------------------------------------------------------
# Control-grid field: a DIFFERENTIABLE coarse spherical grid whose node
# values are the parameters m. Mirrors SphericalTrilinearField's coordinate
# handling exactly (periodic longitude, latitude, radius), but the grid is an
# nn.Parameter and it carries Q channels (Q=1 for 'ne', Q=2 for 'ne_t').
# ----------------------------------------------------------------------

class ControlGridField(nn.Module):
    def __init__(self, lon_axis, phi_axis, r_axis, values, aabb_scale, target):
        """
        lon_axis (L,), phi_axis (P,), r_axis (R,)   -- 1-D coordinate axes (radians, radians, R_sun)
        values   (Q, L, P, R)                        -- log-field node values (the parameters m)
        """
        super().__init__()
        self.target = str(target)
        self.aabb_scale = float(aabb_scale)
        self.values = nn.Parameter(values.clone().float())          # (Q, L, P, R)  <-- this is m

        self.register_buffer("phi_axis", phi_axis.float())
        self.register_buffer("r_axis", r_axis.float())

        # periodic-longitude padding (identical trick to SphericalTrilinearField)
        dlon = (lon_axis[1] - lon_axis[0]) if lon_axis.numel() > 1 else torch.tensor(2 * math.pi)
        self.register_buffer("lon_min", lon_axis[0].float())
        self.register_buffer("lon_max_padded", (lon_axis[-1] + dlon).float())
        self.register_buffer("lon_axis_pad", torch.cat([lon_axis, lon_axis[-1:] + dlon]).float())

    def _wrap_lon(self, lon):
        span = self.lon_max_padded - self.lon_min
        return (lon - self.lon_min) % span + self.lon_min

    def _field_5d(self):
        # pad the longitude axis so grid_sample interpolates across the 0/2π seam,
        # then permute to grid_sample's (N, C, D, H, W) = (1, Q, R, P, L+1).
        vp = torch.cat([self.values, self.values[:, :1]], dim=1)     # (Q, L+1, P, R)
        return vp.permute(0, 3, 2, 1)[None]                          # (1, Q, R, P, L+1)

    def forward(self, x_AABB):
        x_xyz = x_AABB.to(self.values.dtype) * self.aabb_scale
        xx, yy, zz = x_xyz.unbind(-1)
        r = torch.linalg.vector_norm(x_xyz, dim=-1).clamp_min(1e-12)
        lon = torch.atan2(yy, xx)
        lon = (lon + 2 * math.pi) % (2 * math.pi)
        rho = torch.sqrt(xx * xx + yy * yy)
        lon = torch.where(rho < 1e-6 * r, torch.zeros_like(lon), lon)   # canonical lon at the poles
        lon = self._wrap_lon(lon)
        phi = torch.asin(torch.clamp(zz / r, -1.0, 1.0))

        gx = _axis_to_m1p1_nonuniform(lon, self.lon_axis_pad)   # -> W
        gy = _axis_to_m1p1_nonuniform(phi, self.phi_axis)       # -> H
        gz = _axis_to_m1p1_nonuniform(r, self.r_axis)           # -> D
        grid = torch.stack([gx, gy, gz], dim=-1).reshape(1, 1, 1, -1, 3)

        vals = F.grid_sample(self._field_5d(), grid, mode="bilinear",
                             padding_mode="border", align_corners=True)   # (1, Q, 1, 1, Npts)
        Q = self.values.shape[0]
        out = vals.reshape(Q, -1).transpose(0, 1)               # (Npts, Q)
        return out.reshape(*x_AABB.shape[:-1], Q)


def _quantities(target: str) -> list[str]:
    return ["ne", "temp"] if target == "ne_t" else ["ne"]


def build_control_field(state, grid_cfg: dict, target: str, device, dtype=torch.float32, source: str = "gt") -> ControlGridField:
    """Build a coarse control grid and initialise its node values from the GT PSI fields."""
    n_lon = int(grid_cfg.get("n_lon", 18))
    n_lat = int(grid_cfg.get("n_lat", 10))
    n_r = int(grid_cfg.get("n_r", 8))
    r_min = float(grid_cfg.get("r_min", 1.05))
    r_max = float(grid_cfg.get("r_max", 3.0))
    lat_frac = float(grid_cfg.get("lat_frac", 0.98))   # stay just inside the poles

    lon_axis = torch.linspace(0.0, 2 * math.pi, n_lon + 1)[:-1]          # periodic -> drop endpoint
    phi_axis = torch.linspace(-math.pi / 2 * lat_frac, math.pi / 2 * lat_frac, n_lat)
    r_axis = torch.linspace(r_min, r_max, n_r)

    # nodes -> xyz, then query the GT fields (these return log10 of the quantity)
    P, Lg, Rg = phi_axis.numel(), lon_axis.numel(), r_axis.numel()
    lo, ph, rr = torch.meshgrid(lon_axis, phi_axis, r_axis, indexing="ij")  # (L,P,R)
    llr = torch.stack([lo, ph, rr], dim=-1).reshape(-1, 3)
    xyz = llr_to_xyz(llr).to(device=device, dtype=dtype)

    qlist = _quantities(target); vols = []
    with torch.no_grad():
        for q in qlist:
            if source == "gt":
                gt = state.ne_field if q == "ne" else state.temp_field      # analytic PSI (m*)
                v = gt(xyz).reshape(Lg, P, Rg)
            elif source == "recon":                                          # trained model (m̂)
                from ..artifacts.scalar_fields import _predict_scalar_log_field
                x_aabb = (xyz / state.renderer.aabb_scale).to(device=state.renderer.device, dtype=state.renderer.dtype)
                v = _predict_scalar_log_field(state, q, x_aabb).reshape(Lg, P, Rg)
            else:
                raise ValueError(f"unknown control source: {source}")
            vols.append(v.detach().cpu())
    values = torch.stack(vols, dim=0).float()                               # (Q,L,P,R)
    cf = ControlGridField(lon_axis, phi_axis, r_axis, values,
                          aabb_scale=float(state.renderer.aabb_scale), target=target)
    return cf.to(device)


# ----------------------------------------------------------------------
# ray subset + whitening
# ----------------------------------------------------------------------

def gather_ray_subset(state, rays_cfg: dict, split: str = "train", seed: int = 0):
    data = state.train_data if split == "train" else state.test_data
    rays_o_all = data["rays_o"]                # (V, 3)
    rays_d_all = data["rays_d"]                # (V, H, W, 3)
    mask = data["mask"]                        # (H, W) bool
    V = int(rays_o_all.shape[0])

    n_views = int(rays_cfg.get("n_views", 6))
    max_rpv = int(rays_cfg.get("max_rays_per_view", 400))
    vidx = np.linspace(0, V, num=min(n_views, V), endpoint=False).astype(int)
    vidx = np.unique(vidx)

    mask_flat = mask.reshape(-1).bool()
    pix = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)   # corona pixel indices
    rng = np.random.default_rng(int(seed))

    ro, rd = [], []
    for v in vidx:
        rd_v = rays_d_all[v].reshape(-1, 3)[pix]                 # (n_corona, 3)
        n = rd_v.shape[0]
        sel = np.arange(n) if n <= max_rpv else rng.choice(n, size=max_rpv, replace=False)
        sel = torch.as_tensor(sel, dtype=torch.long)
        rd.append(rd_v[sel])
        ro.append(rays_o_all[v][None, :].expand(sel.numel(), 3))
    rays_o = torch.cat(ro, 0).to(device=state.renderer.device, dtype=state.renderer.dtype)
    rays_d = torch.cat(rd, 0).to(device=state.renderer.device, dtype=state.renderer.dtype)
    return rays_o, rays_d


def per_obs_sigma(I0_2d: torch.Tensor, state, whiten_cfg: dict) -> torch.Tensor:
    """σ_i for each observation (ray×channel), from the run's heteroscedastic noise model."""
    mode = str(whiten_cfg.get("mode", "noise"))     # noise | nominal | none
    I0 = I0_2d.detach().cpu().numpy()
    Nr, C = I0.shape
    if mode == "none":
        return torch.ones(Nr * C)

    obs = (state.cfg.get("observation_noise", {}) or {})
    if mode == "noise" and bool(obs.get("enabled", False)) and str(obs.get("model")) == "hetero_gaussian":
        gids = list(state.selected_channel_indices)
        a = np.asarray((obs.get("hetero_gaussian", {}) or {}).get("shot_coeff_by_global_channel", [0.0] * 64))
        b = np.asarray((obs.get("hetero_gaussian", {}) or {}).get("sigma_floor_by_global_channel", [0.0] * 64))
        a_c = np.asarray([a[g] for g in gids], dtype=float)[None, :]    # (1, C)
        b_c = np.asarray([b[g] for g in gids], dtype=float)[None, :]
        var = a_c * np.clip(I0, 0.0, None) + b_c ** 2
    else:
        frac = float(whiten_cfg.get("nominal_frac", 0.01))
        med = np.nanmedian(np.abs(I0), axis=0)[None, :]                 # (1, C) per-channel scale
        var = (frac * np.maximum(med, 1e-30)) ** 2 * np.ones_like(I0)

    sigma = np.sqrt(np.maximum(var, 1e-30)).reshape(-1)
    return torch.from_numpy(sigma).float()


# ----------------------------------------------------------------------
# the Jacobian (matrix, dense) via batched reverse-mode autograd
# ----------------------------------------------------------------------

def compute_jacobian(renderer, control, rays_o, rays_d, out_chunk: int = 512):
    """
    Form J = ∂(rendered intensities)/∂(control node values), shape (N, M).
    Uses torch.autograd.grad(..., is_grads_batched=True) one OUTPUT-chunk at a time
    (only first-order reverse autograd, which the renderer fully supports).
    """
    renderer.model = control
    renderer.use_gt_ne = False
    theta = control.values
    M = theta.numel()

    t_fwd = time.perf_counter()
    I = renderer.forward(rays_o, rays_d)
    I0 = I.detach().clone()
    I_flat = I.reshape(-1)
    N = I_flat.numel()
    logger.info("compute_jacobian: forward over %d rays in %.1fs  ->  N=%d obs, M=%d params",
                int(rays_o.shape[0]), time.perf_counter() - t_fwd, N, M)
    if N < M:
        logger.warning("N=%d < M=%d: underdetermined; null space incomplete (coarsen grid or add rays).", N, M)

    n_chunks = (N + out_chunk - 1) // out_chunk
    logger.info("compute_jacobian: forming J via %d reverse chunks of %d  (~%d backward passes — dominant cost)",
                n_chunks, out_chunk, N)

    J = torch.zeros(N, M, dtype=I_flat.dtype, device=I_flat.device)
    t0 = time.perf_counter()
    log_every = max(1, n_chunks // 20)
    for ci, start in enumerate(range(0, N, out_chunk)):
        end = min(start + out_chunk, N)
        nb = end - start
        cot = torch.zeros(nb, N, dtype=I_flat.dtype, device=I_flat.device)
        cot[torch.arange(nb), torch.arange(start, end)] = 1.0
        g = torch.autograd.grad(I_flat, theta, grad_outputs=cot,
                                is_grads_batched=True, retain_graph=True)[0]
        J[start:end] = g.reshape(nb, M)
        if (ci % log_every == 0) or (ci == n_chunks - 1):
            done = ci + 1
            el = time.perf_counter() - t0
            eta = el / done * (n_chunks - done)
            logger.info("  jacobian %3d/%3d chunks (%3.0f%%)  elapsed %5.1fs  eta %5.1fs",
                        done, n_chunks, 100.0 * done / n_chunks, el, eta)
    logger.info("compute_jacobian: J formed in %.1fs", time.perf_counter() - t0)
    return J, I0

@torch.no_grad()
def finite_difference_check(renderer, control, rays_o, rays_d, J, I0, n_probe: int = 32,
                            eps: float = 1e-3, seed: int = 0,
                            corr_thresh: float = 0.95, ratio_lo: float = 0.5, ratio_hi: float = 2.0) -> dict:
    """VERIFICATION: autograd J vs (F(m0+εe_j)−F(m0))/ε, on STRONG + random columns.
    Reports frac_good among non-trivial (non-near-zero) columns — robust to null-space columns."""
    renderer.model = control
    flat = control.values.reshape(-1); M = flat.numel()
    I0f = I0.reshape(-1).detach().cpu().numpy()
    Jnp = J.detach().cpu().numpy() if hasattr(J, "detach") else np.asarray(J)
    rng = np.random.default_rng(seed)
    col_norm = np.linalg.norm(Jnp, axis=0)
    strong = np.argsort(col_norm)[::-1][: n_probe // 2]                 # the columns that matter
    rand = rng.choice(M, size=max(n_probe - strong.size, 0), replace=False)
    probes = np.unique(np.concatenate([strong, rand]))

    recs = []
    for j in probes:
        o = float(flat[j].item()); flat[j] = o + eps
        Ip = renderer.forward(rays_o, rays_d).reshape(-1).detach().cpu().numpy(); flat[int(j)] = o
        fd = (Ip - I0f) / eps; Jc = Jnp[:, int(j)]
        nfd = float(np.linalg.norm(fd))
        if nfd < 1e-9 or np.std(Jc) == 0:                              # degenerate: both ~0
            recs.append({"j": int(j), "trivial": True}); continue
        recs.append({"j": int(j), "trivial": False,
                     "corr": float(np.corrcoef(fd, Jc)[0, 1]),
                     "ratio": float(np.linalg.norm(Jc) / nfd), "fd_norm": nfd})
    nz = [r for r in recs if not r["trivial"]]
    good = [r for r in nz if r["corr"] > corr_thresh and ratio_lo < r["ratio"] < ratio_hi]
    return {"n_probe": int(len(probes)), "n_nontrivial": len(nz),
            "frac_good": (len(good) / len(nz)) if nz else float("nan"),
            "median_corr": float(np.median([r["corr"] for r in nz])) if nz else float("nan"),
            "median_ratio": float(np.median([r["ratio"] for r in nz])) if nz else float("nan"),
            "eps": eps}


# ----------------------------------------------------------------------
# spectrum / Fisher / resolution
# ----------------------------------------------------------------------

def _effective_ranks(s: np.ndarray, tau: float) -> dict:
    s = np.asarray(s, dtype=float)
    p = s ** 2
    tot = float(p.sum())
    pr = (tot ** 2) / float((p ** 2).sum() + 1e-300)              # participation ratio (threshold-free)
    pn = p / (tot + 1e-300)
    ent = float(np.exp(-(pn * np.log(pn + 1e-300)).sum()))        # spectral-entropy rank
    return {"eff_rank_tau": int((s > tau).sum()),
            "eff_rank_participation": pr,
            "eff_rank_entropy": ent}


def analyze_state(state, *, grid_cfg, rays_cfg, whiten_cfg, tau=1.0,
                  tikhonov_lambda=1e-3, n_fd_probe=6, out_chunk=512, n_null_modes=4, seed=0,
                  prior_cfg=None, rescale=None) -> dict:
    device = state.renderer.device
    target = state.shared["reconstruction_target"]
    qlist = _quantities(target)
    logger.info("identifiability: target=%s  device=%s", target, device)

    t = time.perf_counter()
    control = build_control_field(state, grid_cfg, target, device)
    logger.info("[1/6] control grid (Q,L,P,R)=%s -> M=%d params  (%.1fs)",
                tuple(control.values.shape), control.values.numel(), time.perf_counter() - t)
    
    t = time.perf_counter()
    rays_o, rays_d = gather_ray_subset(state, rays_cfg, split=rays_cfg.get("split", "train"), seed=seed)
    logger.info("[2/6] ray subset: %d rays  (%.1fs)", int(rays_o.shape[0]), time.perf_counter() - t)

    t = time.perf_counter()
    J, I0 = compute_jacobian(state.renderer, control, rays_o, rays_d, out_chunk=out_chunk)  # (N,M),(Nr,C)
    logger.info("[3/6] Jacobian %s  (%.1fs)  <-- usually the bottleneck", tuple(J.shape), time.perf_counter() - t)
    
    t = time.perf_counter()
    fd = finite_difference_check(state.renderer, control, rays_o, rays_d, J, I0, n_probe=n_fd_probe, seed=seed)
    logger.info("[4/6] FD self-test: max_rel_err=%.2e mean=%.2e (want <~1e-2)  (%.1fs)",
                fd["fd_max_rel_err"], fd["fd_mean_rel_err"], time.perf_counter() - t)

    sigma = per_obs_sigma(I0, state, whiten_cfg).to(J.device)            # (N,)
    Jw = (J / sigma[:, None]).detach().cpu().numpy()                     # whitened J̃ (N,M)
    _ms = measurement_scale(Jw.shape[0], rescale)                       # √(N_full/n): subsample -> full operator
    if _ms != 1.0:
        Jw = Jw * _ms                                                   # scales s, H, info_diag consistently
        logger.info("analyze_state: measurement rescale x%.4g  (n=%d)", _ms, Jw.shape[0])

    # full dense SVD -> entire spectrum + all field-space modes
    t = time.perf_counter()
    U, s, Vt = np.linalg.svd(Jw, full_matrices=False)                    # s (M,), Vt (M,M) if N>=M
    logger.info("[5/6] SVD(%dx%d)  (%.1fs)", Jw.shape[0], Jw.shape[1], time.perf_counter() - t)

    eff = _effective_ranks(s, tau)
    s_id = s[s > tau]
    cond_full = float(s[0] / max(s[-1], 1e-30))
    cond_tau = float(s[0] / max(s_id[-1], 1e-30)) if s_id.size else float("inf")

    # Fisher diagonal (per-control-point information) and dense resolution matrix
    t = time.perf_counter()
    H = Jw.T @ Jw                                                        # (M,M) Gauss–Newton Hessian == Fisher
    info_diag = np.einsum("nm,nm->m", Jw, Jw)                            # diag(H) == Σ_i J̃_{ij}²
    R = np.linalg.solve(H + tikhonov_lambda * np.eye(H.shape[0]), H)     # resolution matrix
    res_diag = np.diag(R)
    logger.info("[6/6] Fisher diag + resolution (M=%d)  (%.1fs)", H.shape[0], time.perf_counter() - t)
    logger.info("summary: eff_rank(tau=%.2g)=%d  eff_rank(PR)=%.0f  eff_rank(ent)=%.0f  kappa_full=%.1f  kappa_tau=%.1f",
                tau, eff["eff_rank_tau"], eff["eff_rank_participation"], eff["eff_rank_entropy"],
                cond_full, cond_tau)

    # reshape (M,) -> grid (Q,L,P,R) for 3-D maps
    Lg, P, Rg = control.values.shape[1], control.values.shape[2], control.values.shape[3]
    Q = control.values.shape[0]
    def to_grid(vec):
        return np.asarray(vec, float).reshape(Q, Lg, P, Rg)
    info_grid = to_grid(info_diag)
    res_grid = to_grid(res_diag)
    null_modes = np.stack([to_grid(Vt[-(k + 1)]) for k in range(min(n_null_modes, Vt.shape[0]))], axis=0)

    post = posterior_diagnostics(H, s, Vt, (Q, Lg, P, Rg), prior_cfg or {})
    logger.info("posterior bridge: median CRB std=%.3g dex, median posterior std=%.3g dex, "
                "fraction prior-dominated=%.2f", post["crb_std_median"], post["post_std_median"],
                post["frac_prior_dominated"])

    return {
        "target": target, "quantities": qlist,
        "N": int(Jw.shape[0]), "M": int(Jw.shape[1]),
        "singular_values": s, "tau": float(tau),
        **eff, "condition_full": cond_full, "condition_tau": cond_tau,
        "fd_check": fd,
        "lon_axis": control.lon_axis_pad[:-1].detach().cpu().numpy(),   # unpadded
        "phi_axis": control.phi_axis.detach().cpu().numpy(),
        "r_axis": control.r_axis.detach().cpu().numpy(),
        "info_grid": info_grid, "res_grid": res_grid, "null_modes": null_modes,
        **post,
    }


# ----------------------------------------------------------------------
# single-run entry point + plots  (mirrors the paper-figure convention)
# ----------------------------------------------------------------------

def _shell_index(r_axis: np.ndarray, r_target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(r_axis, float) - float(r_target))))


def _plot_spectrum(res: dict, plot_cfg: dict, out_base: Path = None, ax=None):
    s = res["singular_values"]; tau = res["tau"]
    spectrum_color = plot_cfg.get("spectrum_color", "#009E73")
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=plot_cfg.get("spectrum_figsize", [6.0, 4.2]))
    ax.semilogy(np.arange(1, s.size + 1), np.maximum(s, 1e-30), color = spectrum_color, marker=".", lw=1)
    ax.axhline(tau, color="r", ls="--", lw=1, label=rf"$\tau$ = {tau:g} (identifiability threshold)")
    ax.set_xlabel(r"mode index $\ell$"); ax.set_ylabel(r"singular value $s_\ell$ (SNR units)")
    ax.set_title(rf"effective rank $E(\tau)$: {res['eff_rank_tau']}   |   "
                 rf"PR: {res['eff_rank_participation']:.0f}   |   $\kappa(\tau)$: {res['condition_tau']:.1f}")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    if not own:
        return ax
    formats = list(plot_cfg.get("formats", ["pdf", "png"])); dpi = int(plot_cfg.get("dpi", 300))
    _maybe_suptitle(fig, plot_cfg.get("title", None))
    fig.tight_layout()
    return _savefig(fig, out_base, formats, dpi), (plt.close(fig) or None)

def _plot_maps(res: dict, plot_cfg: dict, out_base: Path):
    formats = list(plot_cfg.get("formats", ["pdf", "png"])); dpi = int(plot_cfg.get("dpi", 300))
    bad = str(plot_cfg.get("bad_color", "black"))
    r_target = float(plot_cfg.get("map_radius", 1.5))
    ri = _shell_index(res["r_axis"], r_target)
    lon = np.rad2deg(res["lon_axis"]); lat = np.rad2deg(res["phi_axis"])
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    qlist = res["quantities"]
    cols = [(r"Fisher information  $\mathrm{diag}(H)$", res["info_grid"], _cmap_with_bad(plot_cfg.get("info_cmap", "viridis"), bad)),
            (r"resolution  $\mathrm{diag}(R)$",         res["res_grid"],  _cmap_with_bad(plot_cfg.get("res_cmap", "magma"), bad)),
            (r"null-space mode (smallest $s_k$)", res["null_modes"][0] if res["null_modes"].shape[0] else None,
             _cmap_with_bad(plot_cfg.get("null_cmap", "coolwarm"), bad))]
    fig, axs = plt.subplots(len(qlist), 3, figsize=plot_cfg.get("maps_figsize", [12, 3.2 * len(qlist)]),
                            squeeze=False, layout="constrained")
    for qi, q in enumerate(qlist):
        for ci, (lab, grid, cmap) in enumerate(cols):
            ax = axs[qi, ci]
            if grid is None:
                ax.axis("off"); continue                        
            up = int(plot_cfg.get("upsample", 0) or 0)
            img = (_upsample_shell(grid[qi, :, :, ri], res["lon_axis"], res["phi_axis"], n=up)
                   if up > 0 else grid[qi, :, :, ri].T)          # (lat,lon); upsample = display smoothing, # (P,L) = (lat,lon)
            diverging = (ci == 2)
            if diverging:
                lim = float(np.nanpercentile(np.abs(img), 99.0)) or 1.0
                im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
            else:
                im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, aspect="auto")
            if qi == 0:
                ax.set_title(lab)
            if ci == 0:
                ax.set_ylabel(_qty_tex(q, "field"))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _maybe_suptitle(fig, plot_cfg.get("maps_title", f"identifiability maps at r ≈ {res['r_axis'][ri]:.2f} $R_\\odot$"))
    return _savefig(fig, out_base, formats, dpi), (plt.close(fig) or None)

def _interp_shell(values_LP, lon_axis_rad, phi_axis_rad, lon_deg_t, lat_deg_t):
    """Bilinearly interpolate a coarse (L,P) control shell onto target (lat,lon) points (deg).
    Shared by the bridge (fine disagreement grid) and map display-upsampling."""
    from scipy.interpolate import RegularGridInterpolator
    itp = RegularGridInterpolator((phi_axis_rad, lon_axis_rad), np.asarray(values_LP).T,
                                  method="linear", bounds_error=False, fill_value=None)
    LON, LAT = np.meshgrid(np.deg2rad(lon_deg_t) % (2 * np.pi), np.deg2rad(lat_deg_t))
    return itp(np.stack([LAT, LON], axis=-1))                    # (len(lat), len(lon))

def _upsample_shell(vals_LP, lon_axis_rad, phi_axis_rad, n=200):
    """Display-only: upsample a coarse shell onto a dense n x n lat-lon grid, via _interp_shell."""
    lon_deg = np.linspace(float(np.rad2deg(lon_axis_rad).min()), float(np.rad2deg(lon_axis_rad).max()), n)
    lat_deg = np.linspace(float(np.rad2deg(phi_axis_rad).min()), float(np.rad2deg(phi_axis_rad).max()), n)
    return _interp_shell(vals_LP, lon_axis_rad, phi_axis_rad, lon_deg, lat_deg)

def _plot_posterior_maps(res, plot_cfg, out_base):
    from matplotlib.colors import LogNorm
    formats = list(plot_cfg.get("formats", ["pdf", "png"])); dpi = int(plot_cfg.get("dpi", 300))
    bad = str(plot_cfg.get("bad_color", "black"))
    r_target = float(plot_cfg.get("map_radius", 1.5)); ri = _shell_index(res["r_axis"], r_target)
    lon = np.rad2deg(res["lon_axis"]); lat = np.rad2deg(res["phi_axis"])
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    qlist = res["quantities"]

    std_cmap = _cmap_with_bad(plot_cfg.get("posterior_cmap", "inferno"), bad)
    dom_cmap = _cmap_with_bad(plot_cfg.get("prior_dominance_cmap", "viridis"), bad)
    cols = [
        (r"data-only CRB  $\sqrt{\mathrm{diag}\,H^{-1}}$ ", res["crb_std_grid"], std_cmap, "log"),
        (r"posterior  $\sqrt{\mathrm{diag}\,(H+\gamma L)^{-1}}$", res["post_std_grid"], std_cmap, "log"),
        (r"prior dominance  $\log_{10}(\mathrm{CRB}/\mathrm{post})$", res["prior_logratio_grid"], dom_cmap, "lin"),
    ]
    fig, axs = plt.subplots(len(qlist), 3, figsize=plot_cfg.get("posterior_figsize", [12, 3.2 * len(qlist)]),
                            squeeze=False, layout="constrained")
    for qi, q in enumerate(qlist):
        for ci, (lab, grid, cmap, scale) in enumerate(cols):
            ax = axs[qi, ci]
            up = int(plot_cfg.get("upsample", 0) or 0)
            img = (_upsample_shell(grid[qi, :, :, ri], res["lon_axis"], res["phi_axis"], n=up)
                    if up > 0 else grid[qi, :, :, ri].T)          # (lat,lon); upsample = display smoothing, # (P,L) = (lat,lon)
            finite = img[np.isfinite(img)]
            if scale == "log":
                lo = float(np.nanpercentile(finite[finite > 0], 1.0)) if np.any(finite > 0) else 1e-3
                hi = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
                norm = LogNorm(vmin=max(lo, 1e-6), vmax=max(hi, lo * 10))
                im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, norm=norm,
                               aspect="auto", interpolation=plot_cfg.get("interpolation", "nearest"))
            else:
                hi = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
                im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=max(hi, 1e-9),
                               aspect="auto", interpolation=plot_cfg.get("interpolation", "nearest"))
            if qi == 0:
                ax.set_title(lab)
            if ci == 0:
                ax.set_ylabel(_qty_tex(q, "field"))
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _maybe_suptitle(fig, plot_cfg.get("posterior_title",
                    rf"data-only vs posterior uncertainty at r $\approx$ {res['r_axis'][ri]:.2f} $R_\odot$"))
    return _savefig(fig, out_base, formats, dpi), None


def generate_identifiability_run_figure(spec_path, device_override=None, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("figure", {})
    name = str(cfg.get("name", spec_path.stem))
    out_dir = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg.get("output_dir", "../../runs_identifiability"), spec_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_override if device_override is not None else cfg.get("device", None)
    path_remap = cfg.get("path_remap", None)
    plot_cfg = dict(cfg.get("plot", {}) or {})

    cond = cfg["condition"]
    bdir = _resolve_path(cond["benchmark_dir"], spec_root)
    rows = collect_run_summaries(bdir)
    sel = {k: cond[k] for k in ("experiment", "where", "filters", "run_names", "run_dirs", "min_seeds", "max_seeds") if k in cond}
    sel.setdefault("min_seeds", 1)
    seed_rows = _collect_condition_seed_rows(rows, sel)          # any seed: analysis is at GT, seed-independent
    run_dir = _resolve_run_dir(seed_rows[0], bdir)

    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=path_remap)
    try:
        res = analyze_state(
            state,
            grid_cfg=dict(cfg.get("grid", {}) or {}),
            rays_cfg=dict(cfg.get("rays", {}) or {}),
            whiten_cfg=dict(cfg.get("whiten", {}) or {}),
            rescale=dict(cfg.get("rescale", {}) or {}),
            tau=float(cfg.get("tau", 1.0)),
            tikhonov_lambda=float(cfg.get("tikhonov_lambda", 1e-3)),
            n_fd_probe=int(cfg.get("n_fd_probe", 6)),
            out_chunk=int(cfg.get("out_chunk", 512)),
            n_null_modes=int(cfg.get("n_null_modes", 4)),
            seed=int(cfg.get("seed", 0)),
            prior_cfg=dict(cfg.get("prior", {}) or {}),
        )
    finally:
        _release_state(state)

    logger.info("FD self-test max rel err = %.2e (want <~1e-2)", res["fd_check"]["fd_max_rel_err"])
    np.savez_compressed(out_dir / f"{name}_arrays.npz",
                        singular_values=res["singular_values"], info_grid=res["info_grid"],
                        res_grid=res["res_grid"], null_modes=res["null_modes"],
                        lon_axis=res["lon_axis"], phi_axis=res["phi_axis"], r_axis=res["r_axis"],
                        quantities=np.array(res["quantities"]),
                        crb_std_grid=res["crb_std_grid"], post_std_grid=res["post_std_grid"],
                        prior_logratio_grid=res["prior_logratio_grid"],)
    spec_paths, _ = _plot_spectrum(res, plot_cfg, out_dir / f"{name}_spectrum")
    map_paths, _ = _plot_maps(res, plot_cfg, out_dir / f"{name}_maps")
    post_paths, _ = _plot_posterior_maps(res, plot_cfg, out_dir / f"{name}_posterior")

    manifest = {"name": name, "kind": "identifiability_run",
                "benchmark_dir": str(bdir), "run_dir": str(run_dir),
                "N": res["N"], "M": res["M"], "tau": res["tau"],
                "eff_rank_tau": res["eff_rank_tau"], "eff_rank_participation": res["eff_rank_participation"],
                "eff_rank_entropy": res["eff_rank_entropy"],
                "condition_full": res["condition_full"], "condition_tau": res["condition_tau"],
                "fd_check": res["fd_check"], "spectrum": spec_paths, "maps": map_paths,
                "posterior": post_paths,
                "prior_lambda": res["prior_lambda"], "crb_floor": res["crb_floor"],
                "frac_prior_dominated": res["frac_prior_dominated"],
                "crb_std_median": res["crb_std_median"], "post_std_median": res["post_std_median"],}
    save_json(manifest, out_dir / f"{name}_manifest.json")
    return manifest


# ======================================================================
# ACROSS-CONDITION SWEEPS: effective rank vs #lines / #views / noise
#
# One Jacobian, many SVDs:
#   compute J once on a base dataset (all channels, many views), then
#     #lines : SVD of J restricted to those channels' rows
#     #views : SVD of J restricted to those views' rows
#     noise  : re-whiten J with sigma scaled by the noise level, re-SVD
# Each panel's x-axis is a list of points in the YAML; fully customizable.
# ======================================================================

_METRIC_LABEL = {
    "eff_rank_tau":           r"effective rank  ($s_k>\tau$)",
    "eff_rank_participation": "participation ratio",
    "eff_rank_entropy":       "spectral-entropy rank",
    "condition_tau":          r"condition number $\kappa(\tau)$",
    "s_max":                  r"$s_{\max}$",
}
_LOGY_DEFAULT = {"condition_tau", "s_max"}   # these read better on a log y-axis


def _gather_rays_with_views(state, rays_cfg, split, seed):
    """Like gather_ray_subset, but also returns the view index of every ray."""
    data = state.train_data if split == "train" else state.test_data
    rays_o_all, rays_d_all, mask = data["rays_o"], data["rays_d"], data["mask"]
    V = int(rays_o_all.shape[0])
    n_views = min(int(rays_cfg.get("n_views", 40)), V)
    max_rpv = int(rays_cfg.get("max_rays_per_view", 20))
    vidx = np.unique(np.linspace(0, V, num=n_views, endpoint=False).astype(int))
    pix = torch.nonzero(mask.reshape(-1).bool(), as_tuple=False).squeeze(-1)
    rng = np.random.default_rng(int(seed))
    ro, rd, vtag = [], [], []
    for v in vidx:
        rd_v = rays_d_all[v].reshape(-1, 3)[pix]
        n = rd_v.shape[0]
        sel = np.arange(n) if n <= max_rpv else rng.choice(n, size=max_rpv, replace=False)
        sel = torch.as_tensor(sel, dtype=torch.long)
        rd.append(rd_v[sel]); ro.append(rays_o_all[v][None, :].expand(sel.numel(), 3))
        vtag.append(np.full(sel.numel(), int(v)))
    rays_o = torch.cat(ro, 0).to(device=state.renderer.device, dtype=state.renderer.dtype)
    rays_d = torch.cat(rd, 0).to(device=state.renderer.device, dtype=state.renderer.dtype)
    return rays_o, rays_d, np.concatenate(vtag), vidx


def _compute_base_operator(spec_root, cfg, device, out_dir, name):
    """Build (or load from cache) the base Jacobian + per-observation tags."""
    cache = out_dir / f"{name}_operator.npz"
    if cache.exists() and not bool(cfg.get("force_recompute", False)):
        logger.info("loading cached operator: %s", cache.name)
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    cond = cfg["base"]
    bdir = _resolve_path(cond["benchmark_dir"], spec_root)
    rows = collect_run_summaries(bdir)
    sel = {k: cond[k] for k in ("experiment", "where", "filters", "run_names", "run_dirs", "min_seeds", "max_seeds") if k in cond}
    sel.setdefault("min_seeds", 1)
    seed_rows = _collect_condition_seed_rows(rows, sel)
    run_dir = _resolve_run_dir(seed_rows[0], bdir)
    logger.info("base operator from run_dir=%s", run_dir.name)

    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=cfg.get("path_remap", None))
    try:
        if cfg.get("renderer_step_size") is not None:           # speed lever: coarsen LOS sampling
            state.renderer.step_size = float(cfg["renderer_step_size"])
            logger.info("overrode renderer.step_size -> %g", state.renderer.step_size)

        target = state.shared["reconstruction_target"]
        control = build_control_field(state, dict(cfg.get("grid", {}) or {}), target, device)
        rays_o, rays_d, view_per_ray, vidx = _gather_rays_with_views(
            state, dict(cfg.get("rays", {}) or {}), cfg.get("rays", {}).get("split", "train"), int(cfg.get("seed", 0)))

        import time
        t = time.perf_counter()
        J, I0 = compute_jacobian(state.renderer, control, rays_o, rays_d, out_chunk=int(cfg.get("out_chunk", 512)))
        logger.info("base Jacobian %s formed in %.1fs", tuple(J.shape), time.perf_counter() - t)

        C = int(I0.shape[1])
        Nr = int(I0.shape[0])
        obs_view = np.repeat(view_per_ray, C)
        chan_local = np.tile(np.arange(C), Nr)
        gids = np.asarray(list(state.selected_channel_indices))
        obs_chan_global = gids[chan_local]
        op = {
            "J": J.detach().cpu().numpy().astype(np.float32),
            "I0": I0.reshape(-1).detach().cpu().numpy().astype(np.float32),
            "obs_view": obs_view.astype(np.int32),
            "obs_chan_global": obs_chan_global.astype(np.int32),
            "vidx": vidx.astype(np.int32),
            "M": np.int64(J.shape[1]),
        }
    finally:
        _release_state(state)

    np.savez_compressed(cache, **op)
    logger.info("cached operator -> %s", cache.name)
    return op

def measurement_scale(n_sampled, rescale_cfg) -> float:
    """Rescale a *subsampled* whitened Jacobian so its singular values represent the FULL
    measurement operator: an unbiased estimate of the full Fisher is (N_full/n)·J̃ᵀJ̃, i.e.
    J̃ is scaled by c=√(N_full/n_sampled). Scaling multiplies every s_ℓ by c and leaves the
    right singular vectors unchanged, so E(τ), f_null and κ_τ shift while PR/κ_full/rankings do not.
    Returns 1.0 when disabled. n_full = n_views·pixels_per_view·n_channels of the full observation."""
    rc = dict(rescale_cfg or {})
    if not rc.get("enabled", False):
        return 1.0
    n_full = rc.get("n_full", None)
    if n_full in (None, 0, "auto"):
        raise ValueError("rescale.enabled=true requires an explicit integer n_full "
                         "(= n_views * pixels_per_view * n_channels of the full observation).")
    return float(np.sqrt(float(n_full) / max(int(n_sampled), 1)))

def _sigma_for(op, shot, floor, mult):
    ch = op["obs_chan_global"].astype(int)
    a = np.asarray(shot, float)[ch]
    b = np.asarray(floor, float)[ch]
    var = float(mult) * a * np.clip(op["I0"].astype(float), 0.0, None) + b ** 2
    sigma = np.sqrt(np.maximum(var, 1e-30))
    # Fold the measurement-count rescale into the whitened operator. Callers form Jw = J / sigma,
    # so dividing sigma by c = √(N_full/n) multiplies J̃ (and its singular values) by c.
    return sigma / float(op["_meta"].get("meas_scale", 1.0))

def _svd_eff(Jw, tau):
    s = np.linalg.svd(Jw, compute_uv=False)              # singular values only -> fast
    p = s ** 2; tot = float(p.sum()) + 1e-300; pn = p / tot
    sid = s[s > tau]
    return {
        "eff_rank_tau": float((s > tau).sum()),
        "eff_rank_participation": float((tot ** 2) / float((p ** 2).sum() + 1e-300)),
        "eff_rank_entropy": float(np.exp(-(pn * np.log(pn + 1e-300)).sum())),
        "condition_tau": float(s[0] / max(sid[-1], 1e-30)) if sid.size else float("nan"),
        "s_max": float(s[0]),
        "n_obs": int(Jw.shape[0]),
    }

def build_grid_laplacian(Q, L, P, R, *, w_lon=1.0, w_lat=1.0, w_r=1.0, periodic_lon=True, ridge=1e-3):
    """
    Anisotropic graph-Laplacian PRECISION on the (Q,L,P,R) control grid, block-diagonal
    over quantities (n_e and T don't couple in the prior). This is R_prior: a GMRF
    smoothness prior whose precision penalizes differences between neighbouring cells
    (lon is periodic). 'ridge' adds a tiny mass term so H + lambda*R is invertible even
    where the data are blind. Dense for coarse grids; swap to scipy.sparse for large M.
    """
    M = Q * L * P * R
    Rp = np.zeros((M, M), dtype=np.float64)
    def idx(q, i, j, k): return ((q * L + i) * P + j) * R + k
    for q in range(Q):
        for i in range(L):
            for j in range(P):
                for k in range(R):
                    a = idx(q, i, j, k); nb = []
                    if L > 1:
                        if periodic_lon:
                            nb += [(idx(q, (i + 1) % L, j, k), w_lon), (idx(q, (i - 1) % L, j, k), w_lon)]
                        else:
                            if i + 1 < L: nb.append((idx(q, i + 1, j, k), w_lon))
                            if i - 1 >= 0: nb.append((idx(q, i - 1, j, k), w_lon))
                    if j + 1 < P: nb.append((idx(q, i, j + 1, k), w_lat))
                    if j - 1 >= 0: nb.append((idx(q, i, j - 1, k), w_lat))
                    if k + 1 < R: nb.append((idx(q, i, j, k + 1), w_r))
                    if k - 1 >= 0: nb.append((idx(q, i, j, k - 1), w_r))
                    deg = 0.0
                    for b, w in nb:
                        Rp[a, b] -= w; deg += w
                    Rp[a, a] += deg
    Rp[np.diag_indices_from(Rp)] += float(ridge)
    return Rp


def posterior_diagnostics(H, s, Vt, grid_shape, prior_cfg):
    """
    The posterior-covariance bridge. Returns per-voxel uncertainty maps (dex), all in the
    SAME field units as ensemble disagreement (because J was noise-whitened):

      crb_std       = sqrt(diag(H^-1))            data-only uncertainty (Cramer-Rao); HUGE in the null space
      post_std      = sqrt(diag((H + lam*R)^-1))  data + smoothness-prior posterior; the prior shrinks the null space
      prior_logratio= log10(crb_std / post_std)   "where is the reconstruction prior-dominated vs data-dominated"

    diag(H^-1) is computed from the SVD (no explicit inverse): diag = sum_k v_k[:]^2 / s_k^2,
    floored by crb_floor so near-null modes stay finite.
    """
    Q, L, P, R = grid_shape
    H = np.asarray(H, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    Vt = np.asarray(Vt, dtype=np.float64)
    pc = dict(prior_cfg or {})

    crb_floor = float(pc.get("crb_floor", 1e-2))
    inv_s2 = 1.0 / (s ** 2 + crb_floor ** 2)
    crb_var = np.einsum("km,k->m", Vt ** 2, inv_s2)                       # diag(H^-1), (M,)

    Rp = build_grid_laplacian(Q, L, P, R,
                              w_lon=float(pc.get("w_lon", 1.0)), w_lat=float(pc.get("w_lat", 1.0)),
                              w_r=float(pc.get("w_r", 1.0)), periodic_lon=bool(pc.get("periodic_lon", True)),
                              ridge=float(pc.get("ridge", 1e-3)))
    lam = float(pc.get("lambda", 1.0))
    Cpost = np.linalg.inv(H + lam * Rp)                                   # dense; fine for coarse M
    post_var = np.clip(np.diag(Cpost), 0.0, None)

    def to_grid(v): return np.asarray(v, float).reshape(Q, L, P, R)
    crb_std = to_grid(np.sqrt(np.clip(crb_var, 0.0, None)))
    post_std = to_grid(np.sqrt(post_var))
    prior_logratio = np.log10(np.maximum(crb_std, 1e-30) / np.maximum(post_std, 1e-30))

    return {
        "crb_std_grid": crb_std,
        "post_std_grid": post_std,
        "prior_logratio_grid": prior_logratio,
        "prior_lambda": lam, "crb_floor": crb_floor,
        "frac_prior_dominated": float(np.mean(prior_logratio > np.log10(2.0))),  # >2x reduction
        "crb_std_median": float(np.median(crb_std)),
        "post_std_median": float(np.median(post_std)),
    }


def _eval_panel(op, panel, base_shot, base_floor, tau):
    kind = str(panel["kind"])
    xs, rows = [], {}
    for pt in panel["points"]:
        x = float(pt["x"])
        if kind == "lines":
            mask = np.isin(op["obs_chan_global"], np.asarray(pt["channels"], int))
            sigma = _sigma_for(op, base_shot, base_floor, 1.0)
        elif kind == "views":
            nsel = min(int(round(x)), op["vidx"].size)
            if int(round(x)) > op["vidx"].size:
                logger.warning("views point %d > %d gathered views; clamped (raise rays.n_views to extend).",
                               int(round(x)), op["vidx"].size)
            sel_views = op["vidx"][np.unique(np.linspace(0, op["vidx"].size, nsel, endpoint=False).astype(int))]
            mask = np.isin(op["obs_view"], sel_views)
            sigma = _sigma_for(op, base_shot, base_floor, 1.0)
        elif kind == "noise":
            mask = np.ones(op["J"].shape[0], bool)
            sigma = _sigma_for(op, base_shot, base_floor, x)        # var scales with the noise multiplier
        else:
            raise ValueError(f"unknown panel kind={kind}")
        Jw = op["J"][mask] / sigma[mask][:, None]
        m = _svd_eff(Jw, tau)
        xs.append(x)
        for k, v in m.items():
            rows.setdefault(k, []).append(v)
    return np.asarray(xs, float), rows


def generate_identifiability_sweeps_figure(spec_path, device_override=None, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("figure", {})
    name = str(cfg.get("name", spec_path.stem))
    out_dir = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg.get("output_dir", "../../runs_identifiability"), spec_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_override if device_override is not None else cfg.get("device", None)

    tau = float(cfg.get("tau", 1.0))
    base_shot = list(cfg.get("base_shot_coeff", [1.0e-2, 5.0e-4, 1.0e-2, 5.0e-2]))
    base_floor = list(cfg.get("base_sigma_floor", [0.0] * len(base_shot)))
    panels = list(cfg.get("panels", []) or [])
    metrics = list(cfg.get("metrics", ["eff_rank_tau", "eff_rank_participation"]))
    plot_cfg = dict(cfg.get("plot", {}) or {})
    if not panels:
        raise ValueError("sweeps figure requires figure.panels")

    op = _compute_base_operator(spec_root, cfg, device, out_dir, name)
    results = {p["name"]: _eval_panel(op, p, base_shot, base_floor, tau) for p in panels}

    # ---- plot: rows = metrics, cols = panels ----
    formats = list(plot_cfg.get("formats", ["pdf", "png"])); dpi = int(plot_cfg.get("dpi", 300))
    logy_metrics = set(plot_cfg.get("logy_metrics", list(_LOGY_DEFAULT)))
    nrow, ncol = len(metrics), len(panels)
    rc = {"font.size": float(plot_cfg.get("font_size", 9)),
          "axes.titlesize": float(plot_cfg.get("title_font_size", 10)),
          "xtick.labelsize": float(plot_cfg.get("tick_font_size", 8)),
          "ytick.labelsize": float(plot_cfg.get("tick_font_size", 8))}
    with plt.rc_context(rc):
        fig, axs = plt.subplots(nrow, ncol, squeeze=False,
                                figsize=plot_cfg.get("figsize", [4.6 * ncol, 3.6 * nrow]))
        for ci, panel in enumerate(panels):
            xs, rows = results[panel["name"]]
            color = panel.get("color", None)
            for ri, met in enumerate(metrics):
                ax = axs[ri, ci]
                ax.plot(xs, rows[met], marker=panel.get("marker", "o"), color=color, lw=1.5)
                if ri == 0:
                    ax.set_title(panel.get("title", panel["name"]))
                if ri == nrow - 1:
                    ax.set_xlabel(panel.get("xlabel", panel["name"]))
                if ci == 0:
                    ax.set_ylabel(plot_cfg.get("ylabels", {}).get(met, _METRIC_LABEL.get(met, met)))
                if panel.get("logx", False):
                    ax.set_xscale("log")
                if met in logy_metrics:
                    ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
        _maybe_suptitle(fig, plot_cfg.get("title", None))
        fig.tight_layout()
        out_paths = _savefig(fig, out_dir / name, formats, dpi)
        plt.close(fig)

    manifest = {"name": name, "kind": "identifiability_sweeps", "tau": tau,
                "base_shot_coeff": base_shot, "metrics": metrics, "out_paths": out_paths,
                "N_base": int(op["J"].shape[0]), "M": int(op["M"]),
                "panels": {p["name"]: {"x": results[p["name"]][0].tolist(),
                                       **{m: results[p["name"]][1][m] for m in metrics}}
                           for p in panels}}
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("sweeps figure saved: %s", out_dir / f"{name}.png")
    return manifest
