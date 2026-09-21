##########################################################
### identifiability/operator.py
### Build ONCE (expensive), save EVERYTHING needed for diagnostics.
### Artifact = operator.npz (arrays) + operator_meta.json (provenance).
### All diagnostics (Stage 2) read this; the Jacobian is never recomputed.
##########################################################

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np

from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..benchmark.condition_compare import _build_state_for_analysis, _release_state, _resolve_run_dir
from ..util.figure import _resolve_path
from ..benchmark.select import _collect_condition_seed_rows
from ._core import build_control_field, _gather_rays_with_views, compute_jacobian, finite_difference_check, measurement_scale

logger = logging.getLogger("coroNeRF.identifiability.operator")

OPERATOR_NPZ = "operator.npz"
OPERATOR_META = "operator_meta.json"


def _spec_tag(name: str, cfg: dict) -> str:
    """Short reproducible tag so different conditions/grids never collide on disk."""
    key = {k: cfg.get(k) for k in ("base", "grid", "rays", "whiten", "renderer_step_size", "seed")}
    h = hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{name}__{h}"


def build_operator(state, *, grid_cfg, rays_cfg, whiten_cfg,
                   renderer_step_size=None, out_chunk=512, seed=0,
                   control_source="gt") -> dict:
    """Compute the linearized operator on the control grid (GT=m* or recon=m̂) and bundle
    everything needed downstream."""
    device = state.renderer.device
    target = state.shared["reconstruction_target"]
    if renderer_step_size is not None:
        state.renderer.step_size = float(renderer_step_size)
        logger.info("renderer.step_size set to %g for the build", state.renderer.step_size)

    control = build_control_field(state, dict(grid_cfg or {}), target, device, source=control_source)
    rays_o, rays_d, view_per_ray, vidx = _gather_rays_with_views(
        state, dict(rays_cfg or {}), (rays_cfg or {}).get("split", "train"), int(seed))

    t = time.perf_counter()
    J, I0 = compute_jacobian(state.renderer, control, rays_o, rays_d, out_chunk=int(out_chunk))
    logger.info("operator: Jacobian %s computed in %.1fs", tuple(J.shape), time.perf_counter() - t)

    try:
        fd = finite_difference_check(state.renderer, control, rays_o, rays_d, J, I0,
                                    n_probe=int(getattr(__import__("builtins"), "int")(6)), seed=int(seed))
    except Exception as e:                       # never let the verification abort a build
        fd = {"fd_error": str(e)}
    logger.info("operator FD self-test: frac_good=%.2f median_corr=%.3f (n=%d)",
                    fd.get("frac_good", float("nan")), fd.get("median_corr", float("nan")),
                    fd.get("n_nontrivial", 0))

    C = int(I0.shape[1]); Nr = int(I0.shape[0])
    obs_view = np.repeat(view_per_ray, C).astype(np.int32)
    chan_local = np.tile(np.arange(C), Nr)
    gids = np.asarray(list(state.selected_channel_indices))
    obs_chan_global = gids[chan_local].astype(np.int32)
    Q, L, P, R = (int(x) for x in control.values.shape)

    return {
        # arrays -> operator.npz
        "J": J.detach().cpu().numpy().astype(np.float32),
        "I0": I0.reshape(-1).detach().cpu().numpy().astype(np.float32),
        "obs_view": obs_view,
        "obs_chan_global": obs_chan_global,
        "vidx": vidx.astype(np.int32),
        "lon_axis": control.lon_axis_pad[:-1].detach().cpu().numpy().astype(np.float32),
        "phi_axis": control.phi_axis.detach().cpu().numpy().astype(np.float32),
        "r_axis": control.r_axis.detach().cpu().numpy().astype(np.float32),
        "control_values": control.values.detach().cpu().numpy().astype(np.float32),
        # provenance -> operator_meta.json
        "_meta": {
            "grid_shape": [Q, L, P, R], "M": int(Q * L * P * R), "N": int(J.shape[0]),
            "target": target, "selected_channel_indices": [int(x) for x in gids],
            "base_shot_coeff": list(whiten_cfg.get("base_shot_coeff", [1.0e-2, 5.0e-4, 1.0e-2, 5.0e-2])),
            "base_sigma_floor": list(whiten_cfg.get("base_sigma_floor", [0.0, 0.0, 0.0, 0.0])),
            "renderer_step_size": float(state.renderer.step_size),
            "seed": int(seed),
            "fd_check": fd,
        },
    }


def save_operator(op: dict, out_dir: Union[str, Path]) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in op.items() if k != "_meta"}
    np.savez_compressed(out_dir / OPERATOR_NPZ, **arrays)
    save_json(op["_meta"], out_dir / OPERATOR_META)
    logger.info("saved operator artifact -> %s", out_dir)
    return out_dir


def load_operator(out_dir: Union[str, Path]) -> dict:
    out_dir = Path(out_dir)
    d = np.load(out_dir / OPERATOR_NPZ, allow_pickle=False)
    op = {k: d[k] for k in d.files}
    with open(out_dir / OPERATOR_META) as f:
        op["_meta"] = json.load(f)
    return op


def build_operator_from_spec(spec_path, device_override=None, output_dir_override=None) -> Path:
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    raw = load_yaml(spec_path)
    cfg = dict(raw.get("operator", raw.get("figure", {})))
    cfg.setdefault("name", spec_path.stem)
    output_root = _resolve_path(output_dir_override, Path.cwd()) if output_dir_override else None
    return build_operator_from_cfg(cfg, spec_root, device_override=device_override, output_root=output_root)

def _stamp_meas_scale(out_dir: Union[str, Path], cfg: dict) -> None:
    """Compute meas_scale=√(N_full/N) from cfg['rescale'] and write it into operator_meta.json.
    Idempotent; safe on cache hits (never rebuilds J). No-op / meas_scale=1.0 when disabled."""
    rescale = dict(cfg.get("rescale", {}) or {})
    meta_path = Path(out_dir) / OPERATOR_META
    if not meta_path.exists():
        return
    with open(meta_path) as f:
        meta = json.load(f)
    ms = measurement_scale(int(meta.get("N", 0)), rescale)
    meta["meas_scale"] = float(ms)
    meta["rescale_cfg"] = rescale
    save_json(meta, meta_path)
    logger.info("operator %s: meas_scale=%.4g  (n_full=%s / N=%s)",
                Path(out_dir).name, ms, rescale.get("n_full"), meta.get("N"))

def build_operator_from_cfg(cfg: dict, spec_root, *, device_override=None, output_root=None) -> Path:
    """Build (or reuse cached) an operator from an in-memory operator cfg dict. Relative paths in
    cfg (benchmark_dir, output_dir) resolve against spec_root. Shared core of build_operator_from_spec
    and the sweep's base build."""
    cfg = dict(cfg)
    name = str(cfg.get("name", "operator"))
    op_pt = dict(cfg.get("operating_point", {}) or {}); _kind = str(op_pt.get("kind", "gt"))
    _si = int(op_pt.get("seed_index", 0))
    if _kind == "recon":    name = f"{name}__mhat-s{_si}"
    elif _kind == "interp": name = f"{name}__interp{float(op_pt.get('alpha',1.0)):g}-s{_si}"
    root = (Path(output_root) if output_root is not None
            else _resolve_path(cfg.get("output_dir", "../../runs_identifiability"), spec_root))
    out_dir = root / "operators" / _spec_tag(name, cfg)

    if (out_dir / OPERATOR_NPZ).exists() and not bool(cfg.get("force_recompute", False)):
        logger.info("operator already built at %s (set force_recompute: true to rebuild)", out_dir)
        _stamp_meas_scale(out_dir, cfg)                     # refresh meas_scale without rebuilding J
        return out_dir

    device = device_override if device_override is not None else cfg.get("device")
    path_remap = cfg.get("path_remap")
    cond = cfg["base"]
    bdir = _resolve_path(cond["benchmark_dir"], spec_root)
    rows = collect_run_summaries(bdir)
    sel = {k: cond[k] for k in ("experiment", "where", "filters", "run_names", "run_dirs", "min_seeds", "max_seeds") if k in cond}
    sel.setdefault("min_seeds", 1)
    
    # get seed rows
    seed_rows = _collect_condition_seed_rows(rows, sel)
    _idx = _si if _kind in ("recon", "interp") else 0
    if _idx >= len(seed_rows):
        raise ValueError(f"seed_index {_idx} >= {len(seed_rows)} completed seeds for {sel}")
    run_dir = _resolve_run_dir(seed_rows[_idx], bdir)
    
    logger.info("building operator | name=%s | run_dir=%s", name, run_dir.name)

    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=path_remap)
    try:
        op = build_operator(
            state, grid_cfg=cfg.get("grid", {}), rays_cfg=cfg.get("rays", {}),
            whiten_cfg=cfg.get("whiten", {}), renderer_step_size=cfg.get("renderer_step_size"),
            out_chunk=int(cfg.get("out_chunk", 512)), seed=int(cfg.get("seed", 0)),
            control_source=("recon" if str(op_pt.get("kind", "gt")) == "recon" else "gt"),
        )
        op["_meta"].update({
            "name": name, "tag": out_dir.name, "run_dir": str(run_dir), "benchmark_dir": str(bdir),
            "condition": cond, "grid_cfg": dict(cfg.get("grid", {})), "rays_cfg": dict(cfg.get("rays", {})),
            "whiten_cfg": dict(cfg.get("whiten", {})), "date": datetime.now().isoformat(timespec="seconds"),
            "operating_point": op_pt,
        })
        save_operator(op, out_dir)
    finally:
        _release_state(state)
    _stamp_meas_scale(out_dir, cfg)
    return out_dir

def subset_operator(op, *, view_subset=None, channel_subset=None) -> dict:
    """Row-subset VIEW of an operator (reuses J; no recompute). Field discretization
    (axes, control_values, grid_shape) is shared and untouched."""
    N = op["J"].shape[0]
    mask = np.ones(N, bool)
    if view_subset is not None:
        mask &= np.isin(op["obs_view"], np.asarray(view_subset, int))
    if channel_subset is not None:
        mask &= np.isin(op["obs_chan_global"], np.asarray(channel_subset, int))
    if not mask.any():
        raise ValueError("subset_operator: empty row selection (check views/channels vs base)")
    sub = dict(op)                                   # shallow copy; share axes/control_values
    for k in ("J", "I0", "obs_view", "obs_chan_global"):
        sub[k] = op[k][mask]
    sub["vidx"] = np.unique(sub["obs_view"]).astype(np.int32)
    meta = dict(op["_meta"]); meta["N"] = int(mask.sum())
    if channel_subset is not None:
        meta["selected_channel_indices"] = sorted(int(c) for c in np.unique(sub["obs_chan_global"]))
    sub["_meta"] = meta
    return sub


def pick_views(vidx, n) -> np.ndarray:
    """Evenly-spaced subset of n view indices from vidx (matches _eval_panel's views logic)."""
    n = min(int(n), int(vidx.size))
    idx = np.unique(np.linspace(0, vidx.size, n, endpoint=False).astype(int))
    return np.asarray(vidx)[idx]
