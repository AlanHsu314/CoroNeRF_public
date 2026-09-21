from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._paper_cli import REPO_ROOT                      # <- shared repo root (parents[2] of coronerf/tools)
from ..benchmark.config import load_yaml
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.select import _collect_condition_seed_rows
from ..benchmark.condition_compare import _build_state_for_analysis, _resolve_run_dir, _release_state
from ..util.figure import _resolve_path
from ..visualization.los_render import render_los_map
from ..visualization.geometry import make_coronal_mask_simple
from ..paper.channel_strip import _stats_limits
from ..paper.los_filmstrip import _QDEF


'''
python -m coronerf.tools.diagram --kind channels --spec diagram_channels.yaml
python -m coronerf.tools.diagram --kind corona   --spec diagram_corona.yaml --device cuda
python -m coronerf.tools.diagram --kind field    --spec diagram_field.yaml  --device cuda
'''

logger = logging.getLogger("coroNeRF.tools.diagram")

SPEC_DIR = REPO_ROOT / "configs" / "diagram"


# ---------------------------------------------------------------- path resolution
def _resolve_spec(spec) -> Path:
    """Resolve a diagram YAML spec, mirroring _paper_cli.resolve_paper_spec_path:
       absolute path | CWD-relative | REPO_ROOT/configs/diagram/<name>."""
    p = Path(spec)
    cands = [p] if p.is_absolute() else [Path.cwd() / p, SPEC_DIR / p]
    for c in cands:
        if c.exists():
            return c.resolve()
    return (p if p.is_absolute() else (SPEC_DIR / p)).resolve()   # clearest error if truly missing

def _cfg(spec, key):
    """Resolve the spec against REPO_ROOT/configs/diagram, then return (block, spec_root).
       spec_root is the resolved spec's parent, so all relative paths in the yaml anchor there."""
    spec_path = _resolve_spec(spec)
    return dict(load_yaml(spec_path).get(key, {}) or {}), spec_path.parent

def _outdir(cfg, root, override, sub) -> Path:
    if override:
        d = Path(override); d = d if d.is_absolute() else (Path.cwd() / d)
    elif cfg.get("output_dir"):
        d = _resolve_path(cfg["output_dir"], root)               # relative to the spec's location
    else:
        d = REPO_ROOT / "paper_outputs" / "diagram" / sub        # REPO_ROOT-anchored default
    d = d.resolve(); d.mkdir(parents=True, exist_ok=True)
    return d


def _save_bare(m, out_base, *, cmap="inferno", vmin=None, vmax=None, log=False,
               keep=None, dpi=300, formats=("png",), pre_logged=False,
               transparent=True, bad_color=None):
    """Write a bare (H,W) map: no axes/titles/colorbars, masked pixels transparent."""
    a = np.asarray(m, float)
    if log and not pre_logged:
        a = np.log10(np.where(a > 0, a, np.nan))
    if keep is not None:
        a = np.where(np.asarray(keep, bool), a, np.nan)
    finite = a[np.isfinite(a)]
    if vmin is None:
        vmin = float(np.percentile(finite, 1)) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=(bad_color if bad_color is not None else (0, 0, 0, 0)))   # solid fill, else transparent
    fig, ax = plt.subplots(figsize=(4, 4))
    if not transparent and bad_color is not None:
        fig.patch.set_facecolor(bad_color); ax.set_facecolor(bad_color)
    ax.imshow(a, origin="lower", cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    paths = []
    for ext in formats:
        p = str(Path(f"{out_base}.{str(ext).lstrip('.')}"))
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=transparent)
        paths.append(p)
    logger.info("  wrote %s  [vmin=%.4g vmax=%.4g]", Path(paths[0]).name, vmin, vmax)
    return float(vmin), float(vmax)


def _resolve_dataset_dir(cfg, root) -> Path:
    """Explicit dataset_dir (relative to the spec) wins; else derive from a benchmark spec (+ optional experiment)."""
    if cfg.get("dataset_dir"):
        return _resolve_path(cfg["dataset_dir"], root)
    bspec_path = _resolve_path(cfg["benchmark_spec"], root)
    bspec = load_yaml(bspec_path).get("benchmark", {})
    fixed = bspec.get("fixed", {}) or {}
    ov = {}
    if cfg.get("experiment"):
        ex = next((e for e in bspec.get("experiments", []) if e.get("name") == cfg["experiment"]), None)
        if ex is None:
            raise ValueError(f"experiment {cfg['experiment']!r} not in {bspec_path.name}")
        ov = ex.get("overrides", {}) or {}
    dd = ov.get("dataset_dir", fixed.get("dataset_dir"))
    return _resolve_path(dd, bspec_path.parent)                   # dataset paths anchor to the benchmark cfg


# ---------------------------------------------------------------- [1] channels (dataset viewpoints)
def generate_channel_images(spec_path, device_override=None, output_dir_override=None):
    cfg, root = _cfg(spec_path, "channels")
    out = _outdir(cfg, root, output_dir_override, "channels")
    dataset_dir = _resolve_dataset_dir(cfg, root)
    channels = [int(g) for g in cfg.get("channels", [0, 1, 2, 3])]
    views = [int(v) for v in cfg.get("views", [0, 750])]           # dataset viewpoint indices (angles)
    apply_mask = bool(cfg.get("mask", True))
    dpi = int(cfg.get("dpi", 300)); formats = cfg.get("formats", ["png"])
    styles = cfg.get("channel_styles", {}) or {}
    def _style(g): return styles.get(g, styles.get(str(g), {})) or {}

    mask = np.load(Path(dataset_dir) / "mask.npy").astype(bool)
    lims, names = _stats_limits(dataset_dir, channels)             # per-channel log10 color bounds + names
    logger.info("channels: dataset=%s  views=%s  channels=%s  mask=%s  -> %s",
                dataset_dir, views, channels, apply_mask, out)
    for v in views:
        z = np.load(Path(dataset_dir) / "viewpoints_gt" / f"vp_{v}.npz")
        I = np.asarray(z["intensities"], float)                    # (H, W, C_all)
        for i, g in enumerate(channels):
            st = _style(g)
            disp = np.log10(np.clip(I[:, :, g], 1e-12, None))      # already log10
            _save_bare(disp, out / f"view{v}_ch{g}",
                       cmap=st.get("cmap", cfg.get("cmap", "inferno")),
                       vmin=st.get("vmin", lims[i][0]), vmax=st.get("vmax", lims[i][1]),
                       log=False, pre_logged=True,
                       keep=(mask if apply_mask else None), dpi=dpi, formats=formats,
                       transparent=bool(cfg.get("transparent", False)),
                       bad_color=cfg.get("bad_color", "black"))
    return {"output_dir": str(out)}


# ---------------------------------------------------------------- shared LOS render ([2],[4])
def _render_los_images(cfg, root, out, *, use_gt, default_quantities, default_mask_modes, device):
    bdir = _resolve_path(cfg["benchmark_dir"], root)
    summaries = collect_run_summaries(bdir)
    sel = {k: cfg[k] for k in ("experiment", "where", "run_names", "min_seeds") if k in cfg}
    sel.setdefault("min_seeds", 1)
    run_dir = _resolve_run_dir(_collect_condition_seed_rows(summaries, sel)[0], bdir)
    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=cfg.get("path_remap"))
    try:
        rp = dict(cfg.get("render", {}) or {})
        fov = float(rp.get("fov_rsun", 2.5))
        n_uniform = int(rp.get("uniform_samples", 512))
        integ_max = rp.get("integ_r_max_rsun")
        if integ_max is None and n_uniform > 0:
            integ_max = fov
        rkw = dict(lat=float(cfg.get("lat", 0.0)), obs_r=float(rp.get("obs_r", 215.0)),
                   H=int(rp.get("H", 512)), W=int(rp.get("W", 512)), fov_rsun=fov,
                   chunk_rays=int(rp.get("chunk_rays", 8192)),
                   r_min_rsun=float(rp.get("integ_r_min_rsun", 1.0)),
                   r_max_rsun=integ_max, uniform_samples=n_uniform)
        lons = cfg.get("lons") or [float(cfg.get("lon", 0.0))]
        quantities = cfg.get("quantities") or default_quantities
        mask_modes = cfg.get("mask_modes", default_mask_modes)
        mcfg = dict(cfg.get("mask", {}) or {})
        dpi = int(cfg.get("dpi", 300)); formats = cfg.get("formats", ["png"])
        logger.info("render (%s): run=%s  lons=%s  quantities=%s  mask_modes=%s  -> %s",
                    "GT" if use_gt else "pred", run_dir.name, lons, quantities, mask_modes, out)
        for lon in lons:
            for q in quantities:
                qd = _QDEF.get(q, {"cmap": "viridis", "log": False})
                m = render_los_map(state, q, lon=float(lon), use_gt=use_gt, **rkw)
                for mode in mask_modes:
                    keep = None
                    if mode == "masked":
                        r_min2d = float(mcfg.get("r_min_rsun", 0.0))   # 0 = do NOT occult the disk (matches los_filmstrip)
                        r_max2d = mcfg.get("r_max_rsun")               # set this to crop the outer corona to a clean disk
                        keep = make_coronal_mask_simple(m.shape[0], m.shape[1], fov_rsun=fov,
                                    r_min_rsun=r_min2d,
                                    r_max_rsun=None if r_max2d is None else float(r_max2d))
                    _save_bare(m, out / f"{q}_lon{int(round(float(lon)))}_{mode}",
                               cmap=cfg.get("cmap", qd["cmap"]), vmin=cfg.get("vmin"), vmax=cfg.get("vmax"),
                               log=bool(cfg.get("log", qd["log"])), keep=keep, dpi=dpi, formats=formats)
    finally:
        _release_state(state)
    return {"output_dir": str(out)}


def generate_corona_render(spec_path, device_override=None, output_dir_override=None):
    cfg, root = _cfg(spec_path, "corona")
    out = _outdir(cfg, root, output_dir_override, "corona")
    return _render_los_images(cfg, root, out, use_gt=True,
                              default_quantities=["emission"],
                              default_mask_modes=["unmasked", "masked"],
                              device=device_override if device_override is not None else cfg.get("device"))


def generate_field_render(spec_path, device_override=None, output_dir_override=None):
    cfg, root = _cfg(spec_path, "field")
    out = _outdir(cfg, root, output_dir_override, "field")
    return _render_los_images(cfg, root, out, use_gt=False,
                              default_quantities=["column_ne", "ewt_temp"],
                              default_mask_modes=["unmasked"],
                              device=device_override if device_override is not None else cfg.get("device"))


_KIND = {"channels": generate_channel_images, "corona": generate_corona_render, "field": generate_field_render}


def main():
    ap = argparse.ArgumentParser(description="Generate raw diagram image assets (separate PNGs for a hand-built figure).")
    ap.add_argument("--kind", required=True, choices=list(_KIND))
    ap.add_argument("--spec", required=True, help="bare name under configs/diagram/, or a path.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("diagram[%s]  spec -> %s", args.kind, _resolve_spec(args.spec))
    _KIND[args.kind](args.spec, device_override=args.device, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()