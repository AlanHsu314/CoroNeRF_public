#####################################
### filmstrip of LOS renders — flexible multi-row (quantity x source x condition)
#####################################
from __future__ import annotations
import json
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from ..benchmark.config import load_yaml
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.select import _collect_condition_seed_rows
from ..benchmark.condition_compare import _build_state_for_analysis, _resolve_run_dir, _release_state
from ..util.figure import _resolve_path, _savefig
from ..visualization.los_render import render_los_map
from ..visualization.geometry import make_coronal_mask_simple

# per-quantity display defaults
_QDEF = {
    "column_ne": {"cmap": "inferno", "label": r"column $n_{\rm e}$",     "log": False},
    "ewt_temp":  {"cmap": "viridis", "label": r"$\epsilon$-weighted $T_{\rm e}$", "log": False},
    "emission":  {"cmap": "magma",   "label": r"emission",         "log": True},
}
_SELK = ("experiment", "where", "run_names", "filters", "min_seeds")  # keys passed to run selection

def _emission_group(quantity, channels):
    """Per-line colour group so each emission channel (and the total) gets its OWN colour scale + colorbar.
    Non-emission quantities keep grouping by quantity. Keys: 'emission_ch0', 'emission_ch0_2', 'emission_total'."""
    if str(quantity) != "emission":
        return str(quantity)
    if channels is None:
        return "emission_total"
    cs = channels if isinstance(channels, (list, tuple)) else [channels]
    return "emission_ch" + "_".join(str(int(c)) for c in cs)

def _color_limits(vals, ccfg):
    ccfg = ccfg or {}
    if ccfg.get("vmin") is not None and ccfg.get("vmax") is not None:
        return float(ccfg["vmin"]), float(ccfg["vmax"])
    pct = ccfg.get("percentile", [1.0, 99.0])
    if np.isscalar(pct): pct = [100.0 - float(pct), float(pct)]
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if v.size == 0: return 0.0, 1.0
    return float(np.percentile(v, pct[0])), float(np.percentile(v, pct[1]))


def _row_specs(cfg):
    '''Expand config into an explicit ordered list of row dicts. Uses cfg["rows"] if present
       (fully general); else the shorthand source(gt|pred|both) x quantities.'''
    top_bdir = cfg["benchmark_dir"]
    top_run  = dict(cfg.get("run", {}) or {})
    top_src  = str(cfg.get("source", "pred")).lower()

    def mkrow(quantity, source, *, benchmark_dir=None, run=None, where=None,
            label=None, base_label=None, cmap=None, log=None, color=None, color_group=None,
            channels=None):
        q = str(quantity)
        d = _QDEF.get(q, {"cmap": "viridis", "label": q, "log": False})
        sel = dict(top_run)
        if run:  sel.update(run)
        if where is not None: sel["where"] = where
        sel = {k: v for k, v in sel.items() if k in _SELK}
        return {"quantity": q, "source": str(source).lower(),
                "benchmark_dir": benchmark_dir or top_bdir, "run": sel,
                "label": label, "base_label": base_label, "cmap": cmap or d["cmap"],
                "log": bool(d["log"] if log is None else log),
                "color": color or {}, "color_group": color_group or _emission_group(q, channels),
                "channels": channels}

    if cfg.get("rows"):
        rows = [mkrow(r["quantity"], r.get("source", top_src),
                      benchmark_dir=r.get("benchmark_dir"), run=r.get("run"), where=r.get("where"),
                      label=r.get("label"), base_label=r.get("base_label"), cmap=r.get("cmap"), log=r.get("log"),
                      color=r.get("color"), color_group=r.get("color_group"), channels=r.get("channels"))
                for r in cfg["rows"]]
    else:
        quantities = cfg.get("quantities") or [{"quantity": "column_ne"}, {"quantity": "ewt_temp"}]
        srcs = ["gt", "pred"] if top_src == "both" else [top_src]
        rows = []
        for qc in quantities:
            qc = qc if isinstance(qc, dict) else {"quantity": qc}
            for s in srcs:
                rows.append(mkrow(qc["quantity"], s, base_label=qc.get("label"), cmap=qc.get("cmap"),
                                  log=qc.get("log"), color=qc.get("color"),
                                  color_group=qc.get("color_group"), channels=qc.get("channels")))

    # row labels: explicit `label` wins; else base (custom `label`/`base_label` or quantity default),
    # with the source tag appended automatically when a quantity appears as BOTH gt and pred.
    src_labels = {"gt": "GT", "pred": "pred", **(cfg.get("source_labels") or {})}
    fmt = cfg.get("source_label_fmt", "{base} ({src})")
    for r in rows:
        if r["label"] is not None:
            continue
        base = r.get("base_label") or _QDEF.get(r["quantity"], {}).get("label", r["quantity"])
        srcs_q = {rr["source"] for rr in rows
                  if rr["quantity"] == r["quantity"] and rr["benchmark_dir"] == r["benchmark_dir"]}
        r["label"] = fmt.format(base=base, src=src_labels.get(r["source"], r["source"])) if len(srcs_q) > 1 else base
    return rows


def generate_los_filmstrip(spec_path, device_override=None, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve(); root = spec_path.parent
    cfg = load_yaml(spec_path).get("figure", {})
    plot = dict(cfg.get("plot", {}) or {})
    device = device_override if device_override is not None else cfg.get("device")

    # shared geometry / render params (columns = viewpoints)
    n = int(cfg.get("n_cols", 5))
    lat = float(cfg.get("lat", 0.0))
    lon0, lon1 = cfg.get("lon_range", [0.0, 360.0])
    lons = np.linspace(float(lon0), float(lon1), n, endpoint=bool(cfg.get("lon_endpoint", False)))
    rp = dict(cfg.get("render", {}) or {})
    fov = float(rp.get("fov_rsun", 2.5))
    integ_r_min = rp.get("integ_r_min_rsun", 1.0)
    integ_r_max = rp.get("integ_r_max_rsun", None)
    n_uniform = int(rp.get("uniform_samples", 512))
    if n_uniform > 0 and integ_r_max is None:
        integ_r_max = float(fov)
    rkw = dict(lat=lat, obs_r=float(rp.get("obs_r", 215.0)), H=int(rp.get("H", 512)),
               W=int(rp.get("W", 512)), fov_rsun=fov, chunk_rays=int(rp.get("chunk_rays", 8192)),
               r_min_rsun=integ_r_min, r_max_rsun=integ_r_max, uniform_samples=n_uniform)

    rows = _row_specs(cfg)

    # group rows by (benchmark_dir, run selection) so each unique run's state is built once
    def selkey(r): return (str(r["benchmark_dir"]), json.dumps(r["run"], sort_keys=True, default=str))
    order = {}
    for ri, r in enumerate(rows):
        order.setdefault(selkey(r), []).append(ri)

    summ = {}
    def summaries(bdir):
        k = str(bdir)
        if k not in summ: summ[k] = collect_run_summaries(bdir)
        return summ[k]

    rendered = [None] * len(rows)
    for _key, ris in order.items():
        bdir = _resolve_path(rows[ris[0]]["benchmark_dir"], root)
        sel = dict(rows[ris[0]]["run"]); sel.setdefault("min_seeds", 1)
        run_dir = _resolve_run_dir(_collect_condition_seed_rows(summaries(bdir), sel)[0], bdir)
        state = _build_state_for_analysis(run_dir, device_override=device, path_remap=cfg.get("path_remap"))
        try:
            for ri in ris:
                use_gt = (rows[ri]["source"] == "gt")
                rendered[ri] = [render_los_map(state, rows[ri]["quantity"], lon=float(L),
                                               use_gt=use_gt, channels=rows[ri].get("channels"), **rkw) for L in lons]
        finally:
            _release_state(state)

    # shared 2D projected pixel crop (optional)
    mcfg = dict(cfg.get("mask", {}) or {})
    r_min2d = float(mcfg.get("r_min_rsun", 0.0)); r_max2d = mcfg.get("r_max_rsun")
    if mcfg.get("enabled", (r_min2d > 0.0) or (r_max2d is not None)):
        H, W = rendered[0][0].shape
        keep = make_coronal_mask_simple(H, W, fov_rsun=fov, r_min_rsun=r_min2d,
                                        r_max_rsun=None if r_max2d is None else float(r_max2d))
        rendered = [[np.where(keep, m, np.nan) for m in maps] for maps in rendered]

    # display transform (log) + shared colour limits per colour_group
    disp = [([np.log10(np.where(m > 0, m, np.nan)) for m in rendered[ri]] if rows[ri]["log"]
             else list(rendered[ri])) for ri in range(len(rows))]
    groups = {}
    for ri, r in enumerate(rows):
        groups.setdefault(r["color_group"], []).append(ri)

    # fixed limits per colour_group -> cross-figure comparison
    clim_cfg = cfg.get("color_limits", {}) or {}  
    glim = {g: _color_limits(np.concatenate([disp[ri][ci].ravel() for ri in ris for ci in range(n)]),
                             clim_cfg.get(g, rows[ris[0]]["color"]))
            for g, ris in groups.items()}

    with plt.rc_context({"font.size": plot.get("font_size", 8), "axes.titlesize": plot.get("title_font_size", 9)}):
        R = len(rows)
        fig, axs = plt.subplots(R, n, figsize=tuple(plot.get("figsize", [2.0 * n, 2.2 * R])),
                                squeeze=False, constrained_layout=True)
        fig.set_constrained_layout_pads(
            wspace=plot.get("wspace", 0.02), hspace=plot.get("hspace", 0.04),
            w_pad=plot.get("w_pad", 0.02),   h_pad=plot.get("h_pad", 0.02),
        )
        last_of = {g: ris[-1] for g, ris in groups.items()}
        for ri, r in enumerate(rows):
            vmin, vmax = glim[r["color_group"]]
            cmap = plt.get_cmap(r["cmap"]).copy(); cmap.set_bad(plot.get("bad_color", "black"))
            im = None
            for ci in range(n):
                ax = axs[ri][ci]
                im = ax.imshow(disp[ri][ci], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                               aspect=plot.get("aspect", "equal"),
                               interpolation=plot.get("interpolation", "nearest"))
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0: ax.set_title(f"{int(round(lons[ci]))}°")
            axs[ri][0].set_ylabel(r["label"])
            if plot.get("colorbar", True) and im is not None and ri == last_of[r["color_group"]]:
                gris = groups[r["color_group"]]
                contiguous = (gris == list(range(gris[0], gris[-1] + 1)))
                cbar_axes = [axs[rr][c] for rr in (gris if contiguous else [ri]) for c in range(n)]
                lab = (r.get("label") or _QDEF.get(r["quantity"], {}).get("label", r["quantity"])) + (r"  [$\log_{10}$]" if r["log"] else "")
                nrow_span = len(gris) if contiguous else 1
                cb = fig.colorbar(im, ax=cbar_axes,
                                  fraction=plot.get("colorbar_fraction", 0.012),
                                  pad=plot.get("colorbar_pad", 0.01),
                                  aspect=plot.get("colorbar_aspect", 20) * nrow_span)
                if plot.get("cb_label"): cb.set_label(lab, fontsize=plot.get("tick_font_size", 7)); cb.ax.tick_params(labelsize=plot.get("tick_font_size", 7))
        if plot.get("title"): fig.suptitle(plot["title"], fontsize=plot.get("title_font_size", 9))
        out = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg.get("output_dir", "../../paper_outputs/figures/misc"), root))
        out.mkdir(parents=True, exist_ok=True)
        paths = _savefig(fig, out / cfg.get("name", "los_filmstrip"),
                         plot.get("formats", cfg.get("formats", ["pdf", "png"])), int(plot.get("dpi", 300)))
    plt.close(fig)
    return {"paths": paths}