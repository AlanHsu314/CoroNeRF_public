from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Union
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ..benchmark.config import load_yaml, save_json
from ._aggregate_utils import load_grouped_table, lookup_one, resolve_metric_column
from .shell_panels import _resolve_path

logger = logging.getLogger("coroNeRF.paper.empirical_analytic")

'''
quick 1 by 3 plot of heatmaps, 1 by 2 empirical, last plot is analytical
'''

def _fmt(v, small="{:.3f}", large="{:.2f}"):
    if not np.isfinite(v): return "—"
    elif int(v) == v: return int(v)
    return small.format(v) if abs(v) < 0.1 else large.format(v)

def _draw(ax, Z, xlabels, ylabels, xlabel, ylabel, title, cmap, vmin, vmax, pc):
    Zp = np.clip(Z, vmin, vmax) if pc.get("clip", True) else Z
    im = ax.imshow(Zp, origin="lower", aspect=pc.get("aspect", "auto"), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels([str(x) for x in xlabels])
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels([str(y) for y in ylabels])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if pc.get("annotate", True):
        for iy in range(Z.shape[0]):
            for ix in range(Z.shape[1]):
                ax.text(ix, iy, _fmt(Z[iy, ix], pc.get("annot_fmt_small", "{:.3f}"), pc.get("annot_fmt_large", "{:.2f}")),
                        ha="center", va="center", color=pc.get("annot_color", "white"), fontsize=pc.get("annot_font_size", 7))
    ax.figure.colorbar(im, ax=ax, fraction=pc.get("colorbar_fraction", 0.046), pad=pc.get("colorbar_pad", 0.04))

def generate_empirical_vs_analytic(spec_path, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve(); root = spec_path.parent
    cfg = load_yaml(spec_path).get("figure", {})
    name = str(cfg.get("name", spec_path.stem))
    out_dir = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg.get("output_dir", "../../paper_outputs/figures"), root))
    out_dir.mkdir(parents=True, exist_ok=True)
    x_cfg = dict(cfg["x"]); y_cfg = dict(cfg["y"]); pc = dict(cfg.get("plot", {}))
    xk, yk = x_cfg["key"], y_cfg["key"]
    xvals = list(x_cfg["values"]); yvals = list(y_cfg["values"])
    xlab = x_cfg.get("labels", xvals); ylab = y_cfg.get("labels", yvals)

    # --- empirical panels (grouped csv) ---
    emp = dict(cfg["empirical"]); df = load_grouped_table(_resolve_path(emp["grouped_csv"], root))
    emp_panels = []
    for p in emp["panels"]:
        col = resolve_metric_column(df, str(p["metric"]), stat=str(p.get("stat", "mean")))
        Z = np.full((len(yvals), len(xvals)), np.nan)
        for iy, yv in enumerate(yvals):
            for ix, xv in enumerate(xvals):
                try: Z[iy, ix] = float(lookup_one(df, {xk: xv, yk: yv})[col])
                except Exception as e: logger.warning("empirical (%s,%s) missing: %s", yv, xv, e)
        emp_panels.append((Z, dict(p)))

    # --- analytic panel (operating-point sweep manifest) ---
    an = dict(cfg["analytic"])
    rows = json.load(open(_resolve_path(an["sweep_manifest"], root)))["rows"]
    metric = str(an.get("metric", "post_std_median")); etas = [float(e) for e in an["eta_values"]]  # row-aligned to y.values
    Za = np.full((len(yvals), len(xvals)), np.nan)
    for iy, eta in enumerate(etas):
        for ix, xv in enumerate(xvals):
            r = next((r for r in rows if int(r.get("views", -1)) == int(xv) and abs(float(r.get("eta", -1)) - eta) < 1e-9), None)
            if r is not None and r.get(metric) is not None: Za[iy, ix] = float(r[metric])

    with plt.rc_context({"font.size": pc.get("font_size", 8), "axes.titlesize": pc.get("title_font_size", 9)}):
        fig, axs = plt.subplots(1, 3, figsize=tuple(pc.get("figsize", [10.5, 3.0])), constrained_layout=True)
        for k, (Z, p) in enumerate(emp_panels):
            vmax = p.get("vmax", float(np.nanpercentile(Z, 99))); vmin = p.get("vmin", 0.0)
            _draw(axs[k], Z, xlab, ylab, x_cfg.get("label", xk), (y_cfg.get("label", yk) if k == 0 else ""),
                  p.get("title", p["metric"]), p.get("cmap", pc.get("cmap", "magma_r")), vmin, vmax, pc)
        vmax_a = an.get("vmax", float(np.nanpercentile(Za, 99))); vmin_a = an.get("vmin", float(np.nanmin(Za)))
        _draw(axs[2], Za, xlab, ylab, x_cfg.get("label", xk), "", an.get("title", "posterior median (analytic)"),
              an.get("cmap", "viridis"), vmin_a, vmax_a, pc)
        paths = {}
        for ext in pc.get("formats", ["png", "pdf"]):
            pth = out_dir / f"{name}.{ext}"; fig.savefig(pth, dpi=pc.get("dpi", 300), bbox_inches="tight"); paths[ext] = str(pth)
        plt.close(fig)
    save_json({"name": name, "outputs": paths}, out_dir / f"{name}_manifest.json")
    logger.info("empirical-vs-analytic 1x3 -> %s", out_dir); return paths