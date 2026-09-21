##########################################################
### Disagreement diagnostics [a], [b], [c]
###
### [a] generate_disagreement_radial_figure
###     correlation(disagreement, |error|) as a function of shell radius
###     (the radial sweep of Paper B, but plotting correlation not MAE)
###
### [b] generate_disagreement_decomp_figure
###     variance / bias decomposition: maps, variance-fraction VF(r),
###     and the corr-vs-VF link (the non-circular "ceiling" check)
###
### [c] generate_disagreement_conditions_figure
###     one summary correlation per condition across benchmarks
###     (Paper-E style discrete x-axis) + corr-vs-VF scatter
###
### Reuses the ensemble machinery in coronerf/identifiability/seed_disagreement.py
##########################################################

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..util.figure import (
    _resolve_path, _longitude_order, _maybe_suptitle, _correlations,
    _cmap_with_bad, _reorder_cols, _savefig, _qty_tex, diagnostics_dir,
)

from ..benchmark.select import _collect_condition_seed_rows
from .seed_disagreement import (
    _accumulate_ensemble, _reduce_field, _reduce_image, _apply_display, _nearest_radius_key,
)

logger = logging.getLogger("coroNeRF.identifiability.disagreement_diagnostics")

__all__ = [
    "generate_disagreement_radial_figure",
    "generate_disagreement_decomp_figure",
    "generate_disagreement_conditions_figure",
    "generate_radial_image_figure",
]

_DEFAULT_RADII = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0, 3.5, 4.0]


# ----------------------------------------------------------------------
# shared front-end: one condition -> reduced ensemble
# ----------------------------------------------------------------------

def _default_diag_out(cfg, kind, spec_root):
    conds = cfg.get("conditions") or ([cfg["condition"]] if cfg.get("condition") else [])
    bdir = conds[0].get("benchmark_dir") if conds else None
    return diagnostics_dir(_resolve_path(bdir, spec_root), kind) if bdir else Path.cwd()

# (_maybe_suptitle now imported from ..util.figure)

def _f(x) -> float:
    """None/non-finite -> nan."""
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _load_condition_reduced(
    cond: dict,
    *,
    quantities: list[str],
    radii: list[float],
    error_mode: str,
    disagreement: str,
    device: str | None,
    default_path_remap: dict | None,
    spec_root: Path,
) -> tuple[dict, list[dict], Path]:
    """Resolve a condition's benchmark, collect its seeds, accumulate + reduce (fields only)."""
    bdir = _resolve_path(cond["benchmark_dir"], spec_root)
    path_remap = cond.get("path_remap", default_path_remap)

    rows = collect_run_summaries(bdir)
    sel = {k: cond[k] for k in
           ("experiment", "where", "filters", "run_names", "run_dirs", "min_seeds", "max_seeds")
           if k in cond}
    seed_rows = _collect_condition_seed_rows(rows, sel)

    field, _img = _accumulate_ensemble(
        seed_rows=seed_rows,
        benchmark_dir=bdir,
        quantities=quantities,
        radii=radii,
        image_cfg={"enabled": False},   # diagnostics [a]/[b]/[c] need fields only
        device=device,
        path_remap=path_remap,
    )
    reduced = _reduce_field(field, error_mode=error_mode, disagreement=disagreement)
    return reduced, seed_rows, bdir


def _cond_label(cond: dict) -> str:
    return str(cond.get("label", cond.get("experiment", "condition")))


def _condition_colors(conditions: list[dict]) -> list[Any]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])
    out = []
    for i, c in enumerate(conditions):
        out.append(c.get("color") or cycle[i % len(cycle)])
    return out


# ----------------------------------------------------------------------
# shared computations
# ----------------------------------------------------------------------

def _pool_pair(reduced_q: dict) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _r, d in reduced_q.items():
        x = np.asarray(d["disag"]).reshape(-1)
        y = np.asarray(d["true_err"]).reshape(-1)
        ok = np.isfinite(x) & np.isfinite(y)
        xs.append(x[ok]); ys.append(y[ok])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def _per_shell_corr(reduced: dict) -> dict:
    """{q: {r:[...], spearman:[...], pearson:[...], n:[...]}}."""
    out = {}
    for q, by_r in reduced.items():
        rs = sorted(by_r.keys())
        sp, pe, ns = [], [], []
        for r in rs:
            d = by_r[r]
            x = np.asarray(d["disag"]).reshape(-1)
            y = np.asarray(d["true_err"]).reshape(-1)
            ok = np.isfinite(x) & np.isfinite(y)
            c = _correlations(x[ok], y[ok])
            sp.append(_f(c["spearman"])); pe.append(_f(c["pearson"])); ns.append(int(c["n"]))
        out[q] = {"r": np.asarray(rs, float),
                  "spearman": np.asarray(sp, float),
                  "pearson": np.asarray(pe, float),
                  "n": np.asarray(ns, int)}
    return out


def _decompose(reduced: dict) -> dict:
    """
    Per (quantity, shell) bias/variance decomposition computed from gt + ensemble mean
    (independent of error_mode, so it never depends on the quantity it explains).
      variance  = sigma^2                          (GT-free)
      bias^2    = (mean - GT)^2 - sigma^2 / S       (bias-corrected; needs GT)
      mse       = bias^2 + variance
      VF        = variance / (variance + bias^2)    (per shell, from band means)
    """
    out = {}
    for q, by_r in reduced.items():
        rs = sorted(by_r.keys())
        rows = {}
        for r in rs:
            d = by_r[r]
            gt = np.asarray(d["gt"], float)
            mu = np.asarray(d["ens_mean"], float)
            sigma = np.asarray(d["disag"], float)
            S = max(int(d["n_seeds"]), 1)

            var = sigma ** 2
            eps = mu - gt
            bias2 = np.clip(eps ** 2 - var / S, 0.0, None)
            mse = bias2 + var

            mvar = float(np.nanmean(var))
            mbias2 = float(np.nanmean(bias2))
            vf_shell = mvar / (mvar + mbias2 + 1e-30)

            x = sigma.reshape(-1); y = np.abs(eps).reshape(-1)
            ok = np.isfinite(x) & np.isfinite(y)
            corr = _correlations(x[ok], y[ok])

            rows[r] = {
                "sigma": sigma, "bias": np.sqrt(bias2), "rmse": np.sqrt(mse),
                "var": var, "bias2": bias2, "mse": mse, "eps": eps,
                "vf_shell": vf_shell, "corr_spearman": _f(corr["spearman"]),
                "lon": np.asarray(d["lon"], float), "lat": np.asarray(d["lat"], float),
            }
        out[q] = {"r": np.asarray(rs, float), "rows": rows}
    return out


def _summary_corr_vf(reduced: dict, band: tuple[float, float] | None,
                     n_boot: int, max_points: int, rng: np.random.Generator) -> dict:
    """Pooled correlation + variance-fraction over a radial band, per quantity, with bootstrap CI."""
    out = {}
    for q, by_r in reduced.items():
        xs, ys, mvar, mbias2 = [], [], [], []
        for r, d in by_r.items():
            if band is not None and not (band[0] <= float(r) <= band[1]):
                continue
            sigma = np.asarray(d["disag"], float)
            eps = np.asarray(d["ens_mean"], float) - np.asarray(d["gt"], float)
            S = max(int(d["n_seeds"]), 1)
            x = sigma.reshape(-1); y = np.abs(eps).reshape(-1)
            ok = np.isfinite(x) & np.isfinite(y)
            xs.append(x[ok]); ys.append(y[ok])
            var = sigma ** 2
            mvar.append(float(np.nanmean(var)))
            mbias2.append(float(np.nanmean(np.clip(eps ** 2 - var / S, 0.0, None))))
        if not xs:
            continue
        X = np.concatenate(xs); Y = np.concatenate(ys)
        base = _correlations(X, Y)

        boots = []
        if X.size > 2:
            n = min(int(max_points), X.size)
            for _ in range(int(n_boot)):
                idx = rng.integers(0, X.size, n)
                boots.append(_f(_correlations(X[idx], Y[idx])["spearman"]))
        boots = np.asarray(boots, float)
        ci_lo = float(np.nanpercentile(boots, 2.5)) if boots.size else float("nan")
        ci_hi = float(np.nanpercentile(boots, 97.5)) if boots.size else float("nan")

        vf = float(np.mean(mvar) / (np.mean(mvar) + np.mean(mbias2) + 1e-30))
        out[q] = {"spearman": _f(base["spearman"]), "pearson": _f(base["pearson"]),
                  "ci_lo": ci_lo, "ci_hi": ci_hi, "vf": vf, "n": int(X.size)}
    return out


def _present_quantities(order: list[str], per_cond: list[dict]) -> list[str]:
    return [q for q in order if any(q in pc for pc in per_cond)]


# ----------------------------------------------------------------------
# [a] correlation vs radius
# ----------------------------------------------------------------------

def generate_disagreement_radial_figure(
    spec_path: Union[str, Path],
    device_override: str | None = None,
    output_dir_override: Union[str, Path, None] = None,
) -> dict:
    spec_path = Path(spec_path).resolve()
    spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("figure", {})

    name = str(cfg.get("name", spec_path.stem))
    out_dir = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg["output_dir"], spec_root) if cfg.get("output_dir")
               else _default_diag_out(cfg, "decomp", spec_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_override if device_override is not None else cfg.get("device", None)

    quantities = [str(q) for q in cfg.get("quantities", ["ne", "temp"])]
    radii = [float(r) for r in cfg.get("radii", _DEFAULT_RADII)]
    error_mode = str(cfg.get("error_mode", "ensemble_mean_abs"))
    disagreement = str(cfg.get("disagreement", "std"))
    metric = str(cfg.get("correlation", "spearman"))
    show_secondary = bool(cfg.get("show_secondary_metric", True))
    default_path_remap = cfg.get("path_remap", None)
    plot_cfg = dict(cfg.get("plot", {}) or {})
    conditions = list(cfg.get("conditions", []) or [])
    if not conditions:
        raise ValueError("[a] requires at least one entry under figure.conditions")

    colors = _condition_colors(conditions)
    results = []
    for cond, color in zip(conditions, colors):
        reduced, seed_rows, _ = _load_condition_reduced(
            cond, quantities=quantities, radii=radii, error_mode=error_mode,
            disagreement=disagreement, device=device,
            default_path_remap=default_path_remap, spec_root=spec_root)
        psc = _per_shell_corr(reduced)
        pooled = {q: _correlations(*_pool_pair(reduced[q])) for q in reduced}
        results.append({"label": _cond_label(cond), "color": color,
                        "psc": psc, "pooled": pooled, "n_seeds": len(seed_rows)})

    qlist = _present_quantities(quantities, [r["psc"] for r in results])
    formats = list(plot_cfg.get("formats", ["png", "pdf"]))
    dpi = int(plot_cfg.get("dpi", 300))
    figsize = plot_cfg.get("figsize", [5.5 * max(len(qlist), 1), 4.5])
    other = "pearson" if metric == "spearman" else "spearman"

    fig, axs = plt.subplots(1, len(qlist), figsize=figsize, squeeze=False)
    axs = axs[0]
    for ax, q in zip(axs, qlist):
        for res in results:
            if q not in res["psc"]:
                continue
            cur = res["psc"][q]
            line, = ax.plot(cur["r"], cur[metric], marker="o",
                            label=f"{res['label']} (S={res['n_seeds']})", color=res["color"])
            if show_secondary:
                ax.plot(cur["r"], cur[other], marker=".", ls="--", alpha=0.55, color=line.get_color())
            if plot_cfg.get("show_pooled_hline", True):
                pv = _f(res["pooled"][q][metric])
                if np.isfinite(pv):
                    ax.axhline(pv, ls=":", alpha=0.5, color=line.get_color())
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel(plot_cfg.get("xlabel", "Radius [$R_\\odot$]"))
        ax.set_ylabel(plot_cfg.get("ylabel", f"{metric} corr( disagreement , |error| )"))
        ax.set_title(q)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    if plot_cfg.get("title"):
        fig.suptitle(str(plot_cfg["title"]))
    fig.tight_layout()
    out_paths = _savefig(fig, out_dir / name, formats, dpi)
    plt.close(fig)

    manifest = {
        "name": name, "kind": "radial", "metric": metric,
        "quantities": qlist, "radii": radii, "out_paths": out_paths,
        "conditions": [
            {"label": r["label"], "n_seeds": r["n_seeds"],
             "per_shell": {q: {"r": r["psc"][q]["r"].tolist(),
                               "spearman": r["psc"][q]["spearman"].tolist(),
                               "pearson": r["psc"][q]["pearson"].tolist()} for q in r["psc"]},
             "pooled": {q: {"spearman": _f(r["pooled"][q]["spearman"]),
                            "pearson": _f(r["pooled"][q]["pearson"])} for q in r["pooled"]}}
            for r in results],
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("[a] saved %s", out_dir / f"{name}.png")
    return manifest


# ----------------------------------------------------------------------
# [b] variance / bias decomposition
# ----------------------------------------------------------------------

def generate_disagreement_decomp_figure(
    spec_path: Union[str, Path],
    device_override: str | None = None,
    output_dir_override: Union[str, Path, None] = None,
) -> dict:
    spec_path = Path(spec_path).resolve()
    spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("figure", {})

    name = str(cfg.get("name", spec_path.stem))
    out_dir = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg["output_dir"], spec_root) if cfg.get("output_dir")
               else _default_diag_out(cfg, "decomp", spec_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_override if device_override is not None else cfg.get("device", None)

    quantities = [str(q) for q in cfg.get("quantities", ["ne", "temp"])]
    radii = [float(r) for r in cfg.get("radii", _DEFAULT_RADII)]
    primary_radius = float(cfg.get("primary_radius", 1.5))
    error_mode = str(cfg.get("error_mode", "ensemble_mean_abs"))
    disagreement = str(cfg.get("disagreement", "std"))
    default_path_remap = cfg.get("path_remap", None)
    plot_cfg = dict(cfg.get("plot", {}) or {})
    lon_mode = str(plot_cfg.get("longitude_mode", "zero_360"))
    bad_color = str(plot_cfg.get("bad_color", "black"))
    formats = list(plot_cfg.get("formats", ["png", "pdf"]))
    dpi = int(plot_cfg.get("dpi", 300))
    conditions = list(cfg.get("conditions", []) or [])
    if not conditions:
        raise ValueError("[b] requires at least one entry under figure.conditions")

    colors = _condition_colors(conditions)
    per_cond = []
    for cond, color in zip(conditions, colors):
        reduced, seed_rows, _ = _load_condition_reduced(
            cond, quantities=quantities, radii=radii, error_mode=error_mode,
            disagreement=disagreement, device=device,
            default_path_remap=default_path_remap, spec_root=spec_root)
        per_cond.append({"label": _cond_label(cond), "color": color,
                         "dec": _decompose(reduced), "n_seeds": len(seed_rows)})

    qlist = _present_quantities(quantities, [pc["dec"] for pc in per_cond])

    # ---- (i) maps for the FIRST condition at primary radius ----
    seq_cmap = _cmap_with_bad(plot_cfg.get("map_cmap", "magma"), bad_color)
    map_cols = ["disagreement $\\sigma$", "|bias|", "RMSE"]
    map_keys = ["sigma", "bias", "rmse"]
    pc0 = per_cond[0]
    fig, axs = plt.subplots(len(qlist), 3, figsize=plot_cfg.get("maps_figsize", [11, 3.0 * len(qlist)]),
                            squeeze=False, layout="constrained")
    for ri, q in enumerate(qlist):
        dec = pc0["dec"][q]
        r_key = _nearest_radius_key(dec["rows"], primary_radius)
        row = dec["rows"][r_key]
        lon_plot, order = _longitude_order(row["lon"], lon_mode)
        extent = [float(np.nanmin(lon_plot)), float(np.nanmax(lon_plot)),
                  float(np.nanmin(row["lat"])), float(np.nanmax(row["lat"]))]
        stacked = np.concatenate([_reorder_cols(row[k], order).reshape(-1) for k in map_keys])
        vmax = float(np.nanpercentile(stacked[np.isfinite(stacked)], 99.0)) if np.isfinite(stacked).any() else 1.0
        im = None
        for ci, (k, lab) in enumerate(zip(map_keys, map_cols)):
            ax = axs[ri, ci]
            im = ax.imshow(_reorder_cols(row[k], order), origin="lower", extent=extent,
                           cmap=seq_cmap, vmin=0.0, vmax=max(vmax, 1e-9),
                           interpolation=plot_cfg.get("interpolation", "nearest"), aspect="auto")
            if ri == 0:
                ax.set_title(lab)
            if ci == 0:
                ax.set_ylabel(f"{_qty_tex(q, 'field')}\n(r={r_key:.2f})")
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axs[ri, :].tolist(), fraction=0.046, pad=0.02, label="dex")
    _maybe_suptitle(fig, plot_cfg.get("maps_title", f"{pc0['label']}: error decomposition"))
    maps_paths = _savefig(fig, out_dir / f"{name}_maps", formats, dpi)
    plt.close(fig)

    # ---- (ii) VF(r) radial curves (all conditions) ----
    fig, axs = plt.subplots(1, len(qlist), figsize=plot_cfg.get("vf_figsize", [5.5 * len(qlist), 4.5]),
                            squeeze=False)
    axs = axs[0]
    for ax, q in zip(axs, qlist):
        for pc in per_cond:
            if q not in pc["dec"]:
                continue
            dec = pc["dec"][q]
            vf = np.asarray([dec["rows"][r]["vf_shell"] for r in dec["r"]], float)
            ax.plot(dec["r"], vf, marker="o", color=pc["color"], label=f"{pc['label']} (S={pc['n_seeds']})")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(plot_cfg.get("xlabel", "Radius [$R_\\odot$]"))
        ax.set_ylabel("variance fraction VF")
        ax.set_title(q); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    vf_paths = _savefig(fig, out_dir / f"{name}_vf_radial", formats, dpi)
    plt.close(fig)

    # ---- (iii) corr vs VF scatter (points = shells) ----
    fig, axs = plt.subplots(1, len(qlist), figsize=plot_cfg.get("scatter_figsize", [5.5 * len(qlist), 4.5]),
                            squeeze=False)
    axs = axs[0]
    for ax, q in zip(axs, qlist):
        for pc in per_cond:
            if q not in pc["dec"]:
                continue
            dec = pc["dec"][q]
            vf = [dec["rows"][r]["vf_shell"] for r in dec["r"]]
            cc = [dec["rows"][r]["corr_spearman"] for r in dec["r"]]
            ax.scatter(vf, cc, color=pc["color"], label=pc["label"], s=28)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("variance fraction VF (per shell)")
        ax.set_ylabel("Spearman corr (per shell)")
        ax.set_title(q); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    _maybe_suptitle(fig, plot_cfg.get("scatter_title", "Per-shell localization $\\rho$ vs variance fraction (not 1:1)"))
    fig.tight_layout()
    scat_paths = _savefig(fig, out_dir / f"{name}_corr_vs_vf", formats, dpi)
    plt.close(fig)

    manifest = {
        "name": name, "kind": "decomp", "quantities": qlist,
        "primary_radius": primary_radius,
        "maps": maps_paths, "vf_radial": vf_paths, "corr_vs_vf": scat_paths,
        "conditions": [
            {"label": pc["label"], "n_seeds": pc["n_seeds"],
             "vf_shell": {q: {float(r): pc["dec"][q]["rows"][r]["vf_shell"] for r in pc["dec"][q]["r"]}
                          for q in pc["dec"]}}
            for pc in per_cond],
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("[b] saved decomposition figures under %s", out_dir)
    return manifest


# ----------------------------------------------------------------------
# [c] cross-condition summary
# ----------------------------------------------------------------------

def generate_disagreement_conditions_figure(spec_path, device_override=None, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve()
    spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("figure", {})

    name = str(cfg.get("name", spec_path.stem))
    out_dir = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg["output_dir"], spec_root) if cfg.get("output_dir")
               else _default_diag_out(cfg, "decomp", spec_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_override if device_override is not None else cfg.get("device", None)

    quantities = [str(q) for q in cfg.get("quantities", ["ne", "temp"])]
    radii = [float(r) for r in cfg.get("radii", _DEFAULT_RADII)]
    band = cfg.get("band", [1.1, 2.0]); band = (float(band[0]), float(band[1])) if band else None
    error_mode = str(cfg.get("error_mode", "ensemble_mean_abs"))
    disagreement = str(cfg.get("disagreement", "std"))
    default_path_remap = cfg.get("path_remap", None)
    n_boot = int(cfg.get("n_boot", 100))
    max_points = int(cfg.get("bootstrap_max_points", 20000))
    plot_cfg = dict(cfg.get("plot", {}) or {})
    formats = list(plot_cfg.get("formats", ["png", "pdf"]))
    dpi = int(plot_cfg.get("dpi", 300))
    legend_mode = str(plot_cfg.get("legend_mode", "last_axes"))   # global | last_axes | per_axes | none
    legend_loc = plot_cfg.get("legend_loc", "best")
    ylabel = plot_cfg.get("ylabel", r"Spearman $\rho$ (band-pooled)")
    conditions = list(cfg.get("conditions", []) or [])
    if not conditions:
        raise ValueError("[c] requires at least one entry under figure.conditions")

    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    families = []
    for c in conditions:
        fam = str(c.get("family", "other"))
        if fam not in families:
            families.append(fam)
    fam_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])
    fam_color = {fam: fam_cycle[i % len(fam_cycle)] for i, fam in enumerate(families)}

    entries = []
    for cond in conditions:
        reduced, seed_rows, _ = _load_condition_reduced(
            cond, quantities=quantities, radii=radii, error_mode=error_mode,
            disagreement=disagreement, device=device,
            default_path_remap=default_path_remap, spec_root=spec_root)
        sv = _summary_corr_vf(reduced, band, n_boot, max_points, rng)
        entries.append({"label": _cond_label(cond), "family": str(cond.get("family", "other")),
                        "summary": sv, "n_seeds": len(seed_rows)})

    qlist = _present_quantities(quantities, [e["summary"] for e in entries])
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=fam_color[f], label=f) for f in families]

    def _apply_legend(fig, axs):
        if legend_mode == "none" or not handles:
            return
        if legend_mode == "global":
            fig.legend(handles=handles, loc=plot_cfg.get("legend_loc", "upper right"), fontsize=7)
        elif legend_mode == "per_axes":
            for ax in axs:
                ax.legend(handles=handles, loc=legend_loc, fontsize=7)
        else:  # last_axes
            axs[-1].legend(handles=handles, loc=legend_loc, fontsize=7)

    # ---- main: spearman per condition (discrete x), grouped by family ----
    fig, axs = plt.subplots(1, len(qlist), figsize=plot_cfg.get("figsize", [6.5 * len(qlist), 4.5]), squeeze=False)
    axs = axs[0]
    for ax, q in zip(axs, qlist):
        xs, ys, los, his, cols, labs, fams = [], [], [], [], [], [], []
        for i, e in enumerate(entries):
            s = e["summary"].get(q)
            if s is None:
                continue
            xs.append(i); ys.append(s["spearman"])
            los.append(s["spearman"] - s["ci_lo"]); his.append(s["ci_hi"] - s["spearman"])
            cols.append(fam_color[e["family"]]); labs.append(e["label"]); fams.append(e["family"])
        ax.errorbar(xs, ys, yerr=[np.abs(los), np.abs(his)], fmt="none", ecolor="gray",
                    elinewidth=1, capsize=3, zorder=1)
        ax.scatter(xs, ys, c=cols, s=55, zorder=2)
        for i in range(1, len(fams)):
            if fams[i] != fams[i - 1]:
                ax.axvline(i - 0.5, color="k", lw=0.6, alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(q); ax.grid(True, axis="y", alpha=0.3)
    _apply_legend(fig, axs)
    _maybe_suptitle(fig, plot_cfg.get("title", None))
    fig.tight_layout()
    main_paths = _savefig(fig, out_dir / name, formats, dpi)
    plt.close(fig)

    # ---- companion: corr vs VF across conditions ----
    fig, axs = plt.subplots(1, len(qlist), figsize=plot_cfg.get("scatter_figsize", [5.5 * len(qlist), 4.5]), squeeze=False)
    axs = axs[0]
    for ax, q in zip(axs, qlist):
        for e in entries:
            s = e["summary"].get(q)
            if s is None:
                continue
            ax.scatter(s["vf"], s["spearman"], color=fam_color[e["family"]], s=55)
            if plot_cfg.get("annotate_points", True):
                ax.annotate(e["label"], (s["vf"], s["spearman"]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(plot_cfg.get("vf_xlabel", "variance fraction VF (band)"))
        ax.set_ylabel(ylabel)
        ax.set_title(q); ax.grid(True, alpha=0.3)
    _apply_legend(fig, axs)
    _maybe_suptitle(fig, plot_cfg.get("scatter_title", "Per-condition localization $\\rho$ vs variance fraction"))
    fig.tight_layout()
    scat_paths = _savefig(fig, out_dir / f"{name}_corr_vs_vf", formats, dpi)
    plt.close(fig)

    manifest = {
        "name": name, "kind": "conditions", "band": list(band) if band else None,
        "quantities": qlist, "out_paths": main_paths, "corr_vs_vf": scat_paths,
        "conditions": [{"label": e["label"], "family": e["family"], "n_seeds": e["n_seeds"],
                        "summary": e["summary"]} for e in entries],
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("[c] saved %s", out_dir / f"{name}.png")
    return manifest

def generate_radial_image_figure(spec_path: Union[str, Path],
    device_override: str | None = None,
    output_dir_override: Union[str, Path, None] = None,
) -> dict:
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("figure", {})
    name = str(cfg.get("name", spec_path.stem))
    out_dir = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg["output_dir"], spec_root) if cfg.get("output_dir")
               else _default_diag_out(cfg, "decomp", spec_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_override if device_override is not None else cfg.get("device", None)
    default_path_remap = cfg.get("path_remap", None)
    quantities = [str(q) for q in cfg.get("quantities", ["ne", "temp"])]
    radii = [float(r) for r in cfg.get("radii", _DEFAULT_RADII)]
    error_mode = str(cfg.get("error_mode", "ensemble_mean_abs"))
    disagreement = str(cfg.get("disagreement", "std"))
    metric = str(cfg.get("correlation", "spearman"))
    show_secondary = bool(cfg.get("show_secondary_metric", True))
    plot_cfg = dict(cfg.get("plot", {}) or {})
    img_panel = dict(cfg.get("image_panel", {}) or {})
    cond = cfg.get("condition", None)
    if cond is None:
        raise ValueError("radial_image requires a single figure.condition")

    bdir = _resolve_path(cond["benchmark_dir"], spec_root)
    path_remap = cond.get("path_remap", default_path_remap)
    rows = collect_run_summaries(bdir)
    sel = {k: cond[k] for k in
           ("experiment", "where", "filters", "run_names", "run_dirs", "min_seeds", "max_seeds")
           if k in cond}
    seed_rows = _collect_condition_seed_rows(rows, sel)

    img_cfg = {"enabled": True, "split": img_panel.get("split", "test"),
               "num_views": 1, "view_index": int(img_panel.get("view_index", 0)),
               "view_strategy": "evenly_spaced", "chunk_size": int(img_panel.get("chunk_size", 8192))}
    field, img = _accumulate_ensemble(seed_rows, bdir, quantities, radii, img_cfg, device, path_remap)
    reduced = _reduce_field(field, error_mode, disagreement)
    image_red = _reduce_image(img, str(img_panel.get("residual_mode", "signed")))

    psc = _per_shell_corr(reduced)
    qlist = [q for q in quantities if q in psc]
    other = "pearson" if metric == "spearman" else "spearman"
    formats = list(plot_cfg.get("formats", ["pdf", "png"]))
    dpi = int(plot_cfg.get("dpi", 300))
    ncols = len(qlist) + (1 if image_red is not None else 0)

    fig, axs = plt.subplots(1, ncols, figsize=plot_cfg.get("figsize", [4.6 * ncols, 4.2]), squeeze=False)
    axs = axs[0]
    for ax, q in zip(axs[:len(qlist)], qlist):
        cur = psc[q]
        ax.plot(cur["r"], cur[metric], marker="o", label=metric)
        if show_secondary:
            ax.plot(cur["r"], cur[other], marker=".", ls="--", alpha=0.6, label=other)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel(plot_cfg.get("xlabel", "Radius [$R_\\odot$]"))
        ax.set_ylabel(plot_cfg.get("ylabel", r"$\rho$( disagreement , |error| )"))
        ax.set_title(q); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    if image_red is not None:
        ax = axs[-1]
        ch = int(img_panel.get("channel_index", 0))
        disp_cfg = dict(img_panel.get("display", {}) or {})
        disp = _apply_display(image_red["residual"][0, ..., ch], disp_cfg, ch)
        cmap = _cmap_with_bad(plot_cfg.get("image_resid_cmap", "coolwarm"), plot_cfg.get("bad_color", "black"))
        pct = disp_cfg.get("percentile", 99.0); pct = pct[-1] if isinstance(pct, (list, tuple)) else pct
        v = np.abs(disp[np.isfinite(disp)])
        lim = float(np.nanpercentile(v, float(pct))) if v.size else 1.0
        interp = plot_cfg.get("interpolation", "nearest")

        if bool(img_panel.get("show_ticks", True)):
            fov = float(img_panel.get("fov_rsun", 3.0))
            im = ax.imshow(disp, cmap=cmap, vmin=-lim, vmax=lim, origin="lower",
                           extent=[-fov, fov, -fov, fov], interpolation=interp)
            ax.set_xlabel(img_panel.get("xlabel", "x [$R_\\odot$]"))
            ax.set_ylabel(img_panel.get("ylabel", "y [$R_\\odot$]"))
        else:
            im = ax.imshow(disp, cmap=cmap, vmin=-lim, vmax=lim, interpolation=interp)
            ax.set_xticks([]); ax.set_yticks([])

        labels = image_red["channel_labels"] or []
        ax.set_title("Held-out image residual\n" + (labels[ch] if ch < len(labels) else f"ch {ch}"))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=disp_cfg.get("label", "render - clean"))

    _maybe_suptitle(fig, plot_cfg.get("title", None))
    fig.tight_layout()
    out_paths = _savefig(fig, out_dir / name, formats, dpi)
    plt.close(fig)

    manifest = {"name": name, "kind": "radial_image", "metric": metric,
                "condition": str(cond.get("label", cond.get("experiment"))),
                "n_seeds": len(seed_rows), "quantities": qlist, "out_paths": out_paths,
                "per_shell": {q: {"r": psc[q]["r"].tolist(),
                                  "spearman": psc[q]["spearman"].tolist(),
                                  "pearson": psc[q]["pearson"].tolist()} for q in qlist}}
    save_json(manifest, out_dir / f"{name}_manifest.json")
    return manifest

