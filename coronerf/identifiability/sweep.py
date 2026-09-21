##########################################################
### identifiability/sweep.py
### Sweep identifiability + bridge across CONDITIONS (and, later, operating points).
### Build ONE base operator, derive each condition (row-subset / re-whiten, no rebuild),
### analyze + bridge, write sweep_manifest.json, then aggregate into cross-condition figures.
##########################################################

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..benchmark.config import load_yaml, save_json
from ..util.figure import _resolve_path, _savefig
from .operator import load_operator, build_operator_from_cfg, subset_operator, pick_views
from .run import analyze_and_bridge, _plot_bridge
from ._core import _plot_spectrum, _plot_maps, _plot_posterior_maps

logger = logging.getLogger("coroNeRF.identifiability.sweep")


def _f(x):
    try:
        v = float(x); return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


# ============================ driver ============================

def run_conditions_sweep(spec_path, device_override=None, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("conditions_sweep", {})
    name = str(cfg.get("name", spec_path.stem))
    runs_root = _resolve_path(cfg.get("output_root", "../../runs_identifiability"), spec_root)
    sweep_dir = runs_root / "sweeps" / name
    (sweep_dir / "conditions").mkdir(parents=True, exist_ok=True)

    tau = float(cfg.get("tau", 1.0))
    prior = dict(cfg.get("prior", {}) or {})
    bridge_cfg = dict(cfg.get("bridge", {}) or {})
    plot_cfg = dict(cfg.get("plot", {}) or {})
    pcp = dict(cfg.get("per_condition_plots", {}) or {})

    # base operator (build once; cached under operators/)
    base_dir = build_operator_from_cfg(dict(cfg["base"]), spec_root,
                                       device_override=device_override, output_root=runs_root)
    base_op = load_operator(base_dir)
    base_shot = list(base_op["_meta"]["base_shot_coeff"])
    base_floor = list(base_op["_meta"].get("base_sigma_floor", [0.0] * len(base_shot)))
    logger.info("sweep '%s': base=%s (N=%d, M=%d, views=%d)", name, base_dir.name,
                int(base_op["_meta"]["N"]), int(base_op["_meta"]["M"]), int(base_op["vidx"].size))

    rows = []
    for cond in cfg.get("conditions", []):
        cid = str(cond["id"]); fam = str(cond.get("family", "other"))
        eta = float(cond.get("eta", 1.0))
        channels = cond.get("channels", None)                       # None = all channels
        views = int(cond.get("views", int(base_op["vidx"].size)))
        mode = str(cond.get("mode", "derive"))
        npz = _resolve_path(cond["disagreement_npz"], spec_root) if cond.get("disagreement_npz") else None

        if mode == "build":                                          # [2] / genuine forward-op change
            op = load_operator(build_operator_from_cfg(dict(cond["operator"]), spec_root,
                                                       device_override=device_override, output_root=runs_root))
            op_tag = op["_meta"].get("tag", "built")
        else:
            vs = pick_views(base_op["vidx"], views) if views < int(base_op["vidx"].size) else None
            op = subset_operator(base_op, view_subset=vs, channel_subset=channels)
            op_tag = f"{base_dir.name}[derive]"

        shot_c = [s * eta for s in base_shot]
        res, row = analyze_and_bridge(op, tau=tau, prior_cfg=prior, base_shot=shot_c, base_floor=base_floor,
                                      disag_npz=str(npz) if npz else None, bridge_cfg=bridge_cfg)
        row.update({"id": cid, "family": fam, "eta": eta, "views": int(op["vidx"].size),
                    "channels": sorted(int(c) for c in np.unique(op["obs_chan_global"])),
                    "mode": mode, "operator_tag": op_tag,
                    "sigma_ens_label": str(cond.get("sigma_ens_label", cid)),
                    "disagreement_npz": str(npz) if npz else None})
        rows.append(row)

        if any(pcp.get(k, False) for k in ("spectrum", "maps", "posterior", "bridge")):
            cd = sweep_dir / "conditions" / cid; cd.mkdir(parents=True, exist_ok=True)
            if pcp.get("spectrum"):  _plot_spectrum(res, dict(plot_cfg), cd / "spectrum")
            if pcp.get("maps"):      _plot_maps(res, dict(plot_cfg), cd / "maps")
            if pcp.get("posterior"): _plot_posterior_maps(res, dict(plot_cfg), cd / "posterior")
            if pcp.get("bridge") and npz is not None:
                _plot_bridge(res, str(npz), {**plot_cfg, **bridge_cfg}, cd / "bridge")

        logger.info("  [%s] eff_rank=%s PR=%.1f%s", cid, row.get("eff_rank_tau"),
                    _f(row.get("eff_rank_participation")), "" if row["bridge"] is None else "  (bridged)")
        
        if cfg.get("save_analysis", False):
            cd = sweep_dir / "conditions" / cid; cd.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cd / "analysis.npz",
                singular_values=res["singular_values"], quantities=np.array(res["quantities"]),
                crb_std_grid=res["crb_std_grid"], post_std_grid=res["post_std_grid"],
                prior_logratio_grid=res["prior_logratio_grid"],
                lon_axis=res["lon_axis"], phi_axis=res["phi_axis"], r_axis=res["r_axis"])

    manifest = {"name": name, "spec_path": str(spec_path), "base_operator": base_dir.name,
                "tau": tau, "prior": prior, "rows": rows}
    save_json(manifest, sweep_dir / "sweep_manifest.json")
    logger.info("sweep '%s' -> %s (%d conditions)", name, sweep_dir, len(rows))
    return manifest


# ============================ aggregate ============================

def aggregate_conditions_sweep(spec_path, device_override=None, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("conditions_sweep", {})
    name = str(cfg.get("name", spec_path.stem))
    runs_root = _resolve_path(cfg.get("output_root", "../../runs_identifiability"), spec_root)
    sweep_dir = runs_root / "sweeps" / name
    agg_dir = sweep_dir / "aggregate"; agg_dir.mkdir(parents=True, exist_ok=True)

    with open(sweep_dir / "sweep_manifest.json") as f:
        rows = json.load(f)["rows"]

    acfg = dict(cfg.get("aggregate", {}) or {})
    plots = dict(acfg.get("plots", {}) or {})
    formats = list(acfg.get("formats", ["pdf", "png"])); dpi = int(acfg.get("dpi", 300))
    group_by = str(acfg.get("group_by", "family"))
    quantities = _row_quantities(rows)

    sig_corr = (_load_sigma_ens_conditions(_resolve_path(acfg["sigma_ens_conditions_manifest"], spec_root))
                if acfg.get("sigma_ens_conditions_manifest") else {})

    outputs = {}
    def _emit(key, fig):
        outputs[key] = _savefig(fig, agg_dir / key, formats, dpi); plt.close(fig)

    if plots.get("corr_post_vs_crb", True):
        _emit("corr_post_vs_crb", _fig_corr(rows, quantities, group_by,
              sig_corr if plots.get("overlay_sigma_ens_error") else None))
    if plots.get("gap_ci", False):
        _emit("gap_ci", _fig_gap(rows, quantities, group_by))
    if plots.get("eff_rank", False):
        _emit("eff_rank", _fig_scalar(rows, "eff_rank_tau", "effective rank", group_by))
    if plots.get("participation_ratio", False):
        _emit("participation_ratio", _fig_scalar(rows, "eff_rank_participation", "participation ratio", group_by))
    if plots.get("condition_number", False):
        _emit("condition_number", _fig_scalar(rows, "condition_tau", r"$\kappa(\tau)$", group_by, logy=True))
    if plots.get("frac_prior_dominated", False):
        _emit("frac_prior_dominated", _fig_scalar(rows, "frac_prior_dominated", "frac prior-dominated", group_by))
    if plots.get("scatter_effrank_vs_rho", False):
        _emit("scatter_effrank_vs_rho", _fig_scatter(rows, quantities, group_by))
    if plots.get("operating_heatmap"):
        oa = dict(cfg.get("operating_axes", {}) or {})
        _LOG = {"crb_std_median": True, "post_std_median": True, "condition_tau": True, "condition_full": True}
        for metric in cfg.get("heatmap_metrics", ["eff_rank_tau"]):
            outputs[f"heatmap_{metric}"] = _fig_operating_heatmap(
                rows, metric=metric, out_base=agg_dir / f"heatmap_{metric}",
                formats=formats, dpi=dpi, views_axis=oa.get("views"), eta_axis=oa.get("eta"),
                log_metric=_LOG.get(metric, False))
    if plots.get("regime_map"):
        outputs["regime_map"] = _fig_regime_map(rows, out_base=agg_dir / "regime_map",
                                                formats=formats, dpi=dpi, group_by = group_by)

    save_json({"name": name, "outputs": outputs, "n_conditions": len(rows)}, agg_dir / "aggregate.json")
    logger.info("aggregate '%s' -> %s", name, agg_dir)
    return {"outputs": outputs}


# ---------------------- aggregate helpers ----------------------

def _row_quantities(rows):
    qs = []
    for r in rows:
        for q in (r.get("bridge") or {}):
            if q not in qs: qs.append(q)
    return qs or ["ne", "temp"]


def _load_sigma_ens_conditions(manifest_path):
    """Parse the seed-disagreement conditions manifest -> {label: {q: spearman}} for overlay."""
    with open(manifest_path) as f:
        d = json.load(f)
    return {str(c.get("label")): {q: _f(v.get("spearman")) for q, v in (c.get("summary", {}) or {}).items()}
            for c in d.get("conditions", [])}


def _group_colors(rows, group_by):
    groups = []
    for r in rows:
        g = str(r.get(group_by, "other"))
        if g not in groups: groups.append(g)
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(10)])
    return {g: cyc[i % len(cyc)] for i, g in enumerate(groups)}


def _fig_corr(rows, quantities, group_by, sig_corr=None):
    gc = _group_colors(rows, group_by); xs = list(range(len(rows))); labels = [r["id"] for r in rows]
    fig, axs = plt.subplots(1, len(quantities), figsize=[6.8 * len(quantities), 4.6], squeeze=False); axs = axs[0]
    for ax, q in zip(axs, quantities):
        for x, r in zip(xs, rows):
            b = (r.get("bridge") or {}).get(q); col = gc[str(r.get(group_by, "other"))]
            if not b: continue
            p, c = _f(b["sigma_vs_post"]["spearman"]), _f(b["sigma_vs_crb"]["spearman"])
            if np.isfinite(p) and np.isfinite(c): ax.plot([x, x], [c, p], color="gray", lw=0.8, alpha=0.5, zorder=1)
            ax.scatter([x], [p], color=col, s=60, marker="o", zorder=3)      # rho(post)
            ax.scatter([x], [c], color=col, s=55, marker="x", zorder=3)      # rho(CRB)
        if sig_corr is not None:
            emp = [_f(sig_corr.get(r.get("sigma_ens_label", r["id"]), {}).get(q)) for r in rows]
            ax.plot(xs, emp, ls=":", color="k", marker="s", ms=4, alpha=0.75)
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(r"Spearman $\rho$"); ax.set_title(q); ax.grid(True, axis="y", alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=g) for g, c in gc.items()]
    handles += [plt.Line2D([0], [0], marker="o", ls="", color="gray", label=r"$\rho$(post)"),
                plt.Line2D([0], [0], marker="x", ls="", color="gray", label=r"$\rho$(CRB)")]
    if sig_corr is not None:
        handles += [plt.Line2D([0], [0], marker="s", ls=":", color="k", label=r"$\rho(\sigma,|{\rm err}|)$")]
    axs[-1].legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout(); return fig


def _fig_gap(rows, quantities, group_by):
    gc = _group_colors(rows, group_by); xs = list(range(len(rows))); labels = [r["id"] for r in rows]
    fig, axs = plt.subplots(1, len(quantities), figsize=[6.8 * len(quantities), 4.6], squeeze=False); axs = axs[0]
    for ax, q in zip(axs, quantities):
        for x, r in zip(xs, rows):
            b = (r.get("bridge") or {}).get(q)
            if not b: continue
            gap, lo, hi = _f(b.get("gap_post_minus_crb")), _f(b.get("gap_ci_lo")), _f(b.get("gap_ci_hi"))
            col = gc[str(r.get(group_by, "other"))]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar([x], [gap], yerr=[[gap - lo], [hi - gap]], fmt="o", color=col, capsize=3)
            else:
                ax.scatter([x], [gap], color=col, s=50)
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(r"$\rho_{\rm post}-\rho_{\rm CRB}$"); ax.set_title(q); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); return fig


def _fig_scalar(rows, key, ylabel, group_by, logy=False):
    gc = _group_colors(rows, group_by); xs = list(range(len(rows))); labels = [r["id"] for r in rows]
    fig, ax = plt.subplots(figsize=[7.2, 4.3])
    ax.scatter(xs, [_f(r.get(key)) for r in rows], c=[gc[str(r.get(group_by, "other"))] for r in rows], s=60)
    if logy: ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=c, label=g) for g, c in gc.items()], fontsize=7)
    fig.tight_layout(); return fig


def _fig_scatter(rows, quantities, group_by):
    gc = _group_colors(rows, group_by)
    fig, axs = plt.subplots(1, len(quantities), figsize=[5.6 * len(quantities), 4.6], squeeze=False); axs = axs[0]
    for ax, q in zip(axs, quantities):
        for r in rows:
            b = (r.get("bridge") or {}).get(q)
            if not b: continue
            er, rp = _f(r.get("eff_rank_tau")), _f(b["sigma_vs_post"]["spearman"])
            ax.scatter([er], [rp], color=gc[str(r.get(group_by, "other"))], s=60)
            ax.annotate(r["id"], (er, rp), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("effective rank"); ax.set_ylabel(r"$\rho(\sigma_{\rm ens},{\rm post})$"); ax.set_title(q)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(); return fig


def _fig_operating_heatmap(rows, *, metric, out_base, formats, dpi,
                           views_axis=None, eta_axis=None, log_metric=False, annotate_trained=True):
    """2-D (noise x views) heatmap of a scalar per-row metric; red box = trained cell (has sigma_ens)."""
    import numpy as np
    import matplotlib.pyplot as plt
    V = [int(v) for v in views_axis] if views_axis else sorted({int(r["views"]) for r in rows if r.get("views") is not None})
    E = [float(e) for e in eta_axis] if eta_axis else sorted({float(r["eta"]) for r in rows if r.get("eta") is not None})
    G = np.full((len(E), len(V)), np.nan); trained = np.zeros((len(E), len(V)), bool)
    for r in rows:
        try: i = E.index(float(r.get("eta"))); j = V.index(int(r.get("views")))
        except (ValueError, TypeError): continue
        v = r.get(metric)
        if isinstance(v, dict): v = v.get("ne", next(iter(v.values()), None))
        if v is not None and np.isfinite(v): G[i, j] = float(v)
        if r.get("disagreement_npz"): trained[i, j] = True
    disp = np.log10(G) if log_metric else G
    fig, ax = plt.subplots(figsize=(1.15 * len(V) + 2.0, 0.95 * len(E) + 1.4))
    im = ax.imshow(disp, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(V))); ax.set_xticklabels(V)
    ax.set_yticks(range(len(E))); ax.set_yticklabels([f"{e:g}" for e in E])
    ax.set_xlabel("# views"); ax.set_ylabel(r"noise multiplier $\eta$")
    intfmt = (metric == "eff_rank_tau")
    for i in range(len(E)):
        for j in range(len(V)):
            if np.isfinite(G[i, j]):
                ax.text(j, i, f"{G[i,j]:.0f}" if intfmt else f"{G[i,j]:.2g}",
                        ha="center", va="center", color="w", fontsize=8)
            if annotate_trained and trained[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2.0))
    fig.colorbar(im, ax=ax, label=(f"log10 {metric}" if log_metric else metric))
    ax.set_title(metric); fig.tight_layout()
    return _savefig(fig, out_base, formats, dpi)


def _localization_rho(npz_path):
    import numpy as np
    d = np.load(npz_path); out = {}
    for q in ("ne", "temp"):
        s = d[f"{q}_disagreement"].astype(float).ravel(); e = d[f"{q}_true_err"].astype(float).ravel()
        ok = np.isfinite(s) & np.isfinite(e); s, e = s[ok], e[ok]
        if s.size > 2:
            rs = np.argsort(np.argsort(s)); re = np.argsort(np.argsort(e))
            out[q] = float(np.corrcoef(rs, re)[0, 1])
        else: out[q] = float("nan")
    return out

def _fig_regime_map(rows, *, out_base, formats, dpi, group_by):
    import numpy as np, matplotlib.pyplot as plt
    gc = _group_colors(rows, group_by)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for qi, q in enumerate(("ne", "temp")):
        ax = axes[qi]
        for r in rows:
            npz = r.get("disagreement_npz")
            if not npz: continue
            bd = (r.get("bridge") or {}).get(q, {}).get("sigma_vs_post", {})
            by = bd.get("spearman") if isinstance(bd, dict) else None
            if by is None: continue
            try: lx = _localization_rho(npz)[q]
            except Exception: continue
            col = gc.get(str(r.get("family", "other")), "C7")
            ax.scatter(lx, by, c=[col], s=70, edgecolor="k", lw=0.5, zorder=3)
            ax.annotate(str(r["id"]), (lx, by), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="grey", lw=0.7); ax.axvline(0, color="grey", lw=0.7)
        ax.plot([-0.4, 0.9], [-0.4, 0.9], ls="--", color="grey", lw=0.8)   # y=x
        ax.set_xlabel(r"localization $\rho$  ($\sigma_\mathrm{ens}$ vs true error)")
        ax.set_ylabel(r"bridge $\rho$  ($\sigma_\mathrm{ens}$ vs posterior)")
        ax.set_title(q)
    fig.tight_layout()
    return _savefig(fig, out_base, formats, dpi)
