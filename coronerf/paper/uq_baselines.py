from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .uncertainty_quantification import _cfg, _outdir          # reuse the paper-fig helpers
from ..util.figure import _resolve_path, _savefig

_COL = {"oracle": "k", "random": "0.6", "sigma": "#009E73", "emis": "#D55E00",
        "dens": "#1f77b4", "fisher": "#9467bd"}

# ci95 (from the block bootstrap) only exists for these five headline scalars:
def _ci_key(source, key):
    if source == "nause":
        return {"sigma": "nause_sigma", "emis": "nause_emis"}.get(key)
    return key if key in ("sigma_vs_err", "emis_vs_err", "sigma_vs_err_given_emis") else None


# ============================ Mode A: correlation / nAUSE ladder ============================
def fig_baseline_ladder(spec_path, device_override=None, output_dir_override=None):
    """Ladder across conditions of any baseline scalar(s): correlations | nause | ause.
    Default: rho(sigma_ens, eps) vs rho(1/emissivity, eps)."""
    cfg, root = _cfg(spec_path, "baseline_ladder")
    out = _outdir(cfg, root, output_dir_override)
    man = json.load(open(_resolve_path(cfg["manifest"], root)))
    conds = {c["id"]: c for c in man["conditions"]}
    order = [i for i in (cfg.get("order") or [c["id"] for c in man["conditions"]]) if i in conds]
    q = str(cfg.get("quantity", "ne"))
    labels = cfg.get("condition_labels", {}) or {}
    plot = dict(cfg.get("plot", {}) or {})
    series = cfg.get("series") or [
        {"source": "correlations", "key": "sigma_vs_err", "label": r"$\rho(\sigma_{\rm ens},\,\epsilon)$", "color": "#1f77b4", "marker": "o"},
        {"source": "correlations", "key": "emis_vs_err",  "label": r"$\rho(1/\varepsilon,\,\epsilon)$",    "color": "#ff7f0e", "marker": "s"},
    ]

    def _val(cid, s):
        qd = conds[cid]["quantities"][q]; src = s["source"]; key = s["key"]
        if src == "nause":
            v = (qd.get("nause") or {}).get(key)
            if v is None:                                       # fallback if run predates the nause patch
                au = qd["ause"]; v = au[key] / au["random"] if au.get("random") else np.nan
        else:
            v = qd.get(src, {}).get(key, np.nan)
        lo = hi = None
        ck = _ci_key(src, key); ci = qd.get("ci95", {})
        if ck and ck in ci: lo, hi = ci[ck]
        return v, lo, hi

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=tuple(plot.get("figsize", [9, 3.6])))
    for s in series:
        ys, elo, ehi, has_ci = [], [], [], False
        for cid in order:
            v, lo, hi = _val(cid, s); ys.append(v)
            if lo is not None and np.isfinite(v): elo.append(v - lo); ehi.append(hi - v); has_ci = True
            else: elo.append(np.nan); ehi.append(np.nan)
        kw = dict(color=s.get("color"), marker=s.get("marker", "o"), ls=s.get("ls", "none"),   # markers only
                  ms=plot.get("ms", 7), mew=plot.get("mew", 1.2), label=s.get("label", s["key"]))
        if bool(cfg.get("errorbars", True)) and has_ci:
            ax.errorbar(x, ys, yerr=[elo, ehi], capsize=2, **kw)
        else:
            ax.plot(x, ys, **kw)
    for xv in (cfg.get("dividers") or []): ax.axvline(float(xv) + 0.5, color="0.9", lw=1)
    if bool(cfg.get("hline0", True)): ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([labels.get(i, i) for i in order], rotation=int(plot.get("tick_rotation", 45)), ha="right", fontsize=int(plot.get("tick_fontsize", 8)))
    ax.set_ylabel(cfg.get("ylabel", r"Spearman $\rho$"), fontsize = int(plot.get("ylabel_fontsize", 8))); ax.set_xlim(-0.5, len(order) - 0.5)
    ax.legend(fontsize=int(plot.get("legend_fontsize", 8)), ncol=int(plot.get("legend_ncol", 2)), framealpha=0.9)
    fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_baseline_ladder"), cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))


# ============================ Mode B: sparsification / SE curves ============================
def fig_sparsification(spec_path, device_override=None, output_dir_override=None):
    """Grid of sparsification curves (retained MAE or SE vs fraction removed), one panel per condition."""
    cfg, root = _cfg(spec_path, "sparsification")
    out = _outdir(cfg, root, output_dir_override)
    bdir = Path(_resolve_path(cfg["baselines_dir"], root))
    q = str(cfg.get("quantity", "ne"))
    conds = list(cfg.get("conditions", ["clean_300v", "noise_x9", "views_5", "si_x1p4"]))
    labels = cfg.get("condition_labels", {}) or {}
    methods = list(cfg.get("methods", ["sigma", "emis", "oracle", "random"]))
    ymode = str(cfg.get("ymode", "risk"))                       # "risk" (retained MAE) | "se" (risk - oracle)
    plot = dict(cfg.get("plot", {}) or {})

    nause_by = {}
    if cfg.get("manifest"):
        for c in json.load(open(_resolve_path(cfg["manifest"], root)))["conditions"]:
            nause_by[c["id"]] = (c["quantities"].get(q, {}) or {}).get("nause", {})

    ncol = int(cfg.get("ncols", 2)); nrow = int(np.ceil(len(conds) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=tuple(plot.get("figsize", [9, 3.4 * nrow])), squeeze=False)
    for i, cid in enumerate(conds):
        ax = axs[i // ncol][i % ncol]
        d = np.load(bdir / cid / "baselines.npz"); t = d["t_grid"]; oracle = d[f"{q}_risk_oracle"]
        for m in methods:
            key = f"{q}_risk_{m}"
            if key not in d.files: continue
            y = d[key] - oracle if ymode == "se" else d[key]
            na = (nause_by.get(cid, {}) or {}).get(m)
            lab = m + (f" (nAUSE={na:.2f})" if na is not None else "")
            ax.plot(t, y, color=_COL.get(m, "C0"), lw=plot.get("lw", 1.6), label=lab)
        ax.set_title(labels.get(cid, cid), fontsize=plot.get("title_fontsize", 10))
        if i // ncol == nrow - 1: ax.set_xlabel("fraction removed  $t$")
        if i % ncol == 0: ax.set_ylabel(("retained MAE" if ymode == "risk" else "sparsification error") + f"  ({q})")
        ax.legend(fontsize=7, ncol=int(plot.get("legend_ncol", 1)))
    for j in range(len(conds), nrow * ncol): axs[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_sparsification"), cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))



# composite for the paper
def _draw_ladder_ax(ax, root, lcfg, manifest_path, q_default="ne"):
    plot = dict(lcfg.get("plot", {}) or {})
    man = json.load(open(_resolve_path(manifest_path, root)))
    conds = {c["id"]: c for c in man["conditions"]}
    order = [i for i in (lcfg.get("order") or [c["id"] for c in man["conditions"]]) if i in conds]
    q = str(lcfg.get("quantity", q_default)); labels = lcfg.get("condition_labels", {}) or {}
    series = lcfg.get("series") or [
        {"source": "correlations", "key": "sigma_vs_err", "label": r"$\rho(\sigma_{\rm ens},\,\epsilon)$", "color": "#009E73", "marker": "o"},
        {"source": "correlations", "key": "emis_vs_err",  "label": r"$\rho(b_{\mathcal E},\,\epsilon)$",    "color": "#D55E00", "marker": "s"},
    ]
    def _val(cid, s):
        qd = conds[cid]["quantities"][q]; src = s["source"]; key = s["key"]
        if src == "nause":
            v = (qd.get("nause") or {}).get(key)
            if v is None:
                au = qd["ause"]; v = au[key] / au["random"] if au.get("random") else np.nan
        else:
            v = qd.get(src, {}).get(key, np.nan)
        lo = hi = None; ck = _ci_key(src, key); ci = qd.get("ci95", {})
        if ck and ck in ci: lo, hi = ci[ck]
        return v, lo, hi
    x = np.arange(len(order))
    for s in series:
        ys, elo, ehi, has_ci = [], [], [], False
        for cid in order:
            v, lo, hi = _val(cid, s); ys.append(v)
            if lo is not None and np.isfinite(v): elo.append(v - lo); ehi.append(hi - v); has_ci = True
            else: elo.append(np.nan); ehi.append(np.nan)
        kw = dict(color=s.get("color"), marker=s.get("marker", "o"), ls=s.get("ls", "none"),
                  ms=plot.get("ms", 7), mew=plot.get("mew", 1.2), label=s.get("label", s["key"]))
        if bool(lcfg.get("errorbars", True)) and has_ci: ax.errorbar(x, ys, yerr=[elo, ehi], capsize=2, **kw)
        else: ax.plot(x, ys, **kw)
    for xv in (lcfg.get("dividers") or []): ax.axvline(float(xv) + 0.5, color="0.9", lw=1)
    if bool(lcfg.get("hline0", True)): ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([labels.get(i, i) for i in order],
                                         rotation=int(plot.get("tick_rotation", 45)), ha="right", fontsize=int(plot.get("tick_fontsize", 8)))
    ax.set_ylabel(lcfg.get("ylabel", r"Spearman $\rho$"), fontsize=int(plot.get("ylabel_fontsize", 9)))
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.legend(fontsize=int(plot.get("legend_fontsize", 8)), ncol=int(plot.get("legend_ncol", 2)),
              loc=plot.get("legend_loc", "best"), framealpha=0.9)

def _draw_spars_ax(ax, root, scfg, baselines_dir, manifest_path, q_default="ne"):
    plot = dict(scfg.get("plot", {}) or {})
    q = str(scfg.get("quantity", q_default)); cid = str(scfg.get("condition", "noise_x9"))
    methods = list(scfg.get("methods", ["oracle", "random", "sigma", "emis"]))
    ymode = str(scfg.get("ymode", "risk"))
    d = np.load(Path(_resolve_path(baselines_dir, root)) / cid / "baselines.npz")
    t = d["t_grid"]; oracle = d[f"{q}_risk_oracle"]
    nause = {}
    if manifest_path:
        for c in json.load(open(_resolve_path(manifest_path, root)))["conditions"]:
            if c["id"] == cid: nause = (c["quantities"].get(q, {}) or {}).get("nause", {})
    for m in methods:
        key = f"{q}_risk_{m}"
        if key not in d.files: continue
        y = d[key] - oracle if ymode == "se" else d[key]
        na = nause.get(m); lab = m + (f" (nAUSE={na:.2f})" if na is not None else "")
        ax.plot(t, y, color=_COL.get(m, "C0"), lw=plot.get("lw", 1.8), label=lab)
    ax.set_xlabel(r"fraction removed  $t$", fontsize=int(plot.get("label_fontsize", 9)))
    ax.set_ylabel(("retained MAE" if ymode == "risk" else "sparsification error") + f"  ({q})",
                  fontsize=int(plot.get("label_fontsize", 9)))
    ax.set_title(scfg.get("title", cid), fontsize=int(plot.get("title_fontsize", 10)))
    ax.tick_params(labelsize=int(plot.get("tick_fontsize", 8)))
    ax.legend(fontsize=int(plot.get("legend_fontsize", 7)), ncol=int(plot.get("legend_ncol", 1)),
              loc=plot.get("legend_loc", "best"))


# correlation + sparsification summary
def fig_baseline_summary(spec_path, device_override=None, output_dir_override=None):
    """Composite: baseline correlation ladder + one sparsification panel (main-text figure)."""
    cfg, root = _cfg(spec_path, "baseline_summary")
    out = _outdir(cfg, root, output_dir_override)
    q = str(cfg.get("quantity", "ne"))
    layout = str(cfg.get("layout", "horizontal"))               # horizontal (side-by-side) | vertical (stacked)
    ratios = list(cfg.get("ratios", [1.7, 1.0]))                # width_ratios if horizontal, height_ratios if vertical
    figsize = tuple(cfg.get("figsize", [12, 3.8] if layout == "horizontal" else [6.5, 7.0]))
    if layout == "horizontal":
        fig, (a_lad, a_spar) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": ratios})
    else:
        fig, (a_lad, a_spar) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": ratios})
    lcfg = dict(cfg.get("ladder", {}) or {});         lcfg.setdefault("quantity", q)
    scfg = dict(cfg.get("sparsification", {}) or {}); scfg.setdefault("quantity", q)
    _draw_ladder_ax(a_lad, root, lcfg, cfg["manifest"], q)
    _draw_spars_ax(a_spar, root, scfg, cfg["baselines_dir"], cfg.get("manifest"), q)
    if bool(cfg.get("panel_labels", True)):
        for ax, lab in zip((a_lad, a_spar), cfg.get("panel_label_text", ["(a)", "(b)"])):
            ax.text(-0.06, 1.03, lab, transform=ax.transAxes, fontweight="bold",
                    fontsize=int(cfg.get("panel_label_fontsize", 11)), va="bottom", ha="right")
    fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_baseline_summary"),
                    cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))



# error capture summary
# ============================ Mode: per-field error-capture ladder ============================
def _ecap_stat(ecap, stat):
    """Return (value, lo, hi) in FRACTION units for one series 'stat' from a manifest ecap dict."""
    if stat == "e_cap":     return ecap["e_cap"], None, None                       # observed absolute capture
    if stat == "split":     return ecap["split_mean"], ecap["split_ci"][0], ecap["split_ci"][1]
    if stat == "rot_null":  return ecap["rot_null_mean"], None, None               # longitude-rotation null mean
    if stat == "rot_max":   return ecap["rot_null_max"], None, None
    if stat == "isotropic": return ecap["e_cap_rand"], None, None                  # r/M rank-matched reference
    if stat == "enrich":    return ecap["enrich"], None, None
    return ecap.get(stat), None, None                                             # any raw scalar key


def fig_errorcap_ladder(spec_path, device_override=None, output_dir_override=None):
    """Per-field error-capture ladder: absolute E_cap (+ split-seed range, rotation null, isotropic ref) across a
    selectable list of conditions. One panel per quantity; every series/colour/marker is yaml-driven."""
    cfg, root = _cfg(spec_path, "errorcap_ladder")
    out = _outdir(cfg, root, output_dir_override)
    man = json.load(open(_resolve_path(cfg["manifest"], root)))
    conds = {c["id"]: c for c in man["conditions"]}
    bdir = Path(_resolve_path(cfg["baselines_dir"], root)) if cfg.get("baselines_dir") else None   # for rot scatter
    quantities = list(cfg.get("quantities", ["ne", "temp"]))
    order = [i for i in (cfg.get("conditions") or [c["id"] for c in man["conditions"]]) if i in conds]
    if not order:
        raise ValueError("errorcap_ladder: no matching conditions (check 'conditions' vs the manifest ids)")
    labels = cfg.get("condition_labels", {}) or {}
    titles = cfg.get("titles", {}) or {"ne": r"density $n_e$", "temp": r"temperature $T$"}
    plot = dict(cfg.get("plot", {}) or {})
    as_pct = bool(cfg.get("percent", True)); scale = 100.0 if as_pct else 1.0

    series = cfg.get("series") or [
        {"stat": "e_cap",     "label": "observed",      "color": "#1f77b4", "marker": "o", "ms": 8, "dx": 0.0},
        {"stat": "split",     "label": "split-seed",    "color": "#2ca02c", "marker": "s", "ms": 6, "dx": 0.14,
         "errorbar": True, "mfc": "none"},
        {"stat": "rot_null",  "label": "rotation null", "color": "#7f7f7f", "marker": "_", "ms": 16, "dx": -0.14},
        {"stat": "isotropic", "label": "isotropic ref", "color": "#d62728", "ls": "--", "hline": True},
    ]

    nq = len(quantities)
    fig, axs = plt.subplots(1, nq, figsize=tuple(plot.get("figsize", [5.4 * nq, 3.8])),
                            squeeze=False, sharey=bool(plot.get("sharey", False)))
    axs = axs[0]
    x = np.arange(len(order))
    for qi, q in enumerate(quantities):
        ax = axs[qi]
        # optional faint scatter of the 29 rotation-null values per condition (behind everything)
        if bool(cfg.get("rot_scatter", False)) and bdir is not None:
            for xi, cid in enumerate(order):
                rot = np.load(bdir / cid / "baselines.npz")[f"{q}_ecap_rot"] * scale
                ax.plot(np.full(rot.size, x[xi]), rot, ".", color=plot.get("rot_scatter_color", "0.8"),
                        ms=plot.get("rot_scatter_ms", 2.5), alpha=plot.get("rot_scatter_alpha", 0.6), zorder=0)
        for s in series:
            if bool(s.get("hline", False)):                                       # constant reference line
                vals = [conds[cid]["quantities"][q]["ecap"].get(
                            "e_cap_rand" if s["stat"] == "isotropic" else s["stat"], np.nan) for cid in order]
                ax.axhline(float(np.nanmean(vals)) * scale, color=s.get("color", "0.5"),
                           ls=s.get("ls", "--"), lw=s.get("lw", 1.0), label=s.get("label"), zorder=1)
                continue
            ys, elo, ehi = [], [], []
            for cid in order:
                v, lo, hi = _ecap_stat(conds[cid]["quantities"][q]["ecap"], s["stat"])
                ys.append(v * scale if v is not None else np.nan)
                elo.append(max(v - lo, 0.0) * scale if (v is not None and lo is not None) else np.nan)
                ehi.append(max(hi - v, 0.0) * scale if (v is not None and hi is not None) else np.nan)
            xx = x + float(s.get("dx", 0.0))
            kw = dict(color=s.get("color"), marker=s.get("marker", "o"), ls=s.get("ls", "none"),
                      ms=s.get("ms", 7), mew=s.get("mew", 1.3), mfc=s.get("mfc", s.get("color")),
                      label=s.get("label", s["stat"]), zorder=s.get("zorder", 3))
            if bool(s.get("errorbar", False)) and np.isfinite(elo).any():
                ax.errorbar(xx, ys, yerr=[np.nan_to_num(elo), np.nan_to_num(ehi)],
                            capsize=s.get("capsize", 3), elinewidth=s.get("elw", 1.2), **kw)
            else:
                ax.plot(xx, ys, **kw)
        ax.set_xticks(x); ax.set_xticklabels([labels.get(i, i) for i in order],
                        rotation=int(plot.get("tick_rotation", 30)), ha="right", fontsize=int(plot.get("tick_fontsize", 8)))
        ax.set_xlim(-0.5, len(order) - 0.5)
        if plot.get("ylim"): ax.set_ylim(*plot["ylim"])
        if plot.get("logy"): ax.set_yscale("log")
        if qi == 0 or not bool(plot.get("shared_ylabel", True)):
            ax.set_ylabel(cfg.get("ylabel", "error capture (%)" if as_pct else "error capture"),
                          fontsize=int(plot.get("ylabel_fontsize", 9)))
        ax.set_title(titles.get(q, q), fontsize=int(plot.get("title_fontsize", 10)))
        if qi == 0 or bool(plot.get("legend_all", False)):
            ax.legend(fontsize=int(plot.get("legend_fontsize", 7)), ncol=int(plot.get("legend_ncol", 1)),
                      loc=plot.get("legend_loc", "best"), framealpha=0.9)
    fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_errorcap_ladder"),
                    cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))

