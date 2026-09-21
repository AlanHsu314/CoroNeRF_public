from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ..benchmark.config import load_yaml
from ..util.figure import _resolve_path, _savefig

def _cfg(spec_path, key):
    spec_path = Path(spec_path).resolve()
    return load_yaml(spec_path).get(key, {}), spec_path.parent

def _outdir(cfg, spec_root, override):
    d = Path(override) if override else _resolve_path(cfg.get("output_dir", "../../paper_outputs/figures/uq"), spec_root)
    d.mkdir(parents=True, exist_ok=True); return d

def _spearman(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if a.size < 3: return np.nan
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

def _localization_rho(npz_path, q):
    d = np.load(npz_path)
    return _spearman(d[f"{q}_disagreement"], d[f"{q}_true_err"])

# ---------- [1] F6 ladder ----------
def fig_ladder(spec_path, device_override=None, output_dir_override=None):
    cfg, root = _cfg(spec_path, "ladder")
    man = json.load(open(_resolve_path(cfg["sweep_manifest"], root)))
    rows = man["rows"]
    if cfg.get("order"):                       # optional explicit id ordering (else manifest order)
        idx = {r["id"]: r for r in rows}; rows = [idx[i] for i in cfg["order"] if i in idx]
    out = _outdir(cfg, root, output_dir_override)
    mc = cfg.get("metric_colors", {"loc": "#1f77b4", "post": "#d62728", "crb": "#7f7f7f"})
    labels = cfg.get("condition_labels", {}) or {}       # id -> pretty x-tick label (LaTeX ok)
    fig, axes = plt.subplots(1, 2, figsize=tuple(cfg.get("figsize", [11, 4.2])), sharey=True)
    surro = cfg.get("surrogate_conditions")            # None -> post/CRB on all conditions; else only these ids
    show_sur = (lambda rid: True) if not surro else (lambda rid: rid in set(surro))
    for qi, q in enumerate(("ne", "temp")):
        ax = axes[qi]; fams = [r.get("family") for r in rows]
        seen = set()                                   # legend: first-of-series (robust to which x plots first)
        def _lab(key, text): return None if key in seen else (seen.add(key) or text)
        for x, r in enumerate(rows):
            b = (r.get("bridge") or {}).get(q, {}) or {}
            post = (b.get("sigma_vs_post") or {}).get("spearman")
            crb  = (b.get("sigma_vs_crb")  or {}).get("spearman")
            npz = r.get("disagreement_npz"); loc = _localization_rho(npz, q) if npz else np.nan
            sur = show_sur(r["id"])
            if sur and post is not None and crb is not None:
                ax.plot([x, x], [crb, post], color="0.75", lw=1.0, zorder=1)
            if np.isfinite(loc): ax.scatter(x, loc, marker="o", s=55, color=mc["loc"], zorder=3, label=_lab("loc", "localization $\\rho_\\epsilon$"))
            if sur and post is not None: ax.scatter(x, post, marker="D", s=42, color=mc["post"], zorder=3, label=_lab("post", "post $\\rho$"))
            if sur and crb is not None:  ax.scatter(x, crb, marker="x", s=45, color=mc["crb"], zorder=3, label=_lab("crb", "CRB $\\rho$"))
        for x in range(1, len(rows)):
            if fams[x] != fams[x-1]: ax.axvline(x-0.5, color="0.9", lw=1)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([labels.get(r["id"], r["id"]) for r in rows], rotation=45, ha="right", fontsize=cfg.get('xticks_fontsize', 8))
        ax.set_title(q); ax.set_ylabel(r"Spearman $\rho$" if qi == 0 else "")
    axes[-1].legend(loc=cfg.get('legend_loc', "lower left"), fontsize=cfg.get('legend_fontsize', 8), framealpha=0.9)
    fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_ladder"), cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))

# ---------- [5] F7 operating heatmaps (selectable metric list) ----------
def fig_operating_heatmaps(spec_path, device_override=None, output_dir_override=None):
    cfg, root = _cfg(spec_path, "opheat")
    rows = json.load(open(_resolve_path(cfg["sweep_manifest"], root)))["rows"]
    out = _outdir(cfg, root, output_dir_override)
    V = cfg.get("views") or sorted({int(r["views"]) for r in rows if r.get("views") is not None})
    E = cfg.get("eta")   or sorted({float(r["eta"]) for r in rows if r.get("eta") is not None})
    LOG = set(cfg.get("log_metrics", ["crb_std_median", "post_std_median", "condition_tau", "condition_full"]))
    paths = {}
    for metric in cfg.get("metrics", ["eff_rank_tau"]):        # <-- selectable list -> one plot each
        G = np.full((len(E), len(V)), np.nan); tr = np.zeros_like(G, bool)
        for r in rows:
            try: i = E.index(float(r["eta"])); j = V.index(int(r["views"]))
            except (ValueError, KeyError, TypeError): continue
            v = r.get(metric)
            if v is not None and np.isfinite(v): G[i, j] = float(v)
            if r.get("disagreement_npz"): tr[i, j] = True
        disp = np.log10(G) if metric in LOG else G
        fig, ax = plt.subplots(figsize=(1.15*len(V)+2.0, 0.95*len(E)+1.4))
        im = ax.imshow(disp, origin="lower", aspect="auto", cmap=cfg.get("cmap", "viridis"))
        ax.set_xticks(range(len(V))); ax.set_xticklabels(V); ax.set_yticks(range(len(E))); ax.set_yticklabels([f"{e:g}" for e in E])
        ax.set_xlabel("# views"); ax.set_ylabel(r"noise multiplier $\eta$")
        for i in range(len(E)):
            for j in range(len(V)):
                if np.isfinite(G[i, j]): ax.text(j, i, f"{G[i,j]:.0f}" if metric == "eff_rank_tau" else f"{G[i,j]:.2g}", ha="center", va="center", color="w", fontsize=8)
                if tr[i, j]: ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec="red", lw=2.0))
        fig.colorbar(im, ax=ax, label=(f"log10 {metric}" if metric in LOG else metric)); ax.set_title(metric); fig.tight_layout()
        paths[metric] = _savefig(fig, out / f"heatmap_{metric}", cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))
    return paths

# ---------- [3] F5 identifiability (2 rows: spectra | canonical maps) ----------

def _draw_projection_row(fig, subspecs, panels, mp_npz, tau, plot):
    """Draw the requested projection panels (any subset/order of energy | cumulative | bar).
    cumulative plots the selected ens/err/rand curves; each crosses tau at its own f_null."""
    mp = np.load(mp_npz)
    s = np.asarray(mp["s"]); g = np.asarray(mp["g_ens"])
    ce = np.asarray(mp["c_eps"]) if "c_eps" in mp.files else None          # err projections v_l^T eps
    fn = {"ens": float(mp["f_null_ens"]), "err": float(mp["f_null_err"]), "rand": float(mp["f_null_rand"])}
    col = {"ens": "#009E73", "err": "#E69F00", "rand": "#7f7f7f"}
    order = np.argsort(s); ss = s[order]; null = s < tau
    M = int(mp["M"]) if "M" in mp.files else s.size; n_struct = max(M - s.size, 0)
    def _series(e, f):
        e = np.asarray(e, float); comp = e.sum(); en = e[null].sum(); den = f - 1.0
        out = 0.0 if abs(den) < 1e-12 else max((en - f * comp) / den, 0.0)
        return (out + np.cumsum(e[order])) / max(comp + out, 1e-30)
    cum = {"ens": _series(g, float(mp["f_null_ens"])),
           "rand": (n_struct + np.cumsum(np.ones_like(s)[order])) / M}
    if ce is not None:
        cum["err"] = _series(ce ** 2, float(mp["f_null_err"]))

    def _energy(ax):
        ax.loglog(np.clip(s, 1e-12, None), np.clip(g, 1e-30, None), ".", ms=3, alpha=0.4, color=col["ens"])
        ax.axvline(tau, color="r", ls="--", lw=1)
        ax.set_xlabel(r"singular value $s_\ell$"); ax.set_ylabel(r"ensemble energy $g_\ell$")
        if plot.get("gs_xlim"): ax.set_xlim(plot["gs_xlim"][0], plot["gs_xlim"][1])

    def _cumulative(ax):
        for c in [c for c in plot.get("cumulative_curves", ["ens", "err", "rand"]) if c in cum]:
            ax.semilogx(np.clip(ss, 1e-12, None), cum[c], lw=1.8, color=col[c],
                        label=f"{c} ($f_{{\\rm null}}$={fn[c]:.2f})")
        ax.axvline(tau, color="r", ls="--", lw=1); ax.set_ylim(0, 1)
        ax.set_xlabel(r"singular value $s_\ell$"); ax.set_ylabel("cumulative energy fraction")
        ax.legend(fontsize=9, loc="lower right")
        if plot.get("gs_xlim"): ax.set_xlim(plot["gs_xlim"][0], plot["gs_xlim"][1])

    def _bar(ax):
        ax.bar(["ens", "err", "rand"], [fn["ens"], fn["err"], fn["rand"]],
               color=[col["ens"], col["err"], col["rand"]]); ax.set_ylim(0, 1)
        ax.set_ylabel("null-space fraction"); ax.set_title(f"$f_{{\\rm null}}$ = {fn['ens']:.2f}", fontsize=9)

    draw = {"energy": _energy, "cumulative": _cumulative, "bar": _bar}
    for spec, name in zip(subspecs, panels):
        draw[name](fig.add_subplot(spec))

def fig_identifiability(spec_path, device_override=None, output_dir_override=None):
    cfg, root = _cfg(spec_path, "identifiability")
    out = _outdir(cfg, root, output_dir_override)
    sw = Path(_resolve_path(cfg["sweep_dir"], root))          # runs_identifiability/sweeps/<name>
    def load_an(cid): return np.load(sw / "conditions" / cid / "analysis.npz")     # from sweep save (patch 2)

    # ---- side-by-side layout: spectrum (left) + projection panel(s) (right) ----
    # kind of a hack, what the current paper uses
    if str(cfg.get("layout", "stacked")) == "row":
        from ..identifiability._core import _plot_spectrum
        tau = float(cfg.get("tau", 1.0))
        panels = list(cfg.get("projection_panels", ["cumulative"]))
        ncol = 1 + len(panels)
        fig = plt.figure(figsize=tuple(cfg.get("figsize", [4.7 * ncol, 3.6])))
        gs = fig.add_gridspec(1, ncol, width_ratios=cfg.get("width_ratios", [1.4] + [1.0] * len(panels)),
                              wspace=cfg.get("wspace", 0.30))
        can = cfg.get("canonical", "noise_x9")
        row = next(r for r in json.load(open(sw / "sweep_manifest.json"))["rows"] if r["id"] == can)
        res = {"singular_values": np.sort(load_an(can)["singular_values"])[::-1], "tau": tau,
               "eff_rank_tau": row["eff_rank_tau"], "eff_rank_participation": row["eff_rank_participation"],
               "condition_tau": row["condition_tau"]}
        ax0 = fig.add_subplot(gs[0, 0]); _plot_spectrum(res, cfg.get("plot", {}) or {}, ax=ax0)
        ax0.set_ylim(cfg.get("spectrum_ylim", [1e-4, 1e4]))
        mp = _resolve_path(cfg["mode_projection_npz"], root)
        _draw_projection_row(fig, [gs[0, 1 + j] for j in range(len(panels))], panels, mp, tau, dict(cfg.get("plot", {}) or {}))
        #fig.tight_layout()
        return _savefig(fig, out / cfg.get("name", "fig_identifiability"),
                        cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))

    bottom = str(cfg.get("bottom", "maps"))                       # maps | projection | both
    n_bot = {"maps": 1, "projection": 1, "both": 2}[bottom]
    fig = plt.figure(figsize=tuple(cfg.get("figsize", [11, 7 if n_bot == 1 else 10])))
    gs = fig.add_gridspec(1 + n_bot, 3, height_ratios=cfg.get("height_ratios", [1.0] + [1.1] * n_bot))
    
    # row 1: overlaid spectra (spans 3 cols)
    axs = fig.add_subplot(gs[0, :])
    tau = float(cfg.get("tau", 1.0))
    if str(cfg.get("spectrum_mode", "multi")) == "single":
        from ..identifiability._core import _plot_spectrum
        can = cfg.get("canonical", "v300_n9")
        row = next(r for r in json.load(open(sw / "sweep_manifest.json"))["rows"] if r["id"] == can)
        res = {"singular_values": np.sort(load_an(can)["singular_values"])[::-1], "tau": tau,
               "eff_rank_tau": row["eff_rank_tau"],
               "eff_rank_participation": row["eff_rank_participation"],
               "condition_tau": row["condition_tau"]}
        _plot_spectrum(res, cfg.get("plot", {}) or {}, ax=axs)
        axs.set_ylim(cfg.get("spectrum_ylim", [0.0001, 10000])) # some deep null space modes have very small singular values
    else:
        for cid, lab in cfg["spectrum_conditions"].items():
            s = load_an(cid)["singular_values"]
            axs.semilogy(np.arange(1, s.size + 1), np.sort(s)[::-1], label=lab)
        axs.axhline(tau, ls="--", color="0.5", lw=1, label=r"$\tau=1$")
        axs.set_xlabel("index"); axs.set_ylabel("singular value"); axs.legend(fontsize=8, ncol=2)

    # bottom row(s)
    def _draw_maps_row(rr):
        can = cfg.get("canonical", "noise_x9"); q = cfg.get("map_quantity", "ne")
        d = load_an(can); qi = list(d["quantities"]).index(q)
        ri = int(np.argmin(np.abs(d["r_axis"] - float(cfg.get("map_radius", 1.5)))))
        lon = np.rad2deg(d["lon_axis"]); lat = np.rad2deg(d["phi_axis"]); ext = [lon.min(), lon.max(), lat.min(), lat.max()]
        panels = [("crb_std_grid", r"$\operatorname{crb}(\mathbf{x})$", "magma", True),
                  ("post_std_grid", r"$\operatorname{post}(\mathbf{x})$", "inferno", True),
                  ("prior_logratio_grid", r"$\operatorname{pdom}(\mathbf{x})$", "viridis", False)]
        for k, (key, title, cm, is_log) in enumerate(panels):
            ax = fig.add_subplot(gs[rr, k]); M = d[key][qi, :, :, ri].T
            D = np.log10(np.clip(M, 1e-30, None)) if is_log else M          # <-- log fix (matches oppoints)
            vmin, vmax = np.nanpercentile(D, (cfg.get("plot", {}) or {}).get("percentile", [2, 98]))
            im = ax.imshow(D, origin="lower", aspect="auto", extent=ext, cmap=cm, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, fraction=0.046)

    r_next = 1
    if bottom in ("projection", "both"):
        mp = _resolve_path(cfg["mode_projection_npz"], root)
        panels = list(cfg.get("projection_panels", ["energy", "cumulative"]))   # subset/order of energy|cumulative|bar
        sub = gs[r_next, :].subgridspec(1, len(panels), wspace=cfg.get("projection_wspace", 0.35))
        _draw_projection_row(fig, [sub[0, j] for j in range(len(panels))], panels, mp, tau, dict(cfg.get("plot", {}) or {}))
        r_next += 1
    if bottom in ("maps", "both"):
        _draw_maps_row(r_next)
    fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_identifiability"),
                    cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))
    
    # row 2: canonical maps (CRB | posterior | prior-dominance) for one quantity
    # can = cfg.get("canonical", "v300_n9"); q = cfg.get("map_quantity", "ne")
    # d = load_an(can); qi = list(d["quantities"]).index(q)
    # ri = int(np.argmin(np.abs(d["r_axis"] - float(cfg.get("map_radius", 1.5)))))
    # lon = np.rad2deg(d["lon_axis"]); lat = np.rad2deg(d["phi_axis"]); ext = [lon.min(), lon.max(), lat.min(), lat.max()]
    # panels = [("crb_std_grid",       r"$\operatorname{crb}(\mathbf{x})$",  "magma"),
    #           ("post_std_grid",       r"$\operatorname{post}(\mathbf{x})$", "inferno"),
    #           ("prior_logratio_grid", r"$\operatorname{pdom}(\mathbf{x})$", "viridis")]
    # from ..identifiability._core import _upsample_shell
    # up = int((cfg.get("plot", {}) or {}).get("upsample", 0) or 0)
    # for k, (key, title, cm) in enumerate(panels):
    #     ax = fig.add_subplot(gs[1, k])
    #     M = (_upsample_shell(d[key][qi, :, :, ri], d["lon_axis"], d["phi_axis"], n=up)
    #          if up > 0 else d[key][qi, :, :, ri].T)
    #     im = ax.imshow(M, origin="lower", aspect="auto", extent=ext, cmap=cm)
    #     ax.set_title(title, fontsize=int((cfg.get("plot", {}) or {}).get("panel_title_fontsize", 10)))
    #     fig.colorbar(im, ax=ax, fraction=0.046)

    # fig.tight_layout()
    # return _savefig(fig, out / cfg.get("name", "fig_identifiability"),
    #                 cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))

# paper ensemble panel, can customize which ones you want from GT, empirical, analytical
def fig_ensemble_panels(spec_path, device_override=None, output_dir_override=None):
    cfg, root = _cfg(spec_path, "ensemble_panels")
    out = _outdir(cfg, root, output_dir_override)
    dz = np.load(_resolve_path(cfg["disagreement_npz"], root))                     # ensemble shells
    az = np.load(_resolve_path(cfg["analytic_npz"], root)) if cfg.get("analytic_npz") else None
    ri, aq = None, None
    if az is not None:
        ri = int(np.argmin(np.abs(np.asarray(az["r_axis"]) - float(cfg.get("map_radius", 1.5)))))
        aq = list(az["quantities"])
    quantities = list(cfg.get("quantities", ["ne", "temp"]))                       # rows (["ne"] to drop T)
    columns    = list(cfg.get("columns", ["ens_mean", "ens_err", "disagreement", "post"]))
    plot = dict(cfg.get("plot", {}) or {}); pct = plot.get("percentile", [2, 98])
    interp = plot.get("analytic_interp", "nearest")     # "bilinear" de-blocks the coarse crb/post/pdom grids

    # so that colorbars can move
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cbar_cols = set(plot.get("colorbar_columns", columns))   # columns that get a colorbar (default: all)
    cbar_pos = str(plot.get("colorbar_position", "bottom"))  # "bottom" (large maps) | "right"
    
    # name -> (source, key, label, cmap, log)
    COLS = {
        "gt":           ("ens", "{q}_gt",            "GT",                                 plot.get("field_cmap", "magma"),  False),
        "ens_mean":     ("ens", "{q}_ens_mean",      r"ensemble mean $\bar{m}(\mathbf{x})$",                      plot.get("field_cmap", "magma"),  False),
        "ens_err":      ("ens", "{q}_true_err",      r"ensemble mean error $\epsilon_{\rm mean}(\mathbf{x})$",                "viridis",                        False),
        "disagreement": ("ens", "{q}_disagreement",  r"cross-seed instability $\sigma_{\rm ens}(\mathbf{x})$",   "cividis",                        False),
        "crb":          ("an",  "crb_std_grid",      r"$\operatorname{crb}(\mathbf{x})$",  "magma",                          True),
        "post":         ("an",  "post_std_grid",     r"$\operatorname{post}(\mathbf{x})$", "inferno",                        True),
        "pdom":         ("an",  "prior_logratio_grid", r"$\operatorname{pdom}(\mathbf{x})$", "viridis",                      False),
    }
    qlab = {"ne": r"$n_{\rm e}$", "temp": r"$T_{\rm e}$"}
    nrow, ncol = len(quantities), len(columns)
    fig, axs = plt.subplots(nrow, ncol, figsize=tuple(plot.get("figsize", [3.0 * ncol, 2.7 * nrow])), squeeze=False)

    # adjust space
    wspace, hspace = plot.get('wspace', 0.025), plot.get('hspace', 0.1)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # main loop
    for r, q in enumerate(quantities):
        # shared color limits for the "field-value" columns (gt, ens_mean) so they're comparable
        field_cols = [c for c in columns if c in ("gt", "ens_mean")]
        fclim = None
        if field_cols:
            fv = np.concatenate([np.asarray(dz[COLS[c][1].format(q=q)]).ravel() for c in field_cols])
            fv = fv[np.isfinite(fv)]; fclim = tuple(np.percentile(fv, pct)) if fv.size else None
        for c, name in enumerate(columns):
            src, keyt, label, cmap, is_log = COLS[name]; ax = axs[r][c]
            if src == "ens":
                M = np.asarray(dz[keyt.format(q=q)])
                lon = np.asarray(dz[f"{q}_lon_deg"]); lat = np.asarray(dz[f"{q}_lat_deg"])
            else:
                if az is None: ax.axis("off"); continue
                M = np.asarray(az[keyt])[aq.index(q), :, :, ri].T
                lon = np.rad2deg(np.asarray(az["lon_axis"])); lat = np.rad2deg(np.asarray(az["phi_axis"]))
            ext = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
            D = np.log10(np.clip(M, 1e-30, None)) if is_log else M
            if name in ("gt", "ens_mean") and fclim is not None: vmin, vmax = fclim
            else: vmin, vmax = np.nanpercentile(D, pct)
            cm = plt.get_cmap(cmap).copy(); cm.set_bad(plot.get("bad_color", "white"))
            im = ax.imshow(D, origin="lower", aspect="auto", extent=ext, cmap=cm, vmin=vmin, vmax=vmax,
                           interpolation=(interp if src == "an" else "nearest"))
            # here we just lazy-interpolate on imshow, but if you want to be more rigorous, make sure to do it on the actual sphere via _core._upsample_shell
            if r == 0: ax.set_title(label, fontsize=plot.get("title_fontsize", 9))
            if c == 0: ax.set_ylabel(qlab.get(q, q), fontsize=plot.get("label_fontsize", 11))
            ax.set_xticks([]); ax.set_yticks([])
            if name in cbar_cols:
                cax = make_axes_locatable(ax).append_axes(cbar_pos, size=plot.get("cbar_size", "5%"),
                                                          pad=plot.get("cbar_pad", 0.05))
                orient = "horizontal" if cbar_pos == "bottom" else "vertical"
                fig.colorbar(im, cax=cax, orientation=orient).ax.tick_params(labelsize=plot.get("cbar_fontsize", 7))
    #fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_ensemble_panels"), cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))

def fig_ensemble_slice_composite(spec_path, device_override=None, output_dir_override=None):
    """Figure-5 ensemble panels in two geometries: lat-lon shell (top) over lat-r meridional slice (bottom).
    Mirrors fig_ensemble_panels exactly (same COLS, clim, analytic handling); the slice row also renders the
    analytic post/crb/pdom by slicing analysis.npz at a fixed longitude. Reads:
      - disagreement_npz : seed-disagreement arrays.npz (shell keys + {q}_slice_* keys from the slice emit)
      - analytic_npz     : identifiability analysis.npz (post_std_grid/crb_std_grid/prior_logratio_grid)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cfg, root = _cfg(spec_path, "ensemble_slice")
    out = _outdir(cfg, root, output_dir_override)
    dz = np.load(_resolve_path(cfg["disagreement_npz"], root))
    az = np.load(_resolve_path(cfg["analytic_npz"], root)) if cfg.get("analytic_npz") else None
    q = str(cfg.get("quantity", "ne"))
    columns = list(cfg.get("columns", ["ens_mean", "ens_err", "disagreement", "post"]))
    plot = dict(cfg.get("plot", {}) or {}); pct = plot.get("percentile", [2, 98])
    interp = plot.get("analytic_interp", "nearest")                 # bilinear de-blocks the coarse analytic grid

    # layout knobs. "readable" (default): right colorbars + r ticks on the slice row.
    #               "mimic": bottom colorbars + tick-free maps == two Figure-5 rows stacked.
    layout   = str(plot.get("layout", "readable"))
    cbar_pos = str(plot.get("colorbar_position", "bottom" if layout == "mimic" else "right"))
    show_r   = bool(plot.get("slice_r_ticks", layout != "mimic"))
    cbar_cols = set(plot.get("colorbar_columns", columns))
    row_labels = list(plot.get("row_labels", ["lat-lon shell", "lat-r slice"]))

    # analytic indices: shell radius (ri) + slice longitude (lon_i)
    aq = list(az["quantities"]) if az is not None else None
    ri = int(np.argmin(np.abs(np.asarray(az["r_axis"]) - float(cfg.get("map_radius", 1.5))))) if az is not None else None
    lon0 = (float(cfg["slice_longitude"]) if cfg.get("slice_longitude") is not None
            else float(dz["slice_longitude"]) if "slice_longitude" in dz.files else 0.0)
    lon_i = int(np.argmin(np.abs(np.rad2deg(np.asarray(az["lon_axis"])) - lon0))) if az is not None else None

    # identical to fig_ensemble_panels: name -> (source, key, label, cmap, is_log)
    COLS = {
        "gt":           ("ens", "{q}_gt",            "GT",                                                     plot.get("field_cmap", "magma"), False),
        "ens_mean":     ("ens", "{q}_ens_mean",      r"ensemble mean $\bar{m}(\mathbf{x})$",                   plot.get("field_cmap", "magma"), False),
        "ens_err":      ("ens", "{q}_true_err",      r"ensemble mean error $\epsilon_{\rm mean}(\mathbf{x})$", "viridis", False),
        "disagreement": ("ens", "{q}_disagreement",  r"disagreement $\sigma_{\rm ens}(\mathbf{x})$",           "cividis", False),
        "crb":          ("an",  "crb_std_grid",      r"$\operatorname{crb}(\mathbf{x})$",                      "magma",   True),
        "post":         ("an",  "post_std_grid",     r"$\operatorname{post}(\mathbf{x})$",                     "inferno", True),
        "pdom":         ("an",  "prior_logratio_grid", r"$\operatorname{pdom}(\mathbf{x})$",                   "viridis", False),
    }

    ncol = len(columns)
    fig, axs = plt.subplots(2, ncol, figsize=tuple(plot.get("figsize", [3.0 * ncol, 4.6])), squeeze=False)
    fig.subplots_adjust(wspace=plot.get("wspace", 0.025),
                        hspace=plot.get("hspace", 0.32 if cbar_pos == "bottom" else 0.14))

    # ---- fetch one cell (M, extent) for each geometry ----
    def _shell(name):
        src, keyt = COLS[name][0], COLS[name][1]
        if src == "ens":
            M = np.asarray(dz[keyt.format(q=q)])
            lon = np.asarray(dz[f"{q}_lon_deg"]); lat = np.asarray(dz[f"{q}_lat_deg"])
        else:
            if az is None: return None, None
            M = np.asarray(az[keyt])[aq.index(q), :, :, ri].T                 # (lon, phi) -> (lat, lon)
            lon = np.rad2deg(np.asarray(az["lon_axis"])); lat = np.rad2deg(np.asarray(az["phi_axis"]))
        return M, [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    def _slice(name):
        src, keyt = COLS[name][0], COLS[name][1]
        if src == "ens":
            slk = keyt.format(q=q).replace(f"{q}_", f"{q}_slice_", 1)         # ne_ens_mean -> ne_slice_ens_mean
            if slk not in dz.files: return None, None
            M = np.asarray(dz[slk])
            rr = np.asarray(dz[f"{q}_slice_r_axis"]); la = np.asarray(dz[f"{q}_slice_lat_deg"])
        else:
            if az is None: return None, None
            M = np.asarray(az[keyt])[aq.index(q), lon_i, :, :]                 # (phi, r) == (lat, r), no transpose
            rr = np.asarray(az["r_axis"]); la = np.rad2deg(np.asarray(az["phi_axis"]))
        return M, [float(rr.min()), float(rr.max()), float(la.min()), float(la.max())]

    # shared field-value clim across gt/ens_mean AND both geometries (mirror of the fig_ensemble_panels fclim)
    field_cols = [c for c in columns if c in ("gt", "ens_mean")]
    fclim = None
    if field_cols:
        parts = []
        for c in field_cols:
            shk = COLS[c][1].format(q=q); parts.append(np.asarray(dz[shk]).ravel())
            slk = shk.replace(f"{q}_", f"{q}_slice_", 1)
            if slk in dz.files: parts.append(np.asarray(dz[slk]).ravel())
        fv = np.concatenate(parts); fv = fv[np.isfinite(fv)]
        fclim = tuple(np.percentile(fv, pct)) if fv.size else None

    for c, name in enumerate(columns):
        src, keyt, label, cmap, is_log = COLS[name]
        cm = plt.get_cmap(cmap).copy(); cm.set_bad(plot.get("bad_color", "white"))
        Msh, ext_sh = _shell(name); Msl, ext_sl = _slice(name)
        Dsh = np.log10(np.clip(Msh, 1e-30, None)) if (is_log and Msh is not None) else Msh
        Dsl = np.log10(np.clip(Msl, 1e-30, None)) if (is_log and Msl is not None) else Msl
        # one clim per column, shared across the two rows so shell and slice are directly comparable
        if name in ("gt", "ens_mean") and fclim is not None:
            vmin, vmax = fclim
        else:
            pool = [d[np.isfinite(d)].ravel() for d in (Dsh, Dsl) if d is not None]
            pool = np.concatenate(pool) if pool else np.array([])
            vmin, vmax = (tuple(np.percentile(pool, pct)) if pool.size else (None, None))
        for rrow, (D, ext) in enumerate([(Dsh, ext_sh), (Dsl, ext_sl)]):
            ax = axs[rrow][c]
            if D is None: ax.axis("off"); continue                            # analytic missing -> blank cell
            im = ax.imshow(D, origin="lower", aspect="auto", extent=ext, cmap=cm, vmin=vmin, vmax=vmax,
                           interpolation=(interp if src == "an" else "nearest"))
            if rrow == 0:
                ax.set_title(label, fontsize=plot.get("title_fontsize", 9)); ax.set_xticks([])
            elif show_r:
                ax.set_xlabel(r"$r\ [R_\odot]$", fontsize=plot.get("label_fontsize", 9))
                ax.tick_params(axis="x", labelsize=plot.get("tick_fontsize", 7))
            else:
                ax.set_xticks([])
            if c == 0: ax.set_ylabel(row_labels[rrow], fontsize=plot.get("label_fontsize", 9))
            ax.set_yticks([])
            if name in cbar_cols:
                cax = make_axes_locatable(ax).append_axes(cbar_pos, size=plot.get("cbar_size", "5%"),
                                                          pad=plot.get("cbar_pad", 0.05))
                orient = "horizontal" if cbar_pos == "bottom" else "vertical"
                fig.colorbar(im, cax=cax, orientation=orient).ax.tick_params(labelsize=plot.get("cbar_fontsize", 7))

    # keep the slice row on one common r-range (analytic grid may cover a sub-interval)
    if plot.get("slice_share_xlim", True) and f"{q}_slice_r_axis" in dz.files:
        rr = np.asarray(dz[f"{q}_slice_r_axis"]); xl = (float(rr.min()), float(rr.max()))
        for c in range(ncol):
            if axs[1][c].has_data(): axs[1][c].set_xlim(*xl)

    return _savefig(fig, out / cfg.get("name", "fig_ensemble_slice"),
                    cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))

# operating point compact figure for the paper
def fig_oppoint(spec_path, device_override=None, output_dir_override=None):
    """Appendix: compact deployment operating-point figure for ONE condition — 3 rows (crb/post/pdom)
    x 4 cols (m*, mean-m̂, cross-m̂ std, scatter). Reads deployment_<cid>.npz from oppoints. Verified render."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cfg, root = _cfg(spec_path, "oppoint")
    out = _outdir(cfg, root, output_dir_override)
    if cfg.get("deployment_npz"):
        dep = Path(_resolve_path(cfg["deployment_npz"], root))
    else:
        dep = Path(_resolve_path(cfg["oppoints_dir"], root)) / f"deployment_{cfg.get('condition', 'noise_x9')}.npz"
    d = np.load(dep)
    plot = dict(cfg.get("plot", {}) or {})
    interp = plot.get("interpolation", "nearest")            # "bilinear" to de-block the coarse maps
    pct = plot.get("percentile", [2, 98])
    aq = list(d["quantities"]); q = str(cfg.get("quantity", "ne")); qi = aq.index(q)
    ri = int(np.argmin(np.abs(np.asarray(d["r_axis"]) - float(cfg.get("map_radius", 1.5)))))

    def _rho(A, B):
        a = np.asarray(A, float).ravel(); b = np.asarray(B, float).ravel()
        ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        if a.size < 3: return float("nan")
        return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

    rows = [("crb", "CRB", plot.get("crb_cmap", "magma"),  True),
            ("post", "post", plot.get("post_cmap", "inferno"), True),
            ("pdom", "pdom", plot.get("pdom_cmap", "viridis"), False)]
    fig = plt.figure(figsize=tuple(plot.get("figsize", [10.5, 6.6])))
    gs = fig.add_gridspec(3, 4, hspace=plot.get("hspace", 0.45), wspace=plot.get("wspace", 0.32))
    
    for r, (key, lab, cmap, is_log) in enumerate(rows):
        t = (lambda M: np.log10(np.clip(M, 1e-30, None))) if is_log else (lambda M: M)
        A = t(d[f"{key}_mstar"][qi, :, :, ri].T)
        B = t(d[f"{key}_mhat_mean"][qi, :, :, ri].T)
        Dg = t(d[f"{key}_mhat_std"][qi, :, :, ri].T)
        vmin, vmax = np.nanpercentile(A, pct[0]), np.nanpercentile(A, pct[1])
        panels = [(A,  rf"{lab} $m^*$",                     (vmin, vmax)),
                  (B,  rf"{lab} $\overline{{\hat m}}$",     (vmin, vmax)),
                  (Dg, rf"{lab} cross-$\hat m$ std",        (None, None))]
        for c, (M, ttl, vl) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            cm = plt.get_cmap(cmap).copy(); cm.set_bad(plot.get("bad_color", "white"))
            im = ax.imshow(M, origin="lower", aspect="auto", cmap=cm, vmin=vl[0], vmax=vl[1], interpolation=interp)
            ax.set_title(ttl, fontsize=plot.get("title_fontsize", 9)); ax.set_xticks([]); ax.set_yticks([])
            cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
            fig.colorbar(im, cax=cax).ax.tick_params(labelsize=plot.get("cbar_fontsize", 6))
        ax = fig.add_subplot(gs[r, 3])
        a = d[f"{key}_mstar"][qi].ravel(); b = d[f"{key}_mhat_mean"][qi].ravel()
        ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        if is_log: a, b = np.log10(np.clip(a, 1e-30, None)), np.log10(np.clip(b, 1e-30, None))
        lim = [np.nanpercentile(a, 1), np.nanpercentile(a, 99)]
        ax.scatter(a, b, s=2, alpha=0.1); ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(rf"$\overline{{\hat m}}$ vs $m^*$ ($\rho$={_rho(b, a):.3f})", fontsize=plot.get("title_fontsize", 9))
        ax.set_xlabel(r"$m^*$", fontsize=8); ax.set_ylabel(r"$\overline{\hat m}$", fontsize=8); ax.tick_params(labelsize=7)
    if plot.get("suptitle", False):
        fig.suptitle(cfg.get("title", f"deployment vs oracle — {q}"), fontsize=11)
    # no tight_layout (conflicts with append_axes); _savefig crops with bbox_inches="tight"
    return _savefig(fig, out / cfg.get("name", "fig_oppoint"),
                    cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))


# calibration curve (scatter of err vs sigma_ens)
def fig_calibration(spec_path, device_override=None, output_dir_override=None):
    """Appendix: calibration of sigma_ens as an error predictor. Per quantity: scatter sigma_ens vs
    |ensemble-mean error| + quantile-binned calibration (reliability) curve + y=x. Reads seed-disagreement npz.
    Verified reproduces the deprecated diagnostic's structure on current data."""
    cfg, root = _cfg(spec_path, "calibration")
    out = _outdir(cfg, root, output_dir_override)
    d = np.load(_resolve_path(cfg["disagreement_npz"], root))
    plot = dict(cfg.get("plot", {}) or {})
    quantities = list(cfg.get("quantities", ["ne", "temp"]))
    n_bins = int(cfg.get("calibration_bins", 20))
    alpha = float(plot.get("scatter_alpha", 0.05)); max_pts = int(plot.get("scatter_max_points", 200000))
    scatter_color = plot.get("scatter_color", "#009E73")
    rng = np.random.default_rng(0); qlab = {"ne": r"$n_e$", "temp": r"$T$"}

    def _corr(x, y):
        ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
        if x.size < 3: return float("nan"), float("nan"), x, y
        sp = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
        pe = float(np.corrcoef(x, y)[0, 1]); return sp, pe, x, y

    n = len(quantities)
    fig, axs = plt.subplots(1, n, figsize=tuple(plot.get("figsize", [5.0 * n, 4.2])), squeeze=False, constrained_layout = True); axs = axs[0]
    fig.set_constrained_layout_pads(
        wspace=plot.get("wspace", 0.02), hspace=plot.get("hspace", 0.02),
        w_pad=plot.get("w_pad", 0.02),   h_pad=plot.get("h_pad", 0.02),
    )
    for ax, q in zip(axs, quantities):
        x = np.asarray(d[f"{q}_disagreement"]).reshape(-1)
        y = np.abs(np.asarray(d[f"{q}_true_err"]).reshape(-1))     # magnitude calibration
        sp, pe, x, y = _corr(x, y)
        xp, yp = x, y
        if x.size > max_pts:
            idx = rng.choice(x.size, size=max_pts, replace=False); xp, yp = x[idx], y[idx]
        ax.scatter(xp, yp, s=2, color = scatter_color, alpha=alpha, rasterized=True)
        if x.size >= n_bins * 5:                                    # quantile-bin reliability curve
            edges = np.unique(np.nanpercentile(x, np.linspace(0, 100, n_bins + 1)))
            cx, cy = [], []
            for i in range(len(edges) - 1):
                m = (x >= edges[i]) & (x <= edges[i + 1])
                if np.any(m): cx.append(float(x[m].mean())); cy.append(float(y[m].mean()))
            ax.plot(cx, cy, "-o", color="k", ms=3, lw=1.2, label="calibration")
        lim = float(cfg.get("lim", max(np.nanmax(x) if x.size else 1.0, np.nanmax(y) if y.size else 1.0)))
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
        ax.set_xlabel(r"ensemble disagreement $\sigma_{\rm ens}$ (dex)")
        ax.set_ylabel(r"$|$ensemble-mean error$|$ (dex)")
        ax.set_title(f"{qlab.get(q, q)}  |  Spearman={sp:.3f}  Pearson={pe:.3f}", fontsize=plot.get("title_fontsize", 10))
        ax.legend(fontsize=7, loc="lower right")
    if plot.get("suptitle", True):
        fig.suptitle(cfg.get("title", r"Calibration of $\sigma_{\rm ens}$ as an error predictor"), fontsize=11)
    #fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_calibration"),
                    cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))

# nullspace projections, mirrors sparsification style 4-panel in uq_baselines.py
def _cum_curves(d, tau):
    """Cumulative energy fractions over ALL M modes (incl. structural null beyond the row rank),
    so each curve reaches its stored f_null at tau. Returns (ss, {ens,err,rand})."""

    "note that this is patched here, but still old in the original diagnostic plots in paperF results"
    s = np.asarray(d["s"]); g = np.asarray(d["g_ens"]); M = int(d["M"]) if "M" in d.files else s.size
    order = np.argsort(s); ss = s[order]; null = s < tau; n_struct = max(M - s.size, 0)
    def _series(e, f_null):                                  # e = per-computed-mode energy
        e = np.asarray(e, float); comp = e.sum(); en_null = e[null].sum(); den = f_null - 1.0
        outside = 0.0 if abs(den) < 1e-12 else max((en_null - f_null * comp) / den, 0.0)   # structural-null energy
        return (outside + np.cumsum(e[order])) / max(comp + outside, 1e-30)
    cum = {"ens": _series(g, float(d["f_null_ens"])),
           "rand": (n_struct + np.cumsum(np.ones_like(s)[order])) / M}
    if "c_eps" in d.files:
        cum["err"] = _series(np.asarray(d["c_eps"]) ** 2, float(d["f_null_err"]))
    return ss, cum

def fig_fnull_curves(spec_path, device_override=None, output_dir_override=None):
    """Grid of cumulative null-space energy curves (ens/err/rand vs singular value), one panel per condition."""
    cfg, root = _cfg(spec_path, "fnull_curves")
    out = _outdir(cfg, root, output_dir_override)
    mdir = Path(_resolve_path(cfg["mode_projection_dir"], root))
    conds = list(cfg.get("conditions", ["clean_300v", "noise_x9", "views_5", "si_x1p4"]))
    labels = cfg.get("condition_labels", {}) or {}
    curves = list(cfg.get("curves", ["ens", "err", "rand"]))
    plot = dict(cfg.get("plot", {}) or {})
    COL = {"ens": "#009E73", "err": "#E69F00", "rand": "#7f7f7f"}
    ncol = int(cfg.get("ncols", 2)); nrow = int(np.ceil(len(conds) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=tuple(plot.get("figsize", [9, 3.4 * nrow])), squeeze=False)
    xlims = cfg.get("xlims")                                  # optional: list of [lo,hi] per panel (log s-axis)
    ylims = cfg.get("ylims")
    dflt_xlim = plot.get("xlim", [1.0e-4, 1.0e4])
    dflt_ylim = plot.get("ylim", [0.0, 1.0])
    for i, cid in enumerate(conds):
        ax = axs[i // ncol][i % ncol]
        d = np.load(mdir / cid / "mode_projection.npz"); tau = float(d["tau"])
        ss, cum = _cum_curves(d, tau)                         # <-- structural-null-aware
        fn = {"ens": float(d["f_null_ens"]), "err": float(d["f_null_err"]), "rand": float(d["f_null_rand"])}
        for c in curves:
            if c not in cum: continue
            ax.semilogx(np.clip(ss, 1e-12, None), cum[c], color=COL.get(c, "C0"), lw=plot.get("lw", 1.6),
                        label=f"{c} ($f_{{\\rm null}}$={fn[c]:.2f})")
        ax.axvline(tau, color="r", ls="--", lw=1); ax.set_ylim(0, 1)
        ax.set_xlim(*(xlims[i] if xlims and i < len(xlims) else dflt_xlim))     # [1] per-panel or default
        ax.set_ylim(*(ylims[i] if ylims and i < len(ylims) else dflt_ylim))     # [1] per-panel or default
        ax.set_title(labels.get(cid, cid), fontsize=plot.get("title_fontsize", 10))
        if i // ncol == nrow - 1: ax.set_xlabel(r"singular value $s_\ell$")
        if i % ncol == 0: ax.set_ylabel("cumulative energy fraction")
        ax.legend(fontsize=10, loc="lower right")
    for j in range(len(conds), nrow * ncol): axs[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    return _savefig(fig, out / cfg.get("name", "fig_fnull_curves"),
                    cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))