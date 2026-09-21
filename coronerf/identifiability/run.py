##########################################################
### identifiability/run.py — diagnostics ON a saved operator (no Jacobian recompute)
##########################################################

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..benchmark.config import load_yaml, save_json
from ..util.figure import _resolve_path, _savefig
from .operator import load_operator
from ._core import (
    _quantities, _effective_ranks, posterior_diagnostics, _sigma_for, _eval_panel,
    _METRIC_LABEL, _LOGY_DEFAULT, _plot_spectrum, _plot_maps, _plot_posterior_maps,
    _interp_shell,
)

logger = logging.getLogger("coroNeRF.identifiability.run")


def analyze_operator(op, *, tau=1.0, prior_cfg=None, res_lambda=1e-3,
                     base_shot=None, base_floor=None, n_null_modes=4) -> dict:
    """SVD / Fisher / resolution / posterior from a LOADED operator (the cheap, re-runnable half)."""
    meta = op["_meta"]
    Q, L, P, R = (int(x) for x in meta["grid_shape"])
    target = meta["target"]; qlist = _quantities(target)
    base_shot = base_shot if base_shot is not None else meta["base_shot_coeff"]
    base_floor = base_floor if base_floor is not None else meta.get("base_sigma_floor", [0.0] * len(base_shot))

    sigma = _sigma_for(op, base_shot, base_floor, 1.0)               # (N,)
    Jw = op["J"].astype(np.float64) / sigma[:, None]
    U, s, Vt = np.linalg.svd(Jw, full_matrices=False)
    eff = _effective_ranks(s, tau); s_id = s[s > tau]
    cond_full = float(s[0] / max(s[-1], 1e-30))
    cond_tau = float(s[0] / max(s_id[-1], 1e-30)) if s_id.size else float("inf")

    H = Jw.T @ Jw
    info_diag = np.einsum("nm,nm->m", Jw, Jw)
    Rres = np.linalg.solve(H + float(res_lambda) * np.eye(H.shape[0]), H)
    res_diag = np.diag(Rres)

    def to_grid(v): return np.asarray(v, float).reshape(Q, L, P, R)
    info_grid, res_grid = to_grid(info_diag), to_grid(res_diag)
    null_modes = np.stack([to_grid(Vt[-(k + 1)]) for k in range(min(n_null_modes, Vt.shape[0]))], axis=0)
    post = posterior_diagnostics(H, s, Vt, (Q, L, P, R), prior_cfg or {})

    return {
        "target": target, "quantities": qlist, "N": int(Jw.shape[0]), "M": int(Jw.shape[1]),
        "singular_values": s, "tau": float(tau), **eff,
        "condition_full": cond_full, "condition_tau": cond_tau, "fd_check": meta.get("fd_check"),
        "lon_axis": np.asarray(op["lon_axis"]), "phi_axis": np.asarray(op["phi_axis"]),
        "r_axis": np.asarray(op["r_axis"]),
        "info_grid": info_grid, "res_grid": res_grid, "null_modes": null_modes, **post,
    }


def _plot_sweeps(results, panels, metrics, plot_cfg, out_base):
    formats = list(plot_cfg.get("formats", ["pdf", "png"])); dpi = int(plot_cfg.get("dpi", 300))
    logy = set(plot_cfg.get("logy_metrics", list(_LOGY_DEFAULT)))
    nrow, ncol = len(metrics), len(panels)
    fig, axs = plt.subplots(nrow, ncol, squeeze=False, figsize=plot_cfg.get("figsize", [4.6 * ncol, 3.6 * nrow]))
    for ci, panel in enumerate(panels):
        xs, rows = results[panel["name"]]; color = panel.get("color")
        for ri, met in enumerate(metrics):
            ax = axs[ri, ci]
            ax.plot(xs, rows[met], marker=panel.get("marker", "o"), color=color, lw=1.5)
            if ri == 0: ax.set_title(panel.get("title", panel["name"]))
            if ri == nrow - 1: ax.set_xlabel(panel.get("xlabel", panel["name"]))
            if ci == 0: ax.set_ylabel(_METRIC_LABEL.get(met, met))
            if panel.get("logx"): ax.set_xscale("log")
            if met in logy: ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    paths = _savefig(fig, out_base, formats, dpi); plt.close(fig)
    return paths


def _resolve_operator_dir(op_ref, runs_root: Path, spec_root: Path) -> Path:
    p = Path(op_ref)
    if p.is_absolute() and (p / "operator.npz").exists():
        return p
    if (spec_root / p / "operator.npz").exists():
        return (spec_root / p).resolve()
    for base in (runs_root / "operators", runs_root):          # operators/ cache first, then legacy flat
        if (base / op_ref / "operator.npz").exists():
            return base / op_ref
        matches = [m for m in sorted(base.glob(f"{op_ref}__*")) if (m / "operator.npz").exists()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning("multiple operators match '%s' under %s; using newest", op_ref, base)
            return max(matches, key=lambda m: (m / "operator.npz").stat().st_mtime)
    raise FileNotFoundError(f"no operator found for '{op_ref}' under {runs_root}[/operators]")

def _load_disagreement_shell(npz_path, q):
    from ..util.figure import _slug
    d = np.load(npz_path); sl = _slug(q)
    need = [f"{sl}_disagreement", f"{sl}_true_err", f"{sl}_lon_deg", f"{sl}_lat_deg"]
    if not all(k in d.files for k in need):
        return None
    return {"sigma": d[f"{sl}_disagreement"], "true_err": d[f"{sl}_true_err"],
            "lon": d[f"{sl}_lon_deg"], "lat": d[f"{sl}_lat_deg"]}

def _corr_ci(x, y, *, n_boot=1000, max_points=20000, seed=0):
    """Spearman rho(x,y) with a percentile bootstrap 95% CI over finite pairs."""
    from ..util.figure import _correlations
    x = np.asarray(x, float).reshape(-1); y = np.asarray(y, float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    base = _correlations(x, y)
    out = {"spearman": base["spearman"], "pearson": base["pearson"],
           "n": int(x.size), "ci_lo": float("nan"), "ci_hi": float("nan")}
    if x.size > 2 and int(n_boot) > 0:
        rng = np.random.default_rng(int(seed)); nb = min(int(max_points), x.size); b = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, x.size, nb)
            r = _correlations(x[idx], y[idx])["spearman"]
            if r is not None: b.append(r)
        if b:
            out["ci_lo"] = float(np.nanpercentile(b, 2.5)); out["ci_hi"] = float(np.nanpercentile(b, 97.5))
    return out


def _bridge_corr_ci(sigma, crb, post, *, n_boot=1000, max_points=20000, seed=0):
    """Paired bootstrap: rho(sigma,CRB), rho(sigma,post) and the GAP (post-crb) on shared resamples."""
    from ..util.figure import _correlations
    s = np.asarray(sigma, float).reshape(-1); c = np.asarray(crb, float).reshape(-1); p = np.asarray(post, float).reshape(-1)
    ok = np.isfinite(s) & np.isfinite(c) & np.isfinite(p); s, c, p = s[ok], c[ok], p[ok]
    bc, bp = _correlations(s, c), _correlations(s, p)
    out = {"sigma_vs_crb":  {"spearman": bc["spearman"], "pearson": bc["pearson"], "n": int(s.size),
                             "ci_lo": float("nan"), "ci_hi": float("nan")},
           "sigma_vs_post": {"spearman": bp["spearman"], "pearson": bp["pearson"], "n": int(s.size),
                             "ci_lo": float("nan"), "ci_hi": float("nan")},
           "gap_post_minus_crb": (None if bp["spearman"] is None or bc["spearman"] is None
                                  else float(bp["spearman"] - bc["spearman"])),
           "gap_ci_lo": float("nan"), "gap_ci_hi": float("nan")}
    if s.size > 2 and int(n_boot) > 0:
        rng = np.random.default_rng(int(seed)); nb = min(int(max_points), s.size)
        gc, gp, gg = [], [], []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, s.size, nb)
            rc = _correlations(s[idx], c[idx])["spearman"]; rp = _correlations(s[idx], p[idx])["spearman"]
            if rc is not None and rp is not None:
                gc.append(rc); gp.append(rp); gg.append(rp - rc)
        def _ci(a):
            a = np.asarray(a, float)
            return (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))) if a.size else (float("nan"), float("nan"))
        out["sigma_vs_crb"]["ci_lo"],  out["sigma_vs_crb"]["ci_hi"]  = _ci(gc)
        out["sigma_vs_post"]["ci_lo"], out["sigma_vs_post"]["ci_hi"] = _ci(gp)
        out["gap_ci_lo"], out["gap_ci_hi"] = _ci(gg)
    return out

def _bridge_correlations(res, disag_npz, *, map_radius=1.5, n_boot=1000, max_points=20000, seed=0):
    """Bridge correlations WITHOUT plotting: rho(sigma_ens, CRB) and rho(sigma_ens, post) per
    quantity on the shell nearest map_radius. Same numbers as _plot_bridge, no figure."""
    ri = int(np.argmin(np.abs(res["r_axis"] - float(map_radius))))
    corr = {}
    for qi, q in enumerate(res["quantities"]):
        dz = _load_disagreement_shell(disag_npz, q)
        if dz is None:
            continue
        crb  = _interp_shell(res["crb_std_grid"][qi, :, :, ri],  res["lon_axis"], res["phi_axis"], dz["lon"], dz["lat"])
        post = _interp_shell(res["post_std_grid"][qi, :, :, ri], res["lon_axis"], res["phi_axis"], dz["lon"], dz["lat"])
        corr[q] = _bridge_corr_ci(dz["sigma"], crb, post, n_boot=n_boot, max_points=max_points, seed=seed)
    return corr

def analyze_and_bridge(op, *, tau=1.0, prior_cfg=None, base_shot=None, base_floor=None,
                       disag_npz=None, bridge_cfg=None, res_lambda=1e-3, n_null_modes=4):
    """Analyze ONE operator (+ optional bridge correlations, no figure). Returns (res, row),
    where `row` is a flat dict of scalars for a sweep manifest."""
    res = analyze_operator(op, tau=tau, prior_cfg=dict(prior_cfg or {}),
                           base_shot=base_shot, base_floor=base_floor,
                           res_lambda=res_lambda, n_null_modes=n_null_modes)
    row = {k: res.get(k) for k in ("eff_rank_tau", "eff_rank_participation", "eff_rank_entropy",
                                   "condition_tau", "condition_full", "frac_prior_dominated",
                                   "crb_std_median", "post_std_median")}
    row["N"] = int(res["N"]); row["M"] = int(res["M"]); row["bridge"] = None
    if disag_npz is not None:
        bc = dict(bridge_cfg or {})
        row["bridge"] = _bridge_correlations(
            res, disag_npz,
            map_radius=float(bc.get("map_radius", 1.5)),
            n_boot=int(bc.get("n_boot", 1000)),
            max_points=int(bc.get("bootstrap_max_points", 20000)),
            seed=int(bc.get("bootstrap_seed", 0)))
    return res, row

def _plot_bridge(res, disag_npz, plot_cfg, out_base):
    from ..util.figure import _savefig, _cmap_with_bad, _correlations, _qty_tex
    formats = list(plot_cfg.get("formats", ["pdf", "png"])); dpi = int(plot_cfg.get("dpi", 300))
    ri = int(np.argmin(np.abs(res["r_axis"] - float(plot_cfg.get("map_radius", 1.5)))))
    qs = [q for q in res["quantities"]]
    corr = {}
    fig, axs = plt.subplots(len(qs), 4, figsize=plot_cfg.get("bridge_figsize", [15, 3.4*len(qs)]),
                            squeeze=False, layout="constrained")
    for qi, q in enumerate(qs):
        dz = _load_disagreement_shell(disag_npz, q)
        if dz is None:
            for c in range(4): axs[qi, c].axis("off")
            continue
        crb = _interp_shell(res["crb_std_grid"][qi, :, :, ri], res["lon_axis"], res["phi_axis"], dz["lon"], dz["lat"])
        post = _interp_shell(res["post_std_grid"][qi, :, :, ri], res["lon_axis"], res["phi_axis"], dz["lon"], dz["lat"])
        panels = [(r"True Error $\epsilon_{\mathrm{bias}}$", np.abs(dz["true_err"]), "magma", False),
                  (r"Ensemble Disagreement $\sigma_{\mathrm{ens}}$", dz["sigma"], "cividis", False),
                  ("CRB (data-only)", crb, "inferno", False),
                  ("posterior (data+prior)", post, "inferno", False)]
        for ci, (lab, img, cmap, _) in enumerate(panels):
            ax = axs[qi, ci]; v = img[np.isfinite(img)]
            hi = float(np.nanpercentile(v, 99)) if v.size else 1.0
            im = ax.imshow(img, origin="lower", cmap=_cmap_with_bad(cmap, "black"), vmin=0, vmax=max(hi, 1e-9), aspect="auto")
            if qi == 0: ax.set_title(lab)
            if ci == 0: ax.set_ylabel(_qty_tex(q, "field"))
            ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        corr[q] = _bridge_corr_ci(dz["sigma"], crb, post,
                                  n_boot=int(plot_cfg.get("n_boot", 1000)),
                                  max_points=int(plot_cfg.get("bootstrap_max_points", 20000)),
                                  seed=int(plot_cfg.get("bootstrap_seed", 0)))
        for ci_idx, key in ((2, "sigma_vs_crb"), (3, "sigma_vs_post")):
            cc = corr[q][key]
            txt = (r"$\rho$=%.3f" % cc["spearman"]) if cc["spearman"] is not None else r"$\rho$=n/a"
            if np.isfinite(cc.get("ci_lo", float("nan"))):
                txt += f"\n[{cc['ci_lo']:.3f}, {cc['ci_hi']:.3f}]"
            axs[qi, ci_idx].text(0.03, 0.97, txt, transform=axs[qi, ci_idx].transAxes,
                                 va="top", ha="left", fontsize=7, color="white",
                                 bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5))
    paths = _savefig(fig, out_base, formats, dpi); plt.close(fig)
    return {"paths": paths, "correlations": corr}

def _plot_bridge_sweep(op, disag_npz, tau, sweep_cfg, plot_cfg, out_base):
    """Sweep prior strength lambda (and optional anisotropy weight-sets); plot rho(sigma_ens, posterior)
    vs lambda, with rho(sigma_ens, CRB) as a lambda-independent dashed reference. One SVD, reused."""
    meta = op["_meta"]; Q, L, P, Rr = (int(x) for x in meta["grid_shape"])
    qlist = _quantities(meta["target"])
    base_shot  = sweep_cfg.get("base_shot_coeff", meta["base_shot_coeff"])
    base_floor = sweep_cfg.get("base_sigma_floor", meta.get("base_sigma_floor", [0.0] * len(base_shot)))

    sigma = _sigma_for(op, base_shot, base_floor, 1.0)
    Jw = op["J"].astype(np.float64) / sigma[:, None]
    _, s, Vt = np.linalg.svd(Jw, full_matrices=False)
    H = Jw.T @ Jw
    lon_axis = np.asarray(op["lon_axis"]); phi_axis = np.asarray(op["phi_axis"]); r_axis = np.asarray(op["r_axis"])
    ri = int(np.argmin(np.abs(r_axis - float(plot_cfg.get("map_radius", 1.5)))))

    lambdas = [float(x) for x in sweep_cfg.get("lambdas", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])]
    weight_sets = sweep_cfg.get("weight_sets") or [{"name": "isotropic", "w_lon": 1.0, "w_lat": 1.0, "w_r": 1.0}]
    ridge = float(sweep_cfg.get("ridge", 1e-3)); crb_floor = float(sweep_cfg.get("crb_floor", 1e-2))
    n_boot = int(sweep_cfg.get("n_boot", 0)); max_points = int(sweep_cfg.get("bootstrap_max_points", 20000))
    seed = int(sweep_cfg.get("seed", 0))

    shells = {q: _load_disagreement_shell(disag_npz, q) for q in qlist}
    table = {}
    fig, axs = plt.subplots(1, len(qlist), squeeze=False,
                            figsize=plot_cfg.get("sweep_figsize", [6.0 * len(qlist), 4.5]))
    axs = axs[0]
    for qi, q in enumerate(qlist):
        ax = axs[qi]; dz = shells.get(q)
        if dz is None:
            ax.axis("off"); continue
        sig = np.asarray(dz["sigma"], float).reshape(-1)
        table[q] = {}; crb_ref = None
        for ws in weight_sets:
            name = str(ws.get("name", f"w=({ws.get('w_lon', 1):g},{ws.get('w_lat', 1):g},{ws.get('w_r', 1):g})"))
            rho, lo, hi = [], [], []
            for lam in lambdas:
                pc = {"lambda": lam, "w_lon": float(ws.get("w_lon", 1.0)), "w_lat": float(ws.get("w_lat", 1.0)),
                      "w_r": float(ws.get("w_r", 1.0)), "ridge": ridge, "crb_floor": crb_floor}
                post = posterior_diagnostics(H, s, Vt, (Q, L, P, Rr), pc)
                ps = _interp_shell(post["post_std_grid"][qi, :, :, ri], lon_axis, phi_axis, dz["lon"], dz["lat"]).reshape(-1)
                cc = _corr_ci(sig, ps, n_boot=n_boot, max_points=max_points, seed=seed)
                rho.append(cc["spearman"]); lo.append(cc["ci_lo"]); hi.append(cc["ci_hi"])
                if crb_ref is None:
                    cs = _interp_shell(post["crb_std_grid"][qi, :, :, ri], lon_axis, phi_axis, dz["lon"], dz["lat"]).reshape(-1)
                    crb_ref = _corr_ci(sig, cs, n_boot=n_boot, max_points=max_points, seed=seed)
            rho = np.asarray([np.nan if r is None else r for r in rho], float)
            line, = ax.plot(lambdas, rho, marker="o", label=name)
            if n_boot > 0 and np.isfinite(np.asarray(lo, float)).any():
                ax.fill_between(lambdas, lo, hi, alpha=0.15, color=line.get_color())
            entry = {"lambdas": lambdas, "rho_post": rho.tolist(), "ci_lo": lo, "ci_hi": hi}
            if np.isfinite(rho).any():
                bi = int(np.nanargmax(rho))
                ax.scatter([lambdas[bi]], [rho[bi]], marker="*", s=140, zorder=5, color=line.get_color())
                entry["best_lambda"] = float(lambdas[bi]); entry["best_rho_post"] = float(rho[bi])
            table[q][name] = entry
        if crb_ref is not None and crb_ref["spearman"] is not None:
            ax.axhline(crb_ref["spearman"], ls="--", color="k", alpha=0.6,
                       label=rf"$\rho(\sigma,\mathrm{{CRB}})$={crb_ref['spearman']:.3f}")
            table[q]["crb_ref"] = crb_ref
        ax.set_xscale("log"); ax.set_xlabel(r"prior strength $\lambda$")
        ax.set_ylabel(r"Spearman $\rho(\sigma_{\mathrm{ens}},\,\cdot)$"); ax.set_title(q)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    paths = _savefig(fig, out_base, list(plot_cfg.get("formats", ["pdf", "png"])), int(plot_cfg.get("dpi", 300)))
    plt.close(fig)
    return {"paths": paths, "table": table}

def diagnose_from_spec(spec_path, device_override=None, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("diagnostics", {})
    runs_root = _resolve_path(cfg.get("runs_root", "../../runs_identifiability"), spec_root)
    op_dir = _resolve_operator_dir(cfg["operator"], runs_root, spec_root)
    op = load_operator(op_dir)
    fig_dir = op_dir / str(cfg.get("output", "figures")); fig_dir.mkdir(parents=True, exist_ok=True)
    tau = float(cfg.get("tau", 1.0)); plot_cfg = dict(cfg.get("plot", {}) or {})
    D = lambda k: dict(cfg.get(k, {}) or {})
    spec_d, maps_d, post_d, sw_d = D("spectrum"), D("maps"), D("posterior"), D("sweeps")
    logger.info("diagnose operator=%s", op_dir.name)
    manifest = {"operator_dir": str(op_dir), "tau": tau, "outputs": {}}

    if any(d.get("enabled", False) for d in (spec_d, maps_d, post_d)):
        res = analyze_operator(op, tau=tau, prior_cfg=dict(post_d.get("prior", {}) or {}),
                               res_lambda=float(maps_d.get("res_lambda", 1e-3)),
                               n_null_modes=int(maps_d.get("n_null_modes", 4)))
        np.savez_compressed(fig_dir / "analysis.npz",
                            singular_values=res["singular_values"], info_grid=res["info_grid"],
                            res_grid=res["res_grid"], null_modes=res["null_modes"],
                            crb_std_grid=res["crb_std_grid"], post_std_grid=res["post_std_grid"],
                            prior_logratio_grid=res["prior_logratio_grid"],
                            lon_axis=res["lon_axis"], phi_axis=res["phi_axis"], r_axis=res["r_axis"])
        if spec_d.get("enabled", True):
            manifest["outputs"]["spectrum"] = _plot_spectrum(res, {**plot_cfg, **spec_d}, fig_dir / "spectrum")[0]
        if maps_d.get("enabled", True):
            manifest["outputs"]["maps"] = _plot_maps(res, {**plot_cfg, **maps_d}, fig_dir / "maps")[0]
        if post_d.get("enabled", True):
            manifest["outputs"]["posterior"] = _plot_posterior_maps(res, {**plot_cfg, **post_d}, fig_dir / "posterior")[0]

        manifest.update({k: res[k] for k in ("eff_rank_tau", "eff_rank_participation", "condition_tau",
                                             "frac_prior_dominated")})
    
    # --- bridge + sweep: independent of spectrum/maps/posterior; uses its OWN prior ---
    bridge_d = D("bridge")
    if bridge_d.get("enabled", False):
        disag_npz = _resolve_path(bridge_d["disagreement_npz"], spec_root)
        res_b = analyze_operator(op, tau=tau, prior_cfg=dict(bridge_d.get("prior", {}) or {}))
        bout = _plot_bridge(res_b, disag_npz, {**plot_cfg, **bridge_d}, fig_dir / "bridge")
        manifest["outputs"]["bridge"] = bout["paths"]; manifest["bridge"] = bout["correlations"]
        manifest["bridge"]["disagreement_npz"] = str(disag_npz)
        sweep_d = dict(bridge_d.get("lambda_sweep", {}) or {})
        if sweep_d.get("enabled", False):
            sout = _plot_bridge_sweep(op, disag_npz, tau, sweep_d, {**plot_cfg, **bridge_d}, fig_dir / "bridge_sweep")
            manifest["outputs"]["bridge_sweep"] = sout["paths"]; manifest["bridge_sweep"] = sout["table"]

    if sw_d.get("enabled", False):
        base_shot = sw_d.get("base_shot_coeff", op["_meta"]["base_shot_coeff"])
        base_floor = sw_d.get("base_sigma_floor", op["_meta"].get("base_sigma_floor", [0.0] * len(base_shot)))
        panels = sw_d["panels"]; metrics = sw_d.get("metrics", ["eff_rank_tau", "eff_rank_participation"])
        results = {p["name"]: _eval_panel(op, p, base_shot, base_floor, tau) for p in panels}
        manifest["outputs"]["sweeps"] = _plot_sweeps(results, panels, metrics, {**plot_cfg, **sw_d}, fig_dir / "sweeps")
        manifest["sweeps"] = {p["name"]: {"x": results[p["name"]][0].tolist(),
                                          **{m: results[p["name"]][1][m] for m in metrics}} for p in panels}

    save_json(manifest, fig_dir / "diagnostics_manifest.json")
    logger.info("diagnostics written -> %s", fig_dir)
    return manifest