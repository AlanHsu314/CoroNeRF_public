from __future__ import annotations
import copy, logging
from pathlib import Path
import numpy as np
from .operator import build_operator_from_cfg, load_operator
from .run import analyze_operator
from ..benchmark.config import load_yaml, save_json
from ..util.figure import _resolve_path, _savefig

logger = logging.getLogger(__name__)


def _spatial_rho(A, B):
    a = np.asarray(A, float).ravel(); b = np.asarray(B, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if a.size < 3: return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])

def _deployment_summary(recs, ref):
    """Collapse m̂ operating points into mean / cross-seed std; keep the m* reference. Grids are (Q,L,P,R)."""
    mhats = [r for r in recs if r is not ref]
    S = {"quantities": list(ref["quantities"]), "r_axis": np.asarray(ref["r_axis"])}
    for key in ("crb", "post", "pdom"):
        S[f"{key}_mstar"] = np.asarray(ref[key])
        stack = np.stack([np.asarray(r[key]) for r in mhats], axis=0)             # (n_mhat,Q,L,P,R)
        S[f"{key}_mhat_mean"] = stack.mean(0)
        S[f"{key}_mhat_std"]  = stack.std(0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(stack[0])
    return S, mhats

def _deployment_stats(S):
    """Agreement between deployment (mean-m̂) and oracle (m*) maps, per quantity."""
    stats = {}
    for key in ("crb", "post", "pdom"):
        A, B = np.asarray(S[f"{key}_mstar"]), np.asarray(S[f"{key}_mhat_mean"])
        a, b = A.ravel(), B.ravel(); ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        rel = np.abs(b - a) / (np.abs(a) + 1e-30)
        stats[key] = {"rho": float(_spatial_rho(B, A)),
                      "median_abs_dev": float(np.median(np.abs(b - a))) if a.size else float("nan"),
                      "median_rel_dev": float(np.median(rel)) if a.size else float("nan"),
                      "cross_seed_std_median": float(np.nanmedian(S[f"{key}_mhat_std"]))}
    return stats

def _fig_deployment(S, stats, out_base, cfg):
    """Diagnostic: 3 rows (crb/post/pdom) x [m*, mean-m̂, cross-m̂ std, scatter]. The main-text CLI figure
       reads the saved npz and shows only the scatter row."""
    import matplotlib.pyplot as plt
    qi = int(cfg.get("map_quantity_index", 0))                                    # 0 = ne
    ri = int(np.argmin(np.abs(S["r_axis"] - float(cfg.get("map_radius", 1.5)))))
    rows = [("crb", "CRB", "magma", True), ("post", "post", "inferno", True), ("pdom", "pdom", "viridis", False)]
    fig, axs = plt.subplots(3, 4, figsize=(13, 8.5), squeeze=False)
    for r, (key, lab, cm, is_log) in enumerate(rows):
        t = (lambda M: np.log10(np.clip(M, 1e-30, None))) if is_log else (lambda M: M)
        A = t(S[f"{key}_mstar"][qi, :, :, ri].T); B = t(S[f"{key}_mhat_mean"][qi, :, :, ri].T)
        D = t(S[f"{key}_mhat_std"][qi, :, :, ri].T)
        vmin, vmax = np.nanpercentile(A, 2), np.nanpercentile(A, 98)
        for c, (M, ttl, vl) in enumerate([(A, f"{lab} $m^*$", (vmin, vmax)),
                                          (B, f"{lab} $\\overline{{\\hat m}}$", (vmin, vmax)),
                                          (D, f"{lab} cross-$\\hat m$ std", (None, None))]):
            ax = axs[r][c]; im = ax.imshow(M, origin="lower", aspect="auto", cmap=cm, vmin=vl[0], vmax=vl[1])
            ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)
        ax = axs[r][3]
        a = S[f"{key}_mstar"][qi].ravel(); b = S[f"{key}_mhat_mean"][qi].ravel()
        ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        if is_log: a, b = np.log10(np.clip(a, 1e-30, None)), np.log10(np.clip(b, 1e-30, None))
        lim = [np.nanpercentile(a, 1), np.nanpercentile(a, 99)]
        ax.scatter(a, b, s=2, alpha=0.1); ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{lab}: $\\overline{{\\hat m}}$ vs $m^*$ ($\\rho$={stats[key]['rho']:.2f})", fontsize=9)
        ax.set_xlabel("$m^*$"); ax.set_ylabel(r"$\overline{\hat m}$")
    fig.suptitle(f"Deployment vs oracle identifiability maps — {S['quantities'][qi]}", fontsize=11)
    fig.tight_layout()
    return _savefig(fig, out_base, cfg.get("formats", ["png", "pdf"]), int(cfg.get("dpi", 200)))


def _fig_oppoints(recs, ref, map_radius, out_base, cfg):
    import matplotlib.pyplot as plt
    Q = ref["crb"].shape[0]; qs = ref["quantities"][:Q]
    ri = int(np.argmin(np.abs(ref["r_axis"] - float(map_radius))))
    ncol = len(recs) + 1
    fig, axes = plt.subplots(Q, ncol, figsize=(3.0 * ncol, 2.6 * Q), squeeze=False)
    for k, r in enumerate(recs):
        for qi in range(Q):
            ax = axes[qi][k]
            im = ax.imshow(r["crb"][qi, :, :, ri].T, origin="lower", aspect="auto", cmap="magma")
            ax.set_title(f"{r['id']} · {qs[qi]} CRB", fontsize=8); fig.colorbar(im, ax=ax, fraction=0.046)
    other = next((r for r in recs if r is not ref), ref)
    for qi in range(Q):
        ax = axes[qi][-1]
        D = np.abs(other["crb"][qi, :, :, ri] - ref["crb"][qi, :, :, ri]).T
        im = ax.imshow(D, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"|Δ CRB| {other['id']}", fontsize=8); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return _savefig(fig, out_base, cfg.get("formats", ["png", "pdf"]), int(cfg.get("dpi", 200)))


def compare_operating_points(spec_path, device_override=None, output_dir_override=None):
    """conditions × operating-points (m*/m̂) → per-condition eff_rank/CRB comparison + significance."""
    spec_path = Path(spec_path).resolve(); root = spec_path.parent
    cfg = load_yaml(spec_path).get("operating_points", {})
    out_root = _resolve_path(cfg.get("output_root", "../../runs_identifiability"), root)
    tau = float(cfg.get("tau", 1.0)); prior = dict(cfg.get("prior", {}) or {})
    map_radius = float(cfg.get("map_radius", 1.5))
    outdir = Path(output_dir_override) if output_dir_override else (out_root / "oppoints" / str(cfg.get("name", "oppoints")))
    outdir.mkdir(parents=True, exist_ok=True)
    base = cfg["base"]; man = {"name": cfg.get("name"), "conditions": []}

    for cond in cfg.get("conditions", [{"id": "base"}]):
        recs = []
        eta = float(cond.get("eta", 1.0))                                   # <-- add
        for op in cfg["operating_points"]:
            c = copy.deepcopy(base); c["base"] = dict(c.get("base", {}))
            for k in ("experiment", "where", "filters", "run_names", "min_seeds", "max_seeds"):
                if k in cond: c["base"][k] = cond[k]
            c["operating_point"] = dict(op)
            opdir = build_operator_from_cfg(c, root, device_override=device_override, output_root=out_root)
            O = load_operator(opdir)
            _bs = [float(s) * eta for s in O["_meta"]["base_shot_coeff"]]    # <-- apply η to whitening
            _bf = list(O["_meta"].get("base_sigma_floor", [0.0] * len(_bs)))
            res = analyze_operator(O, tau=tau, prior_cfg=prior, base_shot=_bs, base_floor=_bf)
            recs.append({"id": str(op["id"]), "kind": str(op.get("kind", "gt")),
                         "eff_rank": int(res["eff_rank_tau"]), "condition_tau": float(res["condition_tau"]),
                         "crb": np.asarray(res["crb_std_grid"]), "post": np.asarray(res["post_std_grid"]),
                         "pdom": np.asarray(res["prior_logratio_grid"]),
                         "r_axis": np.asarray(res["r_axis"]), "quantities": list(res["quantities"])})

        ref = next((r for r in recs if r["kind"] == "gt" or r["id"] == "mstar"), recs[0])
        mhats = [r for r in recs if r is not ref]
        rows = [{"id": r["id"], "kind": r["kind"], "eff_rank": r["eff_rank"], "condition_tau": r["condition_tau"],
                 "d_eff_rank": r["eff_rank"] - ref["eff_rank"],
                 "rho_crb_vs_ref": _spatial_rho(r["crb"], ref["crb"]),
                 "rho_post_vs_ref": _spatial_rho(r["post"], ref["post"])} for r in recs]
        sig = None
        if len(mhats) >= 2:
            er = np.array([r["eff_rank"] for r in mhats], float); sd = float(er.std(ddof=1))
            sig = {"seed_std_eff_rank": sd, "mean_mhat_eff_rank": float(er.mean()),
                   "delta_mstar_vs_meanmhat": float(ref["eff_rank"] - er.mean()),
                   "within_2sigma": bool(abs(ref["eff_rank"] - er.mean()) <= 2 * sd + 1e-9)}

        summary, _ = _deployment_summary(recs, ref)
        dstats = _deployment_stats(summary)
        cid = str(cond.get("id", "base"))
        np.savez_compressed(outdir / f"deployment_{cid}.npz",
            quantities=np.array(summary["quantities"]), r_axis=summary["r_axis"],
            **{k: summary[k] for k in summary if k.endswith(("_mstar", "_mhat_mean", "_mhat_std"))})
        man["conditions"].append({"id": cid, "rows": rows, "significance": sig, "deployment": dstats})
        _fig_deployment(summary, dstats, outdir / f"deployment_{cid}", cfg)

    save_json(man, outdir / "oppoints_manifest.json")
    logger.info("operating-point comparison -> %s", outdir)
    return man