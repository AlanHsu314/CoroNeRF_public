from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import torch
from .operator import build_operator_from_cfg, load_operator, subset_operator, pick_views
from .mode_projection import _seed_control_fields                     # reuse the seed control-field loader
from ..benchmark.config import load_yaml, save_json
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.select import _collect_condition_seed_rows
from ..benchmark.condition_compare import _build_state_for_analysis, _resolve_run_dir
from ..util.figure import _resolve_path, _savefig

logger = logging.getLogger(__name__)

# ----------------------------- stats -----------------------------
def _rank(a):
    a = np.asarray(a, float).ravel(); r = np.empty(a.size); r[np.argsort(a)] = np.arange(a.size); return r

def _spearman(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return float("nan")
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])

def _partial(r_ue, r_up, r_pe):                                       # rho(u,e | p) closed form
    den = np.sqrt(max((1 - r_up ** 2) * (1 - r_pe ** 2), 1e-30))
    return float((r_ue - r_up * r_pe) / den)

# ------------------- risk-coverage / sparsification -------------------
def _risk_curve(e, unc, t_grid):
    """Mean error over the retained (least-uncertain) (1-t) fraction, for each abstention t."""
    e = np.asarray(e, float).ravel(); unc = np.asarray(unc, float).ravel()
    ok = np.isfinite(e) & np.isfinite(unc); e, unc = e[ok], unc[ok]; N = e.size
    order = np.argsort(unc)                                           # ascending unc -> most-trusted first
    ecum = np.cumsum(e[order]); ks = np.arange(1, N + 1)
    means = ecum / ks                                                 # retained-MAE as a function of k kept
    idx = np.clip(np.round((1.0 - t_grid) * N).astype(int) - 1, 0, N - 1)
    return means[idx]

def _risk_random(e, t_grid, seed=0, n_rand=None):
    """Exact expected retained-MAE under uniformly random retention.
    For a uniformly random kept subset of ANY size, E[mean error] = the global mean μ at every
    abstention level, so the expected sparsification curve is the flat line μ — zero variance,
    no terminal spike. This is the exact n_rand -> inf limit of the old permutation average.
    `seed`/`n_rand` are kept only for call-signature compatibility and are ignored."""
    e = np.asarray(e, float).ravel(); e = e[np.isfinite(e)]
    if e.size == 0:
        return np.full(t_grid.size, np.nan)
    return np.full(t_grid.size, float(e.mean()))

def _ause(risk_u, risk_oracle, t_grid):
    se = np.maximum(np.asarray(risk_u) - np.asarray(risk_oracle), 0.0)
    return float(np.trapz(se, t_grid))

def _headline(u, e, proxies, t_grid):
    """The five scalars the paper reports, recomputed cheaply (used by the bootstrap)."""
    r_ue = _spearman(u, e); p = proxies["emis"]; r_pe = _spearman(p, e); r_up = _spearman(u, p)
    ro = _risk_curve(e, e, t_grid); rr = _risk_random(e, t_grid, n_rand=5); ar = _ause(rr, ro, t_grid) or np.nan
    return {"sigma_vs_err": r_ue, "emis_vs_err": r_pe,
            "sigma_vs_err_given_emis": _partial(r_ue, r_up, r_pe),
            "nause_sigma": _ause(_risk_curve(e, u, t_grid), ro, t_grid) / ar,
            "nause_emis":  _ause(_risk_curve(e, p, t_grid), ro, t_grid) / ar}

def _boot_ci(u3, e3, prox3, t_grid, n_boot, seed=0):
    """95% CIs by resampling whole longitude columns (respects spatial autocorrelation)."""
    L = u3.shape[0]; rng = np.random.default_rng(seed); acc = {}
    for _ in range(int(n_boot)):
        li = rng.choice(L, L, replace=True)
        h = _headline(u3[li].ravel(), e3[li].ravel(), {k: v[li].ravel() for k, v in prox3.items()}, t_grid)
        for k, val in h.items(): acc.setdefault(k, []).append(val)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in acc.items()}

def _analyze(u, e, proxies, t_grid, n_rand=20):
    """u,e: (Nvox,); proxies: name -> (Nvox,) uncertainty-oriented (high = expect high error)."""
    cor, ause, risk = {}, {}, {}
    r_ue = _spearman(u, e); cor["sigma_vs_err"] = r_ue
    for nm, p in proxies.items():
        r_pe = _spearman(p, e); r_up = _spearman(u, p)
        cor[f"{nm}_vs_err"] = r_pe; cor[f"sigma_vs_{nm}"] = r_up
        cor[f"sigma_vs_err_given_{nm}"] = _partial(r_ue, r_up, r_pe)
        cor[f"{nm}_vs_err_given_sigma"] = _partial(r_pe, r_up, r_ue)
    risk["oracle"] = _risk_curve(e, e, t_grid)                        # rank by true error
    risk["random"] = _risk_random(e, t_grid, n_rand=n_rand)
    risk["sigma"] = _risk_curve(e, u, t_grid)
    ause["oracle"] = 0.0; ause["random"] = _ause(risk["random"], risk["oracle"], t_grid)
    ause["sigma"] = _ause(risk["sigma"], risk["oracle"], t_grid)
    for nm, p in proxies.items():
        risk[nm] = _risk_curve(e, p, t_grid); ause[nm] = _ause(risk[nm], risk["oracle"], t_grid)
    ar = ause.get("random", 0.0) or 0.0
    nause = {m: (ause[m] / ar if ar > 0 else float("nan")) for m in ause}   # 0=oracle, 1=random, >1 worse
    return {"correlations": cor, "ause": ause, "nause": nause, "risk": risk}

# ----------------------- operator-side quantities -----------------------
def _diag_fisher_and_mstar(op, shot_coeff, floor):
    """diag(H) = diag(J^T Sigma^-1 J) and the GT control field, both length-M. No SVD."""
    meta = op["_meta"]; J = np.asarray(op["J"]); I0 = np.asarray(op["I0"]).reshape(-1)
    sel = [int(x) for x in meta["selected_channel_indices"]]
    g2l = {g: i for i, g in enumerate(sel)}
    local = np.array([g2l[int(g)] for g in np.asarray(op["obs_chan_global"])], dtype=int)
    a = np.asarray(shot_coeff, float); b = np.asarray(floor, float)
    var = a[local] * np.clip(I0, 0.0, None) + b[local] ** 2
    w = 1.0 / np.maximum(var, 1e-30)                                  # (N,)
    diagH = np.zeros(J.shape[1], float)
    for i0 in range(0, J.shape[0], 8192):                             # chunk to bound memory
        Jc = J[i0:i0 + 8192]; diagH += (Jc * Jc * w[i0:i0 + 8192, None]).sum(0)
    return diagH, np.asarray(op["control_values"]).reshape(-1), sel

def _gt_emissivity(state, mstar_g, r_axis, channels):
    """mstar_g: (Q,L,P,R) log10 GT fields -> total emissivity (L,P,R) over `channels`.
    RegularTrilinearField.forward returns (C, N) (channels-first); select channel rows and sum axis 0."""
    if mstar_g.shape[0] < 2:
        raise ValueError("emissivity needs both nₑ and T (target must be 'ne_t')")
    _, L, P, R = mstar_g.shape
    ne = np.ascontiguousarray(mstar_g[0]).reshape(-1)          # log10 nₑ  (Nvox,)
    te = np.ascontiguousarray(mstar_g[1]).reshape(-1)          # log10 T
    logr = (np.log10(r_axis)[None, None, :] * np.ones((L, P, 1))).reshape(-1)
    dev = state.renderer.device; dt = state.renderer.dtype
    ntr = torch.stack([torch.as_tensor(ne), torch.as_tensor(te), torch.as_tensor(logr)], dim=1).to(device=dev, dtype=dt)
    with torch.no_grad():
        ccoef = np.asarray(state.renderer.ccoef_field(ntr).detach().cpu())    # (C, Nvox)  <-- channels-first
    Nvox = L * P * R; ch = [int(c) for c in channels]
    if ccoef.shape[-1] == Nvox and ccoef.shape[0] != Nvox:                    # (C, Nvox): channels = rows
        emis = ccoef[ch, :].sum(0)
    else:                                                                     # (Nvox, C) fallback, just in case
        emis = ccoef[:, ch].sum(1)
    return emis.reshape(L, P, R)
# ------------------------------- figure -------------------------------
def _fig(save, jrec, quantities, out_base, cfg):
    import matplotlib.pyplot as plt
    t = save["t_grid"]; nq = len(quantities)
    fig, axs = plt.subplots(nq, 2, figsize=(11, 3.6 * nq), squeeze=False)
    bars = ["sigma_vs_err", "emis_vs_err", "dens_vs_err", "fisher_vs_err",
            "sigma_vs_err_given_emis", "emis_vs_err_given_sigma"]
    blab = ["σ·e", "emis·e", "dens·e", "fish·e", "σ·e|emis", "emis·e|σ"]
    curves = [("oracle", "k"), ("random", "0.6"), ("sigma", "#1f77b4"),
              ("emis", "#ff7f0e"), ("dens", "#2ca02c"), ("fisher", "#9467bd")]
    for qi, q in enumerate(quantities):
        cor = jrec["quantities"][q]["correlations"]; au = jrec["quantities"][q]["ause"]
        axs[qi][0].bar(range(len(bars)), [cor.get(b, np.nan) for b in bars],
                       color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4", "#ff7f0e"])
        axs[qi][0].set_xticks(range(len(bars))); axs[qi][0].set_xticklabels(blab, rotation=30, ha="right", fontsize=8)
        axs[qi][0].axhline(0, color="k", lw=0.6); axs[qi][0].set_ylabel(f"{q}  Spearman ρ")
        for nm, c in curves:
            axs[qi][1].plot(t, save[f"{q}_risk_{nm}"], color=c, lw=1.6, label=f"{nm} (AUSE={au.get(nm, 0):.3f})")
        axs[qi][1].set_xlabel("fraction removed  t"); axs[qi][1].set_ylabel(f"{q} retained MAE")
        axs[qi][1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _savefig(fig, out_base, cfg.get("formats", ["png", "pdf"]), int(cfg.get("dpi", 200)))

def _ortho_span(Delta, tol_rel=1e-8):
    """Orthonormal basis Q (M,r) of the column span of Delta (M,K); r = numerical rank (<= K-1)."""
    Ud, sd, _ = np.linalg.svd(Delta, full_matrices=False)
    tol = float(sd.max()) * tol_rel if sd.size else 0.0
    r = int((sd > tol).sum())
    return (Ud[:, :r] if r > 0 else np.zeros((Delta.shape[0], 0), float)), r


def _cap(Q, v, vt=None):
    """Fraction of v inside span(Q): ||Q^T v||^2 / ||v||^2."""
    vt = float((v ** 2).sum()) if vt is None else float(vt)
    if vt <= 0.0 or Q.shape[1] == 0:
        return 0.0
    return float(((Q.T @ v) ** 2).sum() / vt)

def _balanced_splits(K, max_splits=2000, seed=0, enum_cap=500000):
    """Balanced (span = K//2 seeds) splits for the E_cap generalization test.
    If C(K, K//2) <= max_splits, ENUMERATE every balanced split so the reported spread is the exact,
    finite split-sensitivity distribution (252 for K=10) rather than a Monte-Carlo estimate. Otherwise
    draw `max_splits` DISTINCT splits without replacement (enumerate-then-subsample while the space is
    small enough to hold, else rejection-sample). Splits are keyed by the span index-set, matching the
    exhaustive semantics where a set and its complement are two distinct generalization directions.
    Returns (list of span-index tuples, exhaustive_bool, n_possible)."""
    from math import comb
    from itertools import combinations
    k = K // 2
    n_all = int(comb(K, k))
    if n_all <= int(max_splits):                              # exact finite distribution
        return list(combinations(range(K), k)), True, n_all
    rng = np.random.default_rng(seed)
    if n_all <= int(enum_cap):                                # small enough to list, then subsample distinct
        allc = list(combinations(range(K), k))
        pick = rng.choice(len(allc), int(max_splits), replace=False)
        return [allc[int(i)] for i in pick], False, n_all
    seen = set()                                              # huge space: rejection-sample distinct splits
    while len(seen) < int(max_splits):
        seen.add(tuple(sorted(int(x) for x in rng.choice(K, k, replace=False))))
    return sorted(seen), False, n_all

def _ecap_field(mh_q, mstar_q, lpr, max_splits=2000, seed=0):
    """Ensemble-geometry E_cap for ONE field block (no Jacobian).
    mh_q:(K,Mq) seed fields, mstar_q:(Mq,) truth, lpr=(L,P,R) with L=longitude.
    Returns scalars + '_split_samples'/'_rot_samples' arrays (popped into the npz by the caller)."""
    #from itertools import combinations
    K, Mq = mh_q.shape; L, P, R = lpr
    mbar = mh_q.mean(0); eps = mbar - mstar_q; et = float((eps ** 2).sum())
    Delta = (mh_q - mbar).T                                              # (Mq, K)
    Q, r = _ortho_span(Delta)
    e_cap = _cap(Q, eps, et); e_rand = r / max(Mq, 1)

    # --- split-seed: span from half the seeds, capture the OTHER half's mean error ---
    # K=10 -> all C(10,5)=252 balanced splits enumerated (exact finite spread); larger K subsamples distinct.
    combos, split_exhaustive, n_split_all = _balanced_splits(K, max_splits=max_splits, seed=seed)
    allset = set(range(K)); sp, sp_rand = [], []
    for A in combos:
        A = list(A); B = sorted(allset - set(A))
        mbA = mh_q[A].mean(0); QA, rA = _ortho_span((mh_q[A] - mbA).T)
        epsB = mh_q[B].mean(0) - mstar_q
        sp.append(_cap(QA, epsB)); sp_rand.append(rA / max(Mq, 1))
    sp = np.asarray(sp, float); sp_rand = np.asarray(sp_rand, float)

    # --- longitude-rotation null: roll eps in longitude vs the fixed span (== rigidly rotating the span) ---
    eps_g = eps.reshape(L, P, R); rot = [_cap(Q, np.roll(eps_g, s, axis=0).reshape(-1), et) for s in range(1, L)]
    rot = np.asarray(rot, float)
    rot_pct = float((rot < e_cap).mean() * 100.0)                       # 100 => observed beats every rotation

    return {"e_cap": e_cap, "e_cap_rand": e_rand, "enrich": e_cap / (e_rand + 1e-30), "r_ens": int(r), "M_field": int(Mq),
            "split_mean": float(sp.mean()), "split_ci": [float(np.percentile(sp, 2.5)), float(np.percentile(sp, 97.5))],
            "split_rand_mean": float(sp_rand.mean()), "n_split": int(sp.size),
            "split_exhaustive": bool(split_exhaustive), "n_split_possible": int(n_split_all),
            "rot_observed": e_cap, "rot_null_mean": float(rot.mean()), "rot_null_max": float(rot.max()),
            "rot_percentile": rot_pct, "n_rot": int(rot.size),
            "_split_samples": sp, "_rot_samples": rot}

def _loo_stability(mhats, mstar, shape, qi, r_mask, p_emis, t_grid):
    """Leave-one-seed-out stability of the localization signal, for ONE field block.
    For each seed j: drop it, recompute sigma_ens^(-j) and eps^(-j)=|mbar^(-j)-m*| from the K-1
    remaining seeds (both move; the emissivity proxy p_emis is GT-based and stays fixed), then
    recompute the headline metrics on the SAME in-band voxels. Returns per-seed arrays (length K)."""
    K = mhats.shape[0]
    Mq = int(np.prod(shape[1:]))                      # voxels per field  (n_lon*n_lat*n_r)
    sl = slice(qi * Mq, (qi + 1) * Mq)
    keys = ["sigma_vs_err", "sigma_vs_err_given_emis", "nause_sigma"]
    out = {k: [] for k in keys}
    for j in range(K):
        mh = np.delete(mhats, j, axis=0)              # (K-1, M)
        sig_j = mh.std(0, ddof=1)[sl].reshape(shape[1:])[:, :, r_mask].ravel()
        err_j = np.abs(mh.mean(0) - mstar)[sl].reshape(shape[1:])[:, :, r_mask].ravel()
        h = _headline(sig_j, err_j, {"emis": p_emis}, t_grid)   # same 5-scalar recipe as the bootstrap
        for k in keys:
            out[k].append(float(h[k]))
    return {k: np.asarray(v, float) for k, v in out.items()}


def _loo_summary(arr):
    """min / median / max / range / IQR over the K leave-one-out ensembles (the appendix-table row)."""
    v = np.asarray(arr, float); v = v[np.isfinite(v)]
    if not v.size:
        return None
    q25, q75 = np.percentile(v, [25, 75])
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "range": float(v.max() - v.min()), "iqr": float(q75 - q25), "n": int(v.size)}

def _fig_shells(jrec, quantities, out_base, cfg):
    """Within-shell rho(sigma,e) and partial rho(sigma,e|b_E) vs radius; dotted line = pooled value."""
    import matplotlib.pyplot as plt
    nq = len(quantities)
    fig, axs = plt.subplots(1, nq, figsize=(5.2 * nq, 3.6), squeeze=False); axs = axs[0]
    for qi, q in enumerate(quantities):
        ps = jrec["quantities"][q].get("per_shell", [])
        if not ps:
            axs[qi].axis("off"); continue
        r = [s["r_center"] for s in ps]
        axs[qi].plot(r, [s["correlations"]["sigma_vs_err"] for s in ps], "o-", label=r"$\rho(\sigma,\epsilon)$")
        axs[qi].plot(r, [s["correlations"]["sigma_vs_err_given_emis"] for s in ps], "s--",
                     label=r"$\rho(\sigma,\epsilon\,|\,b_E)$")
        pooled = jrec["quantities"][q]["correlations"]["sigma_vs_err"]
        axs[qi].axhline(pooled, color="0.5", ls=":", lw=1.2, label=f"pooled ρ={pooled:.2f}")
        axs[qi].axhline(0, color="k", lw=0.6)
        axs[qi].set_xlabel(r"radius ($R_\odot$)"); axs[qi].set_ylabel(f"{q}  Spearman ρ")
        axs[qi].set_title(f"{q}: within-shell localization"); axs[qi].legend(fontsize=7)
    fig.tight_layout()
    return _savefig(fig, out_base, cfg.get("formats", ["png", "pdf"]), int(cfg.get("dpi", 200)))

def _shell_masks(r_axis, band_mask, shells_cfg):
    """Radial shells over the in-band radii. Default: one shell per native in-band radius.
    shells_cfg={'edges':[e0,e1,...]} -> custom bins [e_i, e_{i+1}). Returns [(r_center, r_lo, r_hi, mask_over_R), ...]."""
    r = np.asarray(r_axis, float); band = np.asarray(band_mask, bool); out = []
    edges = (shells_cfg or {}).get("edges")
    if edges:
        edges = [float(x) for x in edges]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = band & (r >= lo) & (r < hi)
            if m.any():
                out.append((float(r[m].mean()), lo, hi, m))
    else:                                                            # one shell per native in-band radius
        for i in np.where(band)[0]:
            m = np.zeros_like(band); m[i] = True
            out.append((float(r[i]), float(r[i]), float(r[i]), m))
    return out


def _shell_summary(shell_recs):
    """Median / min / max / IQR of the key within-shell stats across shells (the radial-confound answer)."""
    def agg(getter):
        v = np.array([getter(s) for s in shell_recs], float); v = v[np.isfinite(v)]
        if not v.size:
            return None
        q25, q75 = np.percentile(v, [25, 75])
        return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "iqr": float(q75 - q25)}
    return {"n_shells": len(shell_recs),
            "sigma_vs_err":            agg(lambda s: s["correlations"]["sigma_vs_err"]),
            "sigma_vs_err_given_emis": agg(lambda s: s["correlations"]["sigma_vs_err_given_emis"]),
            "nause_sigma":             agg(lambda s: s["nause"].get("sigma", float("nan")))}

def _fig_ecap(save, jrec, quantities, out_base, cfg):
    """Per-field E_cap: observed vs split-seed (mean+CI) vs longitude-rotation null (scatter); isotropic dashed line."""
    import matplotlib.pyplot as plt
    nq = len(quantities); fig, axs = plt.subplots(1, nq, figsize=(5.0 * nq, 3.7), squeeze=False); axs = axs[0]
    for qi, q in enumerate(quantities):
        eg = jrec["quantities"][q].get("ecap")
        if not eg:
            axs[qi].axis("off"); continue
        rot = save.get(f"{q}_ecap_rot")
        lo, hi = eg["split_ci"]
        axs[qi].bar([0, 1], [eg["e_cap"], eg["split_mean"]],
                    yerr=[0, (hi - lo) / 2], color=["#1f77b4", "#2ca02c"], width=0.6)
        if rot is not None and rot.size:
            axs[qi].plot(np.full(rot.size, 2.0), rot, ".", color="k", ms=4, alpha=0.5)
            axs[qi].plot([2], [eg["rot_null_mean"]], "_", color="r", ms=18)
        axs[qi].axhline(eg["e_cap_rand"], color="r", ls="--", lw=1, label=f"isotropic {eg['e_cap_rand']:.4f}")
        axs[qi].set_xticks([0, 1, 2]); axs[qi].set_xticklabels(["observed", "split-seed", "rot-null"])
        axs[qi].set_ylabel(f"{q}  E_cap")
        _sx = "all" if eg.get("split_exhaustive") else "sub"
        axs[qi].set_title(f"{q}: {100*eg['e_cap']:.1f}% ({eg['enrich']:.0f}×)  "
                          f"rot pct={eg['rot_percentile']:.0f}  splits={eg.get('n_split', 0)}/"
                          f"{eg.get('n_split_possible', 0)} ({_sx})")
        axs[qi].legend(fontsize=7)
    fig.tight_layout()
    return _savefig(fig, out_base, cfg.get("formats", ["png", "pdf"]), int(cfg.get("dpi", 200)))

# ------------------------------- driver -------------------------------
def run_baselines(spec_path, device_override=None, output_dir_override=None):
    spec_path = Path(spec_path).resolve(); root = spec_path.parent
    cfg = load_yaml(spec_path).get("baselines", {})
    bdir = _resolve_path(cfg["benchmark_dir"], root)
    device = device_override if device_override is not None else cfg.get("device")
    path_remap = cfg.get("path_remap"); target = cfg.get("target", "ne_t")
    grid = dict(cfg["base"].get("grid", {}))
    n_lon = int(grid.get("n_lon", 30)); n_lat = int(grid.get("n_lat", 18)); n_r = int(grid.get("n_r", 8))
    r_axis = np.linspace(float(grid.get("r_min", 1.05)), float(grid.get("r_max", 3.0)), n_r)
    quantities = ["ne", "temp"] if target == "ne_t" else [str(target)]
    shape = (len(quantities), n_lon, n_lat, n_r)

    band = dict(cfg.get("band", {}) or {}); rlist = band.get("radii", None)
    if rlist:                                                         # explicit radii, e.g. [1.5]
        idx = sorted({int(np.argmin(np.abs(r_axis - float(rr)))) for rr in rlist})
        r_mask = np.isin(np.arange(n_r), idx)
    else:                                                            # pooled band [r_min, r_max]
        r_mask = (r_axis >= float(band.get("r_min", 1.1))) & (r_axis <= float(band.get("r_max", 2.0)))
    tg = dict(cfg.get("t_grid", {}) or {}); t_grid = np.linspace(0.0, float(tg.get("t_max", 0.95)), int(tg.get("n", 100)))
    n_rand = int(cfg.get("n_random", 20))
    do_loo = bool(cfg.get("leave_one_out", False)) # leave-one-seed-out stats

    shell_specs = _shell_masks(r_axis, r_mask, dict(cfg.get("shells", {}) or {}))
    logger.info("baselines: %d radial shell(s) in band at r=%s R_sun",
                len(shell_specs), [round(s[0], 3) for s in shell_specs])

    #E_cap, capture fraction config
    ecap_cfg = dict(cfg.get("ecap", {}) or {})

    op_out = _resolve_path(cfg.get("output_root", "../../runs_identifiability"), root)
    out_root = Path(output_dir_override) if output_dir_override else (op_out / "baselines" / str(cfg.get("name", "baselines")))
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = dict(cfg["base"]); base_cfg.setdefault("path_remap", path_remap)     # thread top-level -> operator
    base_op = load_operator(build_operator_from_cfg(base_cfg, root, device_override=device, output_root=op_out))
    base_shot = list(base_op["_meta"]["base_shot_coeff"]); base_floor = list(base_op["_meta"].get("base_sigma_floor", [0.0] * len(base_shot)))
    summaries = collect_run_summaries(bdir)
    man = {"name": cfg.get("name", "baselines"),
            "band": {"radii": rlist, "r_mask": r_axis[r_mask].tolist(),
                    "shells": [round(s[0], 3) for s in shell_specs]}, "conditions": []}

    for cond in cfg["conditions"]:
        cid = str(cond["id"]); eta = float(cond.get("eta", 1.0)); mode = str(cond.get("mode", "derive"))
        channels = cond.get("channels", None); views = int(cond.get("views", int(base_op["vidx"].size)))
        if mode == "build":
            op_cfg = dict(cond["operator"]); op_cfg.setdefault("path_remap", path_remap)
            op = load_operator(build_operator_from_cfg(op_cfg, root, device_override=device, output_root=op_out))
            shot_c = list(op["_meta"]["base_shot_coeff"]); floor = list(op["_meta"].get("base_sigma_floor", [0.0] * len(shot_c)))
        else:
            vs = pick_views(base_op["vidx"], views) if views < int(base_op["vidx"].size) else None
            op = subset_operator(base_op, view_subset=vs, channel_subset=channels)
            shot_c = [s * eta for s in base_shot]; floor = base_floor
        diagH, mstar, sel = _diag_fisher_and_mstar(op, shot_c, floor)

        sel_keys = {k: cond[k] for k in ("experiment", "where", "run_names", "min_seeds", "max_seeds") if k in cond}
        srows = _collect_condition_seed_rows(summaries, sel_keys)
        if not srows:
            raise ValueError(f"baselines[{cid}]: no seed runs matched {sel_keys}")
        mhats = _seed_control_fields(srows, bdir, grid, target, device, path_remap)          # (K, M)
        K = mhats.shape[0]
        mbar = mhats.mean(0); sig = mhats.std(0, ddof=1 if K > 1 else 0); err = np.abs(mbar - mstar)          # (M,)
        mstar_g = mstar.reshape(shape); sig_g = sig.reshape(shape); err_g = err.reshape(shape); fish_g = diagH.reshape(shape)

        st = _build_state_for_analysis(_resolve_run_dir(srows[0], bdir), device_override=device, path_remap=path_remap)
        eps = _gt_emissivity(st, mstar_g, r_axis, sel); del st                                # (L,P,R)
        dens = mstar_g[0]                                                                     # log10 nₑ (L,P,R)

        cdir = out_root / cid; cdir.mkdir(parents=True, exist_ok=True)
        save = {"t_grid": t_grid, "r_axis": r_axis, "r_mask": r_mask}
        jrec = {"id": cid, "eta": eta, "n_seeds": int(mhats.shape[0]), "quantities": {}}
        n_boot = int(cfg.get("n_boot", 0))
        for qi, q in enumerate(quantities):
            u3 = sig_g[qi][:, :, r_mask]; e3 = err_g[qi][:, :, r_mask]          # (L,P,Rin) kept for the block bootstrap
            prox3 = {"emis": -eps[:, :, r_mask], "dens": -dens[:, :, r_mask], "fisher": -fish_g[qi][:, :, r_mask]}
            u = u3.ravel(); e = e3.ravel(); proxies = {k: v.ravel() for k, v in prox3.items()}
            res = _analyze(u, e, proxies, t_grid, n_rand=n_rand)
            save[f"{q}_u"] = u; save[f"{q}_e"] = e
            for nm, p in proxies.items(): save[f"{q}_p_{nm}"] = p
            for nm, rc in res["risk"].items(): save[f"{q}_risk_{nm}"] = rc
            qrec = {"correlations": res["correlations"], "ause": res["ause"],
                    "nause": res["nause"], "n_voxels": int(e.size)}
            if n_boot > 0:
                qrec["ci95"] = _boot_ci(u3, e3, prox3, t_grid, n_boot)          # longitude-block 95% CIs

            # --- radius-controlled: is rho(sigma, e) present WITHIN each shell, not just a global radial trend? ---
            shell_recs = []
            for (rc, rlo, rhi, smask) in shell_specs:
                us = sig_g[qi][:, :, smask].ravel(); es = err_g[qi][:, :, smask].ravel()
                ps = {"emis":   (-eps[:, :, smask]).ravel(),
                      "dens":   (-dens[:, :, smask]).ravel(),
                      "fisher": (-fish_g[qi][:, :, smask]).ravel()}
                rs = _analyze(us, es, ps, t_grid, n_rand=n_rand)               # same metrics, restricted to the shell
                shell_recs.append({"r_center": rc, "r_lo": rlo, "r_hi": rhi, "n_voxels": int(es.size),
                                   "correlations": rs["correlations"], "nause": rs["nause"]})
            qrec["per_shell"] = shell_recs
            qrec["shell_summary"] = _shell_summary(shell_recs)

            # --- ensemble geometry E_cap (per field; no Jacobian): absolute + split-seed + rotation null ---
            Mq = n_lon * n_lat * n_r; sl = slice(qi * Mq, (qi + 1) * Mq)
            eg = _ecap_field(mhats[:, sl], mstar[sl], (n_lon, n_lat, n_r),
                             max_splits=int(ecap_cfg.get("max_splits", 2000)), seed=int(ecap_cfg.get("seed", 0)))
            save[f"{q}_ecap_split"] = eg.pop("_split_samples")
            save[f"{q}_ecap_rot"]   = eg.pop("_rot_samples")
            qrec["ecap"] = eg

            # --- leave-one-seed-out: does dropping any single run change the localization signal? ---
            if do_loo and K > 2:
                loo = _loo_stability(mhats, mstar, shape, qi, r_mask, proxies["emis"], t_grid)
                save[f"{q}_loo_rho"]     = loo["sigma_vs_err"]
                save[f"{q}_loo_partial"] = loo["sigma_vs_err_given_emis"]
                save[f"{q}_loo_nause"]   = loo["nause_sigma"]
                qrec["loo"] = {"n_ensembles": int(loo["sigma_vs_err"].size),
                               "rho":     _loo_summary(loo["sigma_vs_err"]),
                               "partial": _loo_summary(loo["sigma_vs_err_given_emis"]),
                               "nause":   _loo_summary(loo["nause_sigma"]),
                               "per_seed": {"rho":     loo["sigma_vs_err"].tolist(),
                                            "partial": loo["sigma_vs_err_given_emis"].tolist(),
                                            "nause":   loo["nause_sigma"].tolist()}}

            # save it
            jrec["quantities"][q] = qrec
        
        np.savez_compressed(cdir / "baselines.npz", **save)
        save_json(jrec, cdir / "baselines.json")
        _fig(save, jrec, quantities, cdir / "baselines", cfg)
        _fig_shells(jrec, quantities, cdir / "baselines_shells", cfg)
        _fig_ecap(save, jrec, quantities, cdir / "baselines_ecap", cfg)
        man["conditions"].append(jrec)
        c = jrec["quantities"]["ne"]
        logger.info("baselines[%s]: ne  rho(sig,e)=%.3f rho(emis,e)=%.3f  partial(sig|emis)=%.3f  AUSE sig/emis=%.4f/%.4f",
                    cid, c["correlations"]["sigma_vs_err"], c["correlations"]["emis_vs_err"],
                    c["correlations"]["sigma_vs_err_given_emis"], c["ause"]["sigma"], c["ause"]["emis"])
    save_json(man, out_root / "baselines_manifest.json")
    return man