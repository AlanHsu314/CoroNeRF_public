from __future__ import annotations
import copy, logging
from pathlib import Path
import numpy as np

from .operator import build_operator_from_cfg, load_operator
from .mode_projection import _project, _seed_control_fields
from ..benchmark.config import load_yaml, save_json
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.select import _collect_condition_seed_rows
from ..util.figure import _resolve_path

logger = logging.getLogger(__name__)


def _spectrum_metrics(s, tau):
    s = np.asarray(s, float); above = s >= tau
    E = int(above.sum()); lam = s ** 2
    PR = float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-30))
    sa = s[above]
    kappa = float(sa.max() / sa.min()) if sa.size else float("nan")
    return E, PR, kappa


def _whiten_svd(J, I0, local, shot, floor, meas_scale=1.0):
    """Whitened SVD for a given per-(local channel) shot coeff + floor. Returns s, Vt, sigma, fisher_diag, frac_floor.
    meas_scale = √(N_full/n) scales the whitened operator (leaves returned sigma physical)."""
    a = np.asarray(shot, float); b = np.asarray(floor, float)
    shot_var = a[local] * np.clip(I0, 0.0, None)
    var = shot_var + b[local] ** 2
    sigma = np.sqrt(np.maximum(var, 1e-30))
    Jw = J / sigma[:, None] * float(meas_scale)
    _, s, Vt = np.linalg.svd(Jw, full_matrices=False)                 # columns (M) unchanged; rows re-whitened
    fisher_diag = (Jw ** 2).sum(0)                                    # diag(J~^T J~): per-field-cell Fisher info
    frac_floor = float((b[local] ** 2 > shot_var).mean())            # where the floor dominates the shot variance
    return s.astype(np.float64), Vt.astype(np.float64), sigma, fisher_diag.astype(np.float64), frac_floor


def _spearman(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if a.size < 3: return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def _floor_vecs(cfg_floors, med_by_local, shot_eff, n_local):
    """Expand yaml floor specs -> [(id, per-local-channel beta array)].
    Types: sigma_floor (absolute), frac_of_median (of channel median intensity),
           frac_of_shot_sigma (of sqrt(alpha_eff * median) = the median shot sigma; reliably bites)."""
    shot_eff = np.asarray(shot_eff, float)
    out = []
    for f in cfg_floors:
        fid = str(f["id"])
        if "sigma_floor" in f:
            v = f["sigma_floor"]
            beta = np.full(n_local, float(v)) if np.isscalar(v) else np.asarray(v, float)
        elif "frac_of_median" in f:
            beta = float(f["frac_of_median"]) * med_by_local
        elif "frac_of_shot_sigma" in f:
            beta = float(f["frac_of_shot_sigma"]) * np.sqrt(shot_eff * np.clip(med_by_local, 0.0, None))
        else:
            beta = np.zeros(n_local)
        out.append((fid, beta.astype(float)))
    return out


def _write_tables(order, per, outdir):
    cols = [("E_tau", "E(tau)", lambda v: f"{int(v)}"),
            ("PR", "PR", lambda v: f"{v:.1f}"),
            ("kappa", "kappa", lambda v: f"{v:.0f}"),
            ("f_null_ens", "f_null^ens", lambda v: f"{v:.3f}"),
            ("f_null_err", "f_null^err", lambda v: f"{v:.3f}"),
            ("f_null_rand", "f_null^rand", lambda v: f"{v:.3f}"),
            ("fisher_rho_vs_ref", "rho(Fisher-diag)", lambda v: f"{v:.4f}"),
            ("frac_floor_dominated", "frac floor-dom.", lambda v: f"{v:.3f}")]
    # markdown
    md = ["| floor | " + " | ".join(h for _, h, _ in cols) + " |",
          "|" + "---|" * (len(cols) + 1)]
    for fid in order:
        m = per[fid]
        md.append("| " + fid + " | " + " | ".join(fmt(m[k]) for k, _, fmt in cols) + " |")
    (outdir / "floor_sweep_table.md").write_text("\n".join(md) + "\n")
    # latex
    tex = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           "floor & " + " & ".join(h for _, h, _ in cols) + r" \\", r"\midrule"]
    for fid in order:
        m = per[fid]
        tex.append(fid.replace("_", r"\_") + " & " + " & ".join(fmt(m[k]) for k, _, fmt in cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "floor_sweep_table.tex").write_text("\n".join(tex) + "\n")


def generate_floor_sweep(spec_path, device_override=None, output_dir_override=None):
    spec_path = Path(spec_path).resolve(); root = spec_path.parent
    cfg = load_yaml(spec_path).get("floor_sweep", {})
    out_root = _resolve_path(cfg.get("output_root", "../../runs_identifiability"), root)
    device = device_override if device_override is not None else cfg.get("device")
    tau = float(cfg.get("tau", 1.0)); target = str(cfg.get("target", "ne_t"))
    name = str(cfg.get("name", "floor_sweep"))
    outdir = Path(output_dir_override) if output_dir_override else (out_root / "floors" / name)
    outdir.mkdir(parents=True, exist_ok=True)

    base = cfg["base"]
    cond = dict(cfg["condition"]); eta = float(cond.get("eta", 1.0))

    # ---- cached operator at the GT operating point (m*): J is noise-independent, so it is NOT rebuilt ----
    opcfg = copy.deepcopy(base)
    opcfg["operating_point"] = dict(cfg.get("operating_point", {"id": "mstar", "kind": "gt"}))
    O = load_operator(build_operator_from_cfg(opcfg, root, device_override=device, output_root=out_root))
    J = np.asarray(O["J"]); I0 = np.asarray(O["I0"]).reshape(-1)
    meta = O["_meta"]; base_shot = list(meta["base_shot_coeff"])
    sel = [int(x) for x in meta["selected_channel_indices"]]
    g2l = {g: i for i, g in enumerate(sel)}
    local = np.array([g2l[int(g)] for g in np.asarray(O["obs_chan_global"])], dtype=int)
    mstar = np.asarray(O["control_values"]).reshape(-1)
    shot_c = np.array([s * eta for s in base_shot], float)            # apply the condition's noise multiplier
    n_local = len(sel)
    med_by_local = np.array([float(np.median(I0[local == l])) if (local == l).any() else 0.0
                             for l in range(n_local)])

    # ---- ensemble m_hat_k for the condition (loads seed checkpoints; no Jacobian) ----
    bdir = _resolve_path(base["base"]["benchmark_dir"], root)
    ens_sel = {k: cond[k] for k in ("experiment", "where", "run_names", "min_seeds", "max_seeds") if k in cond}
    mhats = _seed_control_fields(_collect_condition_seed_rows(collect_run_summaries(bdir), ens_sel),
                                 bdir, dict(base.get("grid", {})), target, device, base.get("path_remap"))

    floors = _floor_vecs(cfg.get("floors", [{"id": "shot_only",   "sigma_floor": 0.0},
                                        {"id": "bg_2pct",      "frac_of_median": 0.02},       # realistic background
                                        {"id": "sig_0p5",      "frac_of_shot_sigma": 0.5},    # ~half the median shot sigma
                                        {"id": "sig_1p0",      "frac_of_shot_sigma": 1.0}]),  # ~the median shot sigma
                        med_by_local, shot_c, n_local)

    per, order, fisher_ref = {}, [], None
    ms = float(O["_meta"].get("meas_scale", 1.0))                     # stamped by build_operator_from_cfg
    for fid, beta in floors:
        s, Vt, sigma, fdiag, frac_floor = _whiten_svd(J, I0, local, shot_c, beta, meas_scale=ms)
        E, PR, kappa = _spectrum_metrics(s, tau)
        proj = _project(s, Vt, mstar, mhats, tau)
        if fisher_ref is None: fisher_ref = fdiag                     # first entry (shot_only) is the reference
        m = {"beta_by_channel": beta.tolist(), "E_tau": int(E), "PR": float(PR), "kappa": float(kappa),
             "f_null_ens": float(proj["f_null_ens"]), "f_null_err": float(proj["f_null_err"]),
             "f_null_rand": float(proj["f_null_rand"]), "e_capture": float(proj["e_capture"]),
             "fisher_rho_vs_ref": _spearman(fdiag, fisher_ref), "frac_floor_dominated": frac_floor,
             "sigma_min": float(sigma.min()), "n_null": int(proj["n_null"]), "M": int(proj["M"])}
        per[fid] = m; order.append(fid)
        np.savez_compressed(outdir / f"{fid}.npz", s=s.astype(np.float32), fisher_diag=fdiag.astype(np.float32),
                            beta_by_channel=beta, **{k: np.asarray(v) for k, v in m.items() if k != "beta_by_channel"})
        logger.info("floor[%s] beta=%s: E=%d PR=%.1f kappa=%.0f fN_ens=%.3f fN_err=%.3f Ecap=%.3f "
                    "fisher_rho=%.4f floor_dom=%.2f", fid, np.array2string(beta, precision=3),
                    E, PR, kappa, m["f_null_ens"], m["f_null_err"], m["e_capture"], m["fisher_rho_vs_ref"], frac_floor)

    man = {"name": name, "condition": cond.get("id"), "eta": eta, "tau": tau,
           "sigma_evaluated_at": opcfg["operating_point"].get("id", "mstar"),
           "median_intensity_by_channel": med_by_local.tolist(),
           "shot_coeff_effective": shot_c.tolist(), "selected_channels": sel,
           "floors": per, "order": order}
    save_json(man, outdir / "floor_sweep_manifest.json")
    _write_tables(order, per, outdir)
    logger.info("floor-sweep '%s' -> %s", name, outdir)
    return man