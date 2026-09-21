from __future__ import annotations
import copy, logging
from pathlib import Path
import numpy as np
from .operator import build_operator_from_cfg, load_operator, subset_operator, pick_views
from ._core import build_control_field
from ..benchmark.config import load_yaml, save_json
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.select import _collect_condition_seed_rows
from ..benchmark.condition_compare import _build_state_for_analysis, _resolve_run_dir
from ..util.figure import _resolve_path, _savefig

logger = logging.getLogger(__name__)

def _seed_control_fields(seed_rows, benchmark_dir, grid_cfg, target, device, path_remap):
    """Each seed's recovered field on the SAME control grid -> (K, M)."""
    mats = []
    for row in seed_rows:
        run_dir = _resolve_run_dir(row, benchmark_dir)
        state = _build_state_for_analysis(run_dir, device_override=device, path_remap=path_remap)
        bc = build_control_field(state, dict(grid_cfg or {}), target, state.renderer.device, source="recon")
        mats.append(bc.values.detach().cpu().numpy().reshape(-1).astype(np.float64))
        del state
    return np.stack(mats, 0)                                             # (K, M)


def _project(s, Vt, mstar, mhats, tau):
    K = mhats.shape[0]; M = Vt.shape[1]; mbar = mhats.mean(0)
    Delta = (mhats - mbar).T                                            # (M, K)  deviations delta_k
    C = Vt @ Delta                                                      # (R, K)
    g = (C ** 2).sum(1) / max(K - 1, 1)                                 # (R,)
    total = float((Delta ** 2).sum() / max(K - 1, 1))
    outside = max(total - float(g.sum()), 0.0)                          # energy in exact-null dims beyond R
    null = s < tau
    f_null = float((g[null].sum() + outside) / (total + 1e-30))
    eps = mbar - mstar; c_eps = Vt @ eps
    et = float((eps ** 2).sum()); eo = max(et - float((c_eps ** 2).sum()), 0.0)
    f_null_err = float(((c_eps[null] ** 2).sum() + eo) / (et + 1e-30))
    f_null_rand = float((int(null.sum()) + (M - s.size)) / M)           # isotropic baseline

    # ---- E_capture: fraction of the signed error inside the ensemble-deviation span S_ens = span{delta_k} ----
    # Q = orthonormal basis of range(Delta) (== support of C_ens), dim r <= K-1. Direct overlap of eps with S_ens.
    Ud, sd, _ = np.linalg.svd(Delta, full_matrices=False)              # Ud: (M, K), sd: (K,)
    tol = float(sd.max()) * 1e-8 if sd.size else 0.0
    r_ens = int((sd > tol).sum())
    Q = Ud[:, :r_ens] if r_ens > 0 else np.zeros((M, 0), dtype=Delta.dtype)
    e_capture = float(((Q.T @ eps) ** 2).sum() / (et + 1e-30))          # ||P_ens eps||^2 / ||eps||^2
    e_capture_rand = float(r_ens / max(M, 1))                           # random r-dim subspace baseline r/M
    e_capture_enrich = float(e_capture / (e_capture_rand + 1e-30))

    return {"s": s, "g_ens": g, "c_eps": c_eps.astype(np.float64),
            "f_null_ens": f_null, "f_null_err": f_null_err, "f_null_rand": f_null_rand,
            "e_capture": e_capture, "e_capture_rand": e_capture_rand,
            "e_capture_enrich": e_capture_enrich, "r_ens": int(r_ens),
            "tau": float(tau), "n_seeds": int(K), "M": int(M), "n_null": int(null.sum())}


def _fig(res, out_base, cfg):
    import matplotlib.pyplot as plt
    s, g = res["s"], res["g_ens"]; order = np.argsort(s)[::-1]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    ax[0].loglog(np.clip(s, 1e-12, None), np.clip(g, 1e-30, None), ".", ms=3, alpha=0.4)
    ax[0].axvline(res["tau"], color="r", ls="--", lw=1)
    ax[0].set_xlabel(r"singular value $s_\ell$"); ax[0].set_ylabel(r"ensemble energy $g_\ell$")
    ax[0].set_title("energy vs identifiability")
    cg = np.cumsum(g[order]) / (g.sum() + 1e-30)
    ax[1].plot(np.arange(1, g.size + 1), cg); ax[1].axvline(int((s >= res["tau"]).sum()), color="0.5", ls=":")
    ax[1].set_xlabel(r"mode (sorted by $s_\ell$)"); ax[1].set_ylabel("cumulative ensemble energy")
    ax[1].set_title(f"$f_{{\\rm null}}$(ens) = {res['f_null_ens']:.2f}")
    ax[2].bar(["ensemble", "true error", "random"],
              [res["f_null_ens"], res["f_null_err"], res["f_null_rand"]], color=["#1f77b4", "#d62728", "0.6"])
    ax[2].set_ylim(0, 1); ax[2].set_ylabel("fraction in near-null space"); ax[2].set_title("null-space fraction")
    # E_capture: fraction of the signed error inside the ensemble-deviation span vs a random r-dim subspace
    ec, ecr = float(res["e_capture"]), float(res["e_capture_rand"])
    ax[3].bar(["ensemble span", "random $r/M$"], [max(ec, 1e-9), max(ecr, 1e-9)], color=["#1f77b4", "0.6"])
    ax[3].set_yscale("log"); ax[3].set_ylabel(r"error captured $E_{\rm cap}$")
    ax[3].set_title(f"$E_{{\\rm cap}}$ = {ec:.3f}  ({res['e_capture_enrich']:.0f}$\\times$)")
    fig.tight_layout()
    return _savefig(fig, out_base, cfg.get("formats", ["png", "pdf"]), int(cfg.get("dpi", 200)))


def _svd_ctrl(op, shot_coeff, floor):
    """Whiten a (possibly subsetted) operator with the given per-channel shot/floor -> (s, Vt, m*_flat)."""
    meta = op["_meta"]; J = np.asarray(op["J"]); I0 = np.asarray(op["I0"]).reshape(-1)
    sel = [int(x) for x in meta["selected_channel_indices"]]
    g2l = {g: i for i, g in enumerate(sel)}
    local = np.array([g2l[int(g)] for g in np.asarray(op["obs_chan_global"])], dtype=int)
    a = np.asarray(shot_coeff, float); b = np.asarray(floor, float)
    var = a[local] * np.clip(I0, 0.0, None) + b[local] ** 2
    sigma = np.sqrt(np.maximum(var, 1e-30))
    ms = float(op["_meta"].get("meas_scale", 1.0))                        # √(N_full/n); 1.0 when disabled
    _, s, Vt = np.linalg.svd(J / sigma[:, None] * ms, full_matrices=False)  # scale J̃ -> full operator; V unchanged
    mstar = np.asarray(op["control_values"]).reshape(-1)
    return s.astype(np.float64), Vt.astype(np.float64), mstar.astype(np.float64)


def generate_mode_projection(spec_path, device_override=None, output_dir_override=None):
    spec_path = Path(spec_path).resolve(); root = spec_path.parent
    cfg = load_yaml(spec_path).get("mode_projection", {})
    bdir = _resolve_path(cfg["benchmark_dir"], root)
    device = device_override if device_override is not None else cfg.get("device")
    path_remap = cfg.get("path_remap"); tau = float(cfg.get("tau", 1.0))
    target = cfg.get("target", "ne_t")
    grid_cfg = dict(cfg["base"].get("grid", {}))                          # control grid == base's grid
    op_out = _resolve_path(cfg.get("output_root", "../../runs_identifiability"), root)
    out_root = Path(output_dir_override) if output_dir_override else (bdir / "diagnostics" / "mode_projection")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- base operator ONCE (reuses operators/paperF_base__… ; no rebuild) ----
    base_op = load_operator(build_operator_from_cfg(dict(cfg["base"]), root, device_override=device, output_root=op_out))
    base_shot = list(base_op["_meta"]["base_shot_coeff"])
    base_floor = list(base_op["_meta"].get("base_sigma_floor", [0.0] * len(base_shot)))

    summaries = collect_run_summaries(bdir)
    man = {"name": cfg.get("name", "mode_projection"), "conditions": []}
    for cond in cfg["conditions"]:
        cid = str(cond["id"]); eta = float(cond.get("eta", 1.0)); mode = str(cond.get("mode", "derive"))
        channels = cond.get("channels", None); views = int(cond.get("views", int(base_op["vidx"].size)))
        if mode == "build":                                              # abundance: genuine new forward op
            op = load_operator(build_operator_from_cfg(dict(cond["operator"]), root, device_override=device, output_root=op_out))
            shot_c = list(op["_meta"]["base_shot_coeff"]); floor = list(op["_meta"].get("base_sigma_floor", [0.0] * len(shot_c)))
        else:                                                            # noise/views: subset + re-whiten (no rebuild)
            vs = pick_views(base_op["vidx"], views) if views < int(base_op["vidx"].size) else None
            op = subset_operator(base_op, view_subset=vs, channel_subset=channels)
            shot_c = [s * eta for s in base_shot]; floor = base_floor
        s, Vt, mstar = _svd_ctrl(op, shot_c, floor)

        sel = {k: cond[k] for k in ("experiment", "where", "run_names", "min_seeds", "max_seeds") if k in cond}
        mhats = _seed_control_fields(_collect_condition_seed_rows(summaries, sel), bdir, grid_cfg, target, device, path_remap)
        res = _project(s, Vt, mstar, mhats, tau)

        cdir = out_root / cid; cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cdir / "mode_projection.npz", **{k: res[k] for k in
            ("s", "g_ens", "c_eps", "f_null_ens", "f_null_err", "f_null_rand",
             "e_capture", "e_capture_rand", "e_capture_enrich", "r_ens",
             "tau", "n_seeds", "M", "n_null")})
        _fig(res, cdir / "mode_projection", cfg)
        man["conditions"].append({"id": cid, "eta": eta, **{k: res[k] for k in
            ("f_null_ens", "f_null_err", "f_null_rand",
             "e_capture", "e_capture_rand", "e_capture_enrich", "r_ens",
             "n_seeds", "n_null", "M")}})
        logger.info("mode_projection[%s] eta=%g: f_null ens=%.3f err=%.3f rand=%.3f | E_cap=%.3f rand=%.4f (%.0fx)",
                    cid, eta, res["f_null_ens"], res["f_null_err"], res["f_null_rand"],
                    res["e_capture"], res["e_capture_rand"], res["e_capture_enrich"])
    save_json(man, out_root / "mode_projection_manifest.json")
    return man