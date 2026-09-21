from __future__ import annotations
import copy, logging
from pathlib import Path
import numpy as np

from .operator import build_operator_from_cfg, load_operator
from .mode_projection import _svd_ctrl, _project, _seed_control_fields
from ..benchmark.config import load_yaml, save_json
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.select import _collect_condition_seed_rows
from ..util.figure import _resolve_path, _savefig

logger = logging.getLogger(__name__)


def _spectrum_metrics(s: np.ndarray, tau: float) -> tuple[int, float, float]:
    """Effective rank E(tau), participation ratio PR, condition number kappa(tau) of the identifiable modes."""
    s = np.asarray(s, float)
    above = s >= tau
    E = int(above.sum())
    lam = s ** 2
    PR = float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-30))
    sa = s[above]
    kappa = float(sa.max() / sa.min()) if sa.size else float("nan")
    return E, PR, kappa


def _fig_oppoint_spectrum(recs, per_op, ref_id, out_base, cfg):
    """Diagnostic: (a) spectrum overlay m* vs m̂ mean±std, (b) per-mode scatter s(m̂) vs s(m*),
       (c) derived diagnostics (E/f_null/E_cap) across operating points."""
    import matplotlib.pyplot as plt
    ref = next(r for r in recs if r["id"] == ref_id)
    mhats = [r for r in recs if r["id"] != ref_id]
    n = min(len(r["s"]) for r in recs)
    x = np.arange(1, n + 1)
    s_ref = np.clip(np.asarray(ref["s"])[:n], 1e-12, None)
    S = np.stack([np.clip(np.asarray(r["s"])[:n], 1e-12, None) for r in mhats], 0) if mhats else s_ref[None]
    s_mean = S.mean(0); s_std = S.std(0, ddof=1) if S.shape[0] > 1 else np.zeros_like(s_mean)
    tau = float(cfg.get("tau", 1.0))

    fig, ax = plt.subplots(1, 3, figsize=tuple(cfg.get("figsize", [15, 4.2])))

    # (a) spectrum overlay
    ax[0].fill_between(x, np.clip(s_mean - s_std, 1e-12, None), s_mean + s_std, color="#1f77b4", alpha=0.3,
                       label=r"$\hat m$ mean$\pm$std")
    ax[0].loglog(x, s_mean, color="#1f77b4", lw=1.0)
    ax[0].loglog(x, s_ref, color="k", lw=1.2, label=r"$m^*$ (GT)")
    ax[0].axhline(tau, color="r", ls="--", lw=1, label=r"$\tau=1$")
    ax[0].set_xlabel(r"mode index $\ell$"); ax[0].set_ylabel(r"singular value $s_\ell$")
    ax[0].set_title("whitened spectrum: GT vs recovered"); ax[0].legend(fontsize=8)

    # (b) per-mode agreement
    a, b = s_ref, s_mean
    ax[1].loglog(a, b, ".", ms=2, alpha=0.3)
    lim = [min(a.min(), b.min()), max(a.max(), b.max())]
    ax[1].plot(lim, lim, "k--", lw=1); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    rho = float(np.corrcoef(np.log(a), np.log(b))[0, 1]); medr = float(np.median(b / a))
    ax[1].set_xlabel(r"$s_\ell(m^*)$"); ax[1].set_ylabel(r"$s_\ell(\hat m)$")
    ax[1].set_title(f"per-mode agreement  ($\\rho$={rho:.4f}, median ratio={medr:.3f})")

    # (c) derived diagnostics across operating points
    ids = [r["id"] for r in recs]
    mets = ["E_rel", "f_null_ens", "f_null_err", "e_capture"]
    labels = {"E_rel": r"$E(\tau)/E^*(\tau)$", "f_null_ens": r"$f_{\rm null}^{\rm ens}$",
              "f_null_err": r"$f_{\rm null}^{\rm err}$", "e_capture": r"$E_{\rm cap}$"}
    E_star = float(per_op[ref_id]["E_tau"])
    xb = np.arange(len(ids)); w = 0.2
    for j, mt in enumerate(mets):
        if mt == "E_rel":
            vals = [per_op[i]["E_tau"] / (E_star + 1e-30) for i in ids]
        else:
            vals = [per_op[i][mt] for i in ids]
        ax[2].bar(xb + j * w, vals, w, label=labels[mt])
    ax[2].axhline(1.0, color="0.7", lw=0.8, zorder=0)
    ax[2].set_xticks(xb + 1.5 * w); ax[2].set_xticklabels(ids, rotation=30, ha="right", fontsize=7)
    ax[2].set_title("derived diagnostics across operating points"); ax[2].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    return _savefig(fig, out_base, cfg.get("formats", ["png", "pdf"]), int(cfg.get("dpi", 200)))


def generate_oppoint_spectrum(spec_path, device_override=None, output_dir_override=None):
    """For each condition: linearize about m* and 2-3 m̂ operating points (cached operators), and compute the
    whitened-spectrum diagnostics E(tau)/PR/kappa + f_null(ens/err/rand) + E_cap. The ensemble and the error
    (eps = mbar - m*) are held fixed; only the linearization point (i.e. the SVD modes) changes."""
    spec_path = Path(spec_path).resolve(); root = spec_path.parent
    cfg = load_yaml(spec_path).get("oppoint_spectrum", {})
    out_root = _resolve_path(cfg.get("output_root", "../../runs_identifiability"), root)
    device = device_override if device_override is not None else cfg.get("device")
    tau = float(cfg.get("tau", 1.0)); target = str(cfg.get("target", "ne_t"))
    name = str(cfg.get("name", "oppoint_spectrum"))
    outdir = Path(output_dir_override) if output_dir_override else (out_root / "oppoints_spectrum" / name)
    outdir.mkdir(parents=True, exist_ok=True)

    base = cfg["base"]
    grid_cfg = dict(base.get("grid", {}))
    path_remap = base.get("path_remap")
    bdir = _resolve_path(base["base"]["benchmark_dir"], root)             # nested: base.base.benchmark_dir
    summaries = collect_run_summaries(bdir)

    man = {"name": name, "benchmark_dir": str(bdir), "tau": tau, "conditions": []}
    for cond in cfg["conditions"]:
        cid = str(cond["id"]); eta = float(cond.get("eta", 1.0))

        # ---- ensemble m̂_k on the control grid (loads seed checkpoints; NO Jacobian) ----
        sel = {k: cond[k] for k in ("experiment", "where", "run_names", "min_seeds", "max_seeds") if k in cond}
        seed_rows = _collect_condition_seed_rows(summaries, sel)
        mhats = _seed_control_fields(seed_rows, bdir, grid_cfg, target, device, path_remap)   # (K, M)

        # ---- per operating point: cached operator -> whitened SVD -> spectrum + projection ----
        recs = []; mstar_gt = None; ref_id = None
        for op in cfg["operating_points"]:
            c = copy.deepcopy(base); c["base"] = dict(c.get("base", {}))
            for k in ("experiment", "where", "run_names", "min_seeds", "max_seeds"):
                if k in cond:
                    c["base"][k] = cond[k]
            c["operating_point"] = dict(op)
            O = load_operator(build_operator_from_cfg(c, root, device_override=device, output_root=out_root))
            base_shot = list(O["_meta"]["base_shot_coeff"])
            base_floor = list(O["_meta"].get("base_sigma_floor", [0.0] * len(base_shot)))
            shot_c = [x * eta for x in base_shot]                          # apply the condition's noise multiplier
            s, Vt, mstar_op = _svd_ctrl(O, shot_c, base_floor)
            if str(op.get("kind", "gt")) == "gt" or str(op["id"]) == "mstar":
                mstar_gt = mstar_op; ref_id = str(op["id"])
            recs.append({"id": str(op["id"]), "kind": str(op.get("kind", "gt")), "s": s, "Vt": Vt})
        if mstar_gt is None:                                              # fallback if no explicit GT point
            _, _, mstar_gt = _svd_ctrl(load_operator(build_operator_from_cfg(
                {**base, "operating_point": dict(cfg["operating_points"][0])}, root,
                device_override=device, output_root=out_root)),
                [x * eta for x in base_shot], base_floor)
            ref_id = recs[0]["id"]

        # ---- metrics per operating point (GT-referenced error + fixed ensemble; op modes) ----
        per_op = {}
        for r in recs:
            E, PR, kappa = _spectrum_metrics(r["s"], tau)
            proj = _project(r["s"], r["Vt"], mstar_gt, mhats, tau)
            per_op[r["id"]] = {
                "kind": r["kind"], "E_tau": int(E), "PR": float(PR), "kappa": float(kappa),
                "f_null_ens": float(proj["f_null_ens"]), "f_null_err": float(proj["f_null_err"]),
                "f_null_rand": float(proj["f_null_rand"]),
                "e_capture": float(proj["e_capture"]), "e_capture_rand": float(proj["e_capture_rand"]),
                "e_capture_enrich": float(proj["e_capture_enrich"]),
                "n_null": int(proj["n_null"]), "M": int(proj["M"]), "n_seeds": int(proj["n_seeds"]),
            }

        # ---- save npz (spectra + metrics) ----
        npz = {f"s__{r['id']}": np.asarray(r["s"], np.float32) for r in recs}
        for oid, m in per_op.items():
            for k, v in m.items():
                if k != "kind":
                    npz[f"{k}__{oid}"] = np.asarray(v)
        npz["op_ids"] = np.array([r["id"] for r in recs])
        npz["op_kinds"] = np.array([r["kind"] for r in recs])
        npz["ref_id"] = np.array(ref_id); npz["tau"] = np.array(tau)
        np.savez_compressed(outdir / f"{cid}.npz", **npz)

        _fig_oppoint_spectrum(recs, per_op, ref_id, outdir / f"{cid}_diag", cfg)

        man["conditions"].append({"id": cid, "eta": eta, "ref_id": ref_id, "operating_points": per_op})
        logger.info("oppoint_spectrum[%s] tau=%.2f | " + " | ".join(
            f"{oid}: E={m['E_tau']} PR={m['PR']:.1f} fN_ens={m['f_null_ens']:.3f} fN_err={m['f_null_err']:.3f} "
            f"Ecap={m['e_capture']:.3f}" for oid, m in per_op.items()), cid, tau)

    save_json(man, outdir / "oppoint_spectrum_manifest.json")
    logger.info("oppoint-spectrum -> %s", outdir)
    return man