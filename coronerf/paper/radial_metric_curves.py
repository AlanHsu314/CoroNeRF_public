from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..artifacts.scalar_fields import evaluate_scalar_shells, get_scalar_field_spec
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..benchmark.condition_compare import _build_state_for_analysis, _release_state
from .shell_panels import (
    _metric_value,
    _resolve_path,
    _resolve_run_dir,
    _row_matches_where,
    _seed_equal,
    _slug,
)

logger = logging.getLogger("coroNeRF.paper.radial_metric_curves")


_DEFAULT_BINS = {
    "low_corona": (1.1, 1.5),
    "inner_mid": (1.5, 2.0),
    "inner_outer": (2.0, 2.6),
    "outer_fov": (2.6, 3.0),
    "los_outer": (3.0, 4.0),
}


def _linspace_values(cfg: dict) -> list[float]:
    start = float(cfg["start"])
    stop = float(cfg["stop"])
    num = int(cfg["num"])
    if num <= 0:
        raise ValueError("radii.linspace.num must be positive")
    return [float(x) for x in np.linspace(start, stop, num)]


def _resolve_radii(radii_cfg: Any) -> tuple[list[float], list[str] | None]:
    """
    Return requested shell radii and optional x tick labels.

    Supported forms:
      radii: [1.1, 1.2, 1.3]
      radii: {values: [1.1, 1.2]}
      radii: {linspace: {start: 1.1, stop: 2.0, num: 10}}
      radii: {bins: [{label: "1.1-1.3", r_min: 1.1, r_max: 1.3}, ...]}

    The bin mode uses the bin centers as x positions. It is intentionally
    incremental/non-overlapping, unlike the cumulative inner/inner+/full bands.
    """
    if radii_cfg is None:
        return [float(x) for x in np.linspace(1.1, 2.0, 10)], None

    if isinstance(radii_cfg, (list, tuple)):
        return [float(x) for x in radii_cfg], None

    if not isinstance(radii_cfg, dict):
        raise TypeError(f"Unsupported radii config: {radii_cfg!r}")

    if "values" in radii_cfg:
        return [float(x) for x in radii_cfg["values"]], None

    if "linspace" in radii_cfg:
        return _linspace_values(dict(radii_cfg["linspace"] or {})), None

    if "bins" in radii_cfg:
        radii = []
        labels = []
        for b in list(radii_cfg["bins"] or []):
            if isinstance(b, str):
                if b not in _DEFAULT_BINS:
                    raise KeyError(f"Unknown default radial bin {b!r}; available={sorted(_DEFAULT_BINS)}")
                r_min, r_max = _DEFAULT_BINS[b]
                label = b.replace("_", " ")
            else:
                b = dict(b)
                r_min = float(b["r_min"])
                r_max = float(b["r_max"])
                label = str(b.get("label", f"{r_min:g}-{r_max:g}"))
            radii.append(0.5 * (float(r_min) + float(r_max)))
            labels.append(label)
        return radii, labels

    if "default_bins" in radii_cfg:
        names = list(radii_cfg.get("default_bins") or _DEFAULT_BINS.keys())
        bins = []
        for name in names:
            r_min, r_max = _DEFAULT_BINS[str(name)]
            bins.append({"label": str(name).replace("_", " "), "r_min": r_min, "r_max": r_max})
        return _resolve_radii({"bins": bins})

    raise ValueError(f"Could not interpret radii config: {radii_cfg}")


def _select_rows_for_experiment(
    rows: list[dict],
    exp_cfg: dict,
    default_metric: str,
    default_policy: str,
    default_mode: str,
) -> list[dict]:
    exp = exp_cfg.get("experiment", exp_cfg.get("name", None))
    if exp is None and exp_cfg.get("run_dir", None) is None and exp_cfg.get("run_name", None) is None:
        raise ValueError("Each experiment entry needs experiment/name, run_name, or run_dir")

    if exp_cfg.get("run_dir", None) is not None:
        return [{
            "status": "completed",
            "run_dir": str(exp_cfg["run_dir"]),
            "run_name": Path(str(exp_cfg["run_dir"])).name,
            "experiment_name": exp or exp_cfg.get("label", Path(str(exp_cfg["run_dir"])).name),
            "seed": exp_cfg.get("seed", None),
        }]

    if exp_cfg.get("run_name", None) is not None:
        run_name = str(exp_cfg["run_name"])
        matches = [r for r in rows if str(r.get("run_name")) == run_name]
    else:
        matches = [r for r in rows if str(r.get("experiment_name")) == str(exp)]

    matches = [r for r in matches if _row_matches_where(r, exp_cfg.get("where", None) or exp_cfg.get("filters", None))]

    seeds = exp_cfg.get("seeds", None)
    if seeds is not None:
        matches = [r for r in matches if any(_seed_equal(r.get("seed"), s) for s in list(seeds))]

    seed = exp_cfg.get("seed", None)
    if seed is not None:
        matches = [r for r in matches if _seed_equal(r.get("seed"), seed)]

    if not matches:
        raise ValueError(f"No completed rows matched experiment entry: {exp_cfg}")

    policy = str(exp_cfg.get("selection_policy", default_policy))
    metric_key = str(exp_cfg.get("selection_metric", default_metric))
    mode = str(exp_cfg.get("select_mode", default_mode))

    if policy == "all":
        selected = list(matches)
    elif policy == "first":
        selected = [sorted(matches, key=lambda r: str(r.get("run_name")))[0]]
    elif policy == "best":
        metric_matches = [r for r in matches if r.get(metric_key) is not None]
        if not metric_matches:
            raise ValueError(f"No matches for {exp_cfg} contain selection_metric={metric_key}")
        selected = [
            min(metric_matches, key=lambda r: _metric_value(r, metric_key))
            if mode == "min"
            else max(metric_matches, key=lambda r: _metric_value(r, metric_key))
        ]
    elif policy == "median":
        metric_matches = [r for r in matches if r.get(metric_key) is not None]
        if not metric_matches:
            raise ValueError(f"No matches for {exp_cfg} contain selection_metric={metric_key}")
        vals = np.asarray([_metric_value(r, metric_key) for r in metric_matches], dtype=float)
        med = float(np.median(vals))
        selected = [min(metric_matches, key=lambda r: abs(_metric_value(r, metric_key) - med))]
    else:
        raise ValueError(f"Unknown selection_policy={policy!r}")

    max_runs = exp_cfg.get("max_runs", None)
    if max_runs is not None:
        selected = selected[: int(max_runs)]

    return selected


def _shell_metric_from_row(shell_row: dict, metric: str) -> float:
    if metric not in shell_row:
        raise KeyError(f"Shell metric {metric!r} not in row; available={sorted(shell_row)}")
    val = shell_row.get(metric, None)
    return float(val) if val is not None else float("nan")


def _evaluate_run_radial(
    row: dict,
    benchmark_dir: Path,
    quantities: list[str],
    radii: list[float],
    metric: str,
    device: str | None,
    path_remap: dict | None,
) -> tuple[list[dict], Path]:
    run_dir = _resolve_run_dir(row, benchmark_dir)
    if run_dir is None:
        raise FileNotFoundError(f"Could not resolve run_dir for row: {row}")

    logger.info("Evaluating radial curves for run: %s", run_dir)
    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=path_remap)

    records: list[dict] = []
    try:
        for quantity in quantities:
            result = evaluate_scalar_shells(state, quantity=quantity, radii=radii, metric_bands={})
            if not result.get("available", False):
                logger.warning("Skipping unavailable quantity=%s for run=%s", quantity, run_dir)
                continue
            for r_req, shell_row in zip(radii, result["shells"]):
                records.append({
                    "quantity": quantity,
                    "r_requested": float(r_req),
                    "r_actual": float(shell_row.get("r", np.nan)),
                    "r": float(r_req),
                    "metric": metric,
                    "value": _shell_metric_from_row(shell_row, metric),
                    "experiment_name": row.get("experiment_name"),
                    "run_name": row.get("run_name"),
                    "seed": row.get("seed"),
                    "run_dir": str(run_dir),
                })
    finally:
        _release_state(state)

    return records, run_dir


def _aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    meta: dict[tuple, dict] = {}
    for rec in records:
        key = (rec["curve_id"], rec["curve_label"], rec["quantity"], float(rec["r"]))
        groups.setdefault(key, []).append(float(rec["value"]))
        meta[key] = {
            "curve_id": rec["curve_id"],
            "curve_label": rec["curve_label"],
            "quantity": rec["quantity"],
            "r": float(rec["r"]),
        }

    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3])):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        m = dict(meta[key])
        m.update({
            "n": int(arr.size),
            "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
            "std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
        })
        out.append(m)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_radial_curves(
    aggregates: list[dict],
    quantity_cfgs: list[dict],
    experiment_cfgs: list[dict],
    out_base: Path,
    plot_cfg: dict,
    radii: list[float],
    tick_labels: list[str] | None,
) -> dict:
    n_cols = len(quantity_cfgs)
    figsize = plot_cfg.get("figsize", [6.6, 2.8])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])

    rc = {
        "font.size": float(plot_cfg.get("font_size", 8)),
        "axes.titlesize": float(plot_cfg.get("title_font_size", 9)),
        "axes.labelsize": float(plot_cfg.get("font_size", 8)),
        "xtick.labelsize": float(plot_cfg.get("tick_font_size", 7)),
        "ytick.labelsize": float(plot_cfg.get("tick_font_size", 7)),
    }

    agg_by_key = {
        (a["curve_id"], a["quantity"], float(a["r"])): a
        for a in aggregates
    }

    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, n_cols, figsize=figsize, squeeze=False, constrained_layout=True)

        for j, qcfg in enumerate(quantity_cfgs):
            quantity = str(qcfg["quantity"])
            spec = get_scalar_field_spec(quantity)
            ax = axs[0, j]

            for exp_idx, exp_cfg in enumerate(experiment_cfgs):
                curve_id = str(exp_cfg.get("id", exp_cfg.get("experiment", exp_cfg.get("name", exp_idx))))
                label = str(exp_cfg.get("label", exp_cfg.get("experiment", exp_cfg.get("name", curve_id))))
                style = dict(exp_cfg.get("style", {}) or {})

                xs, ys, es = [], [], []
                for r in sorted(set(float(x) for x in radii)):
                    a = agg_by_key.get((curve_id, quantity, float(r)), None)
                    if a is None:
                        continue
                    xs.append(float(r))
                    ys.append(float(a["mean"]))
                    es.append(float(a["std"]))

                if not xs:
                    logger.warning("No radial data for curve=%s quantity=%s", curve_id, quantity)
                    continue

                ax.errorbar(
                    xs,
                    ys,
                    yerr=es if bool(style.get("show_std", plot_cfg.get("show_std", True))) else None,
                    label=label,
                    marker=style.get("marker", plot_cfg.get("marker", "o")),
                    linestyle=style.get("linestyle", plot_cfg.get("linestyle", "-")),
                    linewidth=float(style.get("linewidth", plot_cfg.get("linewidth", 1.5))),
                    capsize=float(style.get("capsize", plot_cfg.get("capsize", 2.0))),
                    color=style.get("color", None),
                    alpha=float(style.get("alpha", plot_cfg.get("alpha", 1.0))),
                )

            ax.set_title(str(qcfg.get("title", spec.display_name)))
            ax.set_xlabel(str(plot_cfg.get("xlabel", r"Radius [$R_\odot$]")))
            ylabel = qcfg.get("ylabel", plot_cfg.get("ylabel", rf"MAE({spec.log_label})"))
            ax.set_ylabel(str(ylabel))
            ax.grid(True, alpha=float(plot_cfg.get("grid_alpha", 0.25)))

            if plot_cfg.get("yscale", None) is not None:
                ax.set_yscale(str(plot_cfg["yscale"]))
            if qcfg.get("yscale", None) is not None:
                ax.set_yscale(str(qcfg["yscale"]))
            if plot_cfg.get("xlim", None) is not None:
                ax.set_xlim(*plot_cfg["xlim"])
            if qcfg.get("ylim", None) is not None:
                ax.set_ylim(*qcfg["ylim"])

            if tick_labels is not None:
                ax.set_xticks([float(x) for x in radii])
                ax.set_xticklabels(
                    tick_labels,
                    rotation=float(plot_cfg.get("xtick_rotation", 0.0)),
                    ha=str(plot_cfg.get("xtick_ha", "center")),
                )

        if bool(plot_cfg.get("legend", True)):
            handles, labels = axs[0, 0].get_legend_handles_labels()
            loc = plot_cfg.get("legend_loc", "best")
            if str(loc) == "figure_bottom":
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, float(plot_cfg.get("legend_y", -0.04))),
                    ncol=int(plot_cfg.get("legend_ncol", len(labels))),
                    frameon=bool(plot_cfg.get("legend_frame", False)),
                )
            else:
                axs[0, 0].legend(
                    loc=loc,
                    frameon=bool(plot_cfg.get("legend_frame", False)),
                    ncol=int(plot_cfg.get("legend_ncol", 1)),
                )

        if plot_cfg.get("title", None):
            fig.suptitle(str(plot_cfg["title"]))

        out_paths = []
        for ext in formats:
            ext = str(ext).lstrip(".")
            path = out_base.with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))

        plt.close(fig)

    return {"out_paths": out_paths}


def generate_radial_metric_curves(
    spec_path: Union[str, Path],
    device_override: str | None = None,
    output_dir_override: Union[str, Path, None] = None,
) -> dict:
    spec_path = Path(spec_path).resolve()
    spec_root = spec_path.parent
    raw = load_yaml(spec_path)
    cfg = raw.get("figure", raw)

    name = str(cfg.get("name", spec_path.stem))
    benchmark_dir = _resolve_path(cfg["benchmark_dir"], spec_root)

    if output_dir_override is not None:
        out_dir = _resolve_path(output_dir_override, Path.cwd())
    else:
        out_dir = _resolve_path(cfg.get("output_dir", benchmark_dir / "paper_figures"), spec_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_remap = cfg.get("path_remap", None)
    device = device_override if device_override is not None else cfg.get("device", None)

    radii, tick_labels = _resolve_radii(cfg.get("radii", None))
    metric = str(cfg.get("metric", "mean_abs_dlog"))

    quantity_cfgs = list(cfg.get("quantities", []) or [{"quantity": "ne"}, {"quantity": "temp"}])
    quantities = list(dict.fromkeys(str(q["quantity"]) for q in quantity_cfgs))

    experiment_cfgs = list(cfg.get("experiments", []) or [])
    if not experiment_cfgs:
        raise ValueError("figure.experiments must be non-empty")

    default_metric = str(cfg.get("selection_metric", "final/temp_shell_inner/shell_mean_abs_dlog_mean"))
    default_policy = str(cfg.get("selection_policy", "all"))
    default_mode = str(cfg.get("select_mode", "min"))

    rows = collect_run_summaries(benchmark_dir)
    rows = [r for r in rows if r.get("status") == "completed"]
    if not rows:
        raise ValueError(f"No completed run summaries found in {benchmark_dir}")

    raw_records: list[dict] = []
    selected_runs: list[dict] = []

    for exp_idx, exp_cfg in enumerate(experiment_cfgs):
        curve_id = str(exp_cfg.get("id", exp_cfg.get("experiment", exp_cfg.get("name", exp_idx))))
        curve_label = str(exp_cfg.get("label", exp_cfg.get("experiment", exp_cfg.get("name", curve_id))))
        selected_rows = _select_rows_for_experiment(
            rows=rows,
            exp_cfg=dict(exp_cfg),
            default_metric=default_metric,
            default_policy=default_policy,
            default_mode=default_mode,
        )

        for row in selected_rows:
            recs, run_dir = _evaluate_run_radial(
                row=row,
                benchmark_dir=benchmark_dir,
                quantities=quantities,
                radii=radii,
                metric=metric,
                device=device,
                path_remap=path_remap,
            )
            selected_runs.append({
                "curve_id": curve_id,
                "curve_label": curve_label,
                "experiment": row.get("experiment_name"),
                "run_name": row.get("run_name"),
                "seed": row.get("seed"),
                "run_dir": str(run_dir),
            })
            for rec in recs:
                rec["curve_id"] = curve_id
                rec["curve_label"] = curve_label
                raw_records.append(rec)

    aggregates = _aggregate_records(raw_records)

    raw_csv = out_dir / f"{name}_per_run.csv"
    agg_csv = out_dir / f"{name}_grouped.csv"
    _write_csv(raw_csv, raw_records)
    _write_csv(agg_csv, aggregates)

    plot_info = _plot_radial_curves(
        aggregates=aggregates,
        quantity_cfgs=quantity_cfgs,
        experiment_cfgs=experiment_cfgs,
        out_base=out_dir / name,
        plot_cfg=dict(cfg.get("plot", {}) or {}),
        radii=radii,
        tick_labels=tick_labels,
    )

    manifest = {
        "name": name,
        "spec_path": str(spec_path),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(out_dir),
        "device": device,
        "path_remap": path_remap,
        "metric": metric,
        "radii": [float(r) for r in radii],
        "selected_runs": selected_runs,
        "per_run_csv": str(raw_csv),
        "grouped_csv": str(agg_csv),
        **plot_info,
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("Saved radial metric curve manifest: %s", out_dir / f"{name}_manifest.json")
    return manifest
