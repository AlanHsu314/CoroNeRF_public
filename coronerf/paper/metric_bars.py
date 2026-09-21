from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from .shell_panels import _resolve_path

logger = logging.getLogger("coroNeRF.paper.metric_bars")


def _completed_rows_for_experiment(rows: list[dict], experiment: str) -> list[dict]:
    return [
        r for r in rows
        if r.get("status") == "completed" and str(r.get("experiment_name")) == str(experiment)
    ]


def _mean_std(vals: list[float]) -> tuple[float, float, int]:
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan"), 0

    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def _metric_label(metric_cfg: dict) -> str:
    return str(metric_cfg.get("label", metric_cfg.get("key", "metric")))


def generate_metric_bar_figure(
    spec_path: Union[str, Path],
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

    rows = collect_run_summaries(benchmark_dir)
    rows = [r for r in rows if r.get("status") == "completed"]
    if not rows:
        raise ValueError(f"No completed run summaries found in {benchmark_dir}")

    experiments = list(cfg.get("experiments", []) or [])
    metrics = list(cfg.get("metrics", []) or [])
    if not experiments:
        raise ValueError("figure.experiments must be non-empty")
    if not metrics:
        raise ValueError("figure.metrics must be non-empty")

    table_rows = []
    means = np.zeros((len(experiments), len(metrics)), dtype=float)
    stds = np.zeros_like(means)
    counts = np.zeros_like(means, dtype=int)

    for i, exp_cfg in enumerate(experiments):
        if isinstance(exp_cfg, str):
            exp_cfg = {"name": exp_cfg, "label": exp_cfg}

        exp_name = str(exp_cfg["name"])
        exp_label = str(exp_cfg.get("label", exp_name))
        exp_rows = _completed_rows_for_experiment(rows, exp_name)

        for j, metric_cfg in enumerate(metrics):
            key = str(metric_cfg["key"])
            vals = [r.get(key, None) for r in exp_rows]
            mean, std, n = _mean_std(vals)

            means[i, j] = mean
            stds[i, j] = std
            counts[i, j] = n

            table_rows.append({
                "experiment": exp_name,
                "label": exp_label,
                "metric": key,
                "metric_label": _metric_label(metric_cfg),
                "mean": mean,
                "std": std,
                "n": n,
            })

    csv_path = out_dir / f"{name}_data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "label", "metric", "metric_label", "mean", "std", "n"],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    plot_cfg = dict(cfg.get("plot", {}) or {})
    figsize = plot_cfg.get("figsize", [6.2, 3.0])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])
    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))

    exp_labels = [
        str(e.get("label", e["name"]) if isinstance(e, dict) else e)
        for e in experiments
    ]

    metric_labels = [_metric_label(m) for m in metrics]

    x = np.arange(len(experiments), dtype=float)
    n_metrics = len(metrics)
    total_width = float(plot_cfg.get("total_bar_width", 0.78))
    bar_width = total_width / max(n_metrics, 1)

    rc = {
        "font.size": font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_font_size,
        "ytick.labelsize": tick_font_size,
        "legend.fontsize": font_size,
    }

    out_paths = []
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        for j, metric_cfg in enumerate(metrics):
            offset = (j - (n_metrics - 1) / 2.0) * bar_width
            kwargs = {}
            if metric_cfg.get("color", None) is not None:
                kwargs["color"] = metric_cfg["color"]
            if metric_cfg.get("hatch", None) is not None:
                kwargs["hatch"] = metric_cfg["hatch"]

            ax.bar(
                x + offset,
                means[:, j],
                width=bar_width * 0.92,
                yerr=stds[:, j],
                capsize=float(plot_cfg.get("capsize", 2.0)),
                label=metric_labels[j],
                **kwargs,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, rotation=float(plot_cfg.get("xtick_rotation", 0)))
        ax.set_ylabel(plot_cfg.get("ylabel", "Metric"))
        if plot_cfg.get("title", None):
            ax.set_title(str(plot_cfg["title"]))

        if plot_cfg.get("yscale", None):
            ax.set_yscale(str(plot_cfg["yscale"]))

        if plot_cfg.get("ylim", None) is not None:
            ax.set_ylim(*plot_cfg["ylim"])

        ax.grid(True, axis="y", alpha=float(plot_cfg.get("grid_alpha", 0.25)))
        ax.legend(frameon=bool(plot_cfg.get("legend_frame", False)))

        for ext in formats:
            ext = str(ext).lstrip(".")
            path = (out_dir / name).with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))

        plt.close(fig)

    manifest = {
        "name": name,
        "spec_path": str(spec_path),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(out_dir),
        "csv": str(csv_path),
        "out_paths": out_paths,
        "experiments": experiments,
        "metrics": metrics,
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("Saved metric bar figure manifest: %s", out_dir / f"{name}_manifest.json")
    return manifest