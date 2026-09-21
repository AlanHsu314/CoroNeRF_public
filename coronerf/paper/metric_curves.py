from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark.config import load_yaml, save_json
from ._aggregate_utils import load_grouped_table, lookup_one, metric_mean_std, resolve_metric_column
from .shell_panels import _resolve_path

logger = logging.getLogger("coroNeRF.paper.metric_curves")


'''
generic scripts for generating metric curves over some sweep parameter
for example, in noise ablation, over noise multiplier
'''

def _x_value(label, mapping: dict | None):
    if mapping is None:
        return label
    return mapping[label]

def _panel_metric_entries(panel_cfg: dict) -> list[dict]:
    """
    Backward-compatible metric spec parser.

    Old style:
      - metric: final/foo
        label: Foo

    New style:
      - title: "$n_e$"
        metrics:
          - metric: final/foo_abs
            label: MAE
          - metric: final/foo_signed
            label: ME
    """
    if "metrics" in panel_cfg:
        entries = []
        for item in list(panel_cfg.get("metrics", []) or []):
            entry = dict(item)
            if "metric" not in entry:
                raise ValueError(f"Metric entry is missing 'metric': {entry}")
            entries.append(entry)
        if not entries:
            raise ValueError(f"Panel has empty metrics list: {panel_cfg}")
        return entries

    if "metric" not in panel_cfg:
        raise ValueError(
            "Each panel must define either 'metric' or 'metrics'. "
            f"Got panel={panel_cfg}"
        )

    return [{
        "metric": panel_cfg["metric"],
        "label": panel_cfg.get("label", panel_cfg.get("title", panel_cfg["metric"])),
        "style": panel_cfg.get("style", {}),
    }]

def _curve_label(metric_cfg: dict, series_cfg: dict, n_metrics: int, n_series: int) -> str | None:
    metric_label = str(metric_cfg.get("label", metric_cfg.get("metric", "metric")))
    series_label = str(series_cfg.get("label", ""))

    if n_metrics > 1 and n_series > 1:
        return f"{series_label} {metric_label}".strip()
    if n_metrics > 1:
        return metric_label
    if n_series > 1:
        return series_label
    return metric_label if bool(metric_cfg.get("show_label", False)) else None

def _style_value(metric_cfg: dict, series_cfg: dict, plot_cfg: dict, key: str, default=None):
    metric_style = dict(metric_cfg.get("style", {}) or {})
    series_style = dict(series_cfg.get("style", {}) or {})

    if key in metric_cfg:
        return metric_cfg[key]
    if key in metric_style:
        return metric_style[key]
    if key in series_cfg:
        return series_cfg[key]
    if key in series_style:
        return series_style[key]
    return plot_cfg.get(key, default)

def _x_tick_labels(x_values: list, x_cfg: dict) -> list[str]:
    tick_cfg = x_cfg.get("tick_labels", None)
    if tick_cfg is None:
        return [str(x) for x in x_values]

    if isinstance(tick_cfg, dict):
        out = []
        for x in x_values:
            out.append(str(tick_cfg.get(x, tick_cfg.get(str(x), x))))
        return out

    out = [str(x) for x in tick_cfg]
    if len(out) != len(x_values):
        raise ValueError(
            f"x.tick_labels has length {len(out)}, expected {len(x_values)}"
        )
    return out

def _apply_x_ticks_and_guides(ax, x_values: list, x_mapping: dict | None, x_cfg: dict, plot_cfg: dict) -> None:
    if x_mapping is not None:
        tick_positions = [_x_value(x, x_mapping) for x in x_values]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            _x_tick_labels(x_values, x_cfg),
            rotation=float(plot_cfg.get("xtick_rotation", 0.0)),
            ha=str(plot_cfg.get("xtick_ha", "center")),
        )

    for guide in list(plot_cfg.get("vertical_lines", []) or []):
        ax.axvline(
            float(guide["x"]),
            color=guide.get("color", "black"),
            linewidth=float(guide.get("linewidth", 0.8)),
            linestyle=guide.get("linestyle", ":"),
            alpha=float(guide.get("alpha", 0.8)),
        )

def generate_metric_curves(
    spec_path: Union[str, Path],
    output_dir_override: Union[str, Path, None] = None,
) -> dict:
    spec_path = Path(spec_path).resolve()
    spec_root = spec_path.parent

    raw = load_yaml(spec_path)
    cfg = raw.get("figure", raw)

    name = str(cfg.get("name", spec_path.stem))
    grouped_csv = _resolve_path(cfg["grouped_csv"], spec_root)
    df = load_grouped_table(grouped_csv)

    if output_dir_override is not None:
        out_dir = _resolve_path(output_dir_override, Path.cwd())
    else:
        out_dir = _resolve_path(cfg.get("output_dir", grouped_csv.parent / "paper_figures"), spec_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_filter = dict(cfg.get("filter", {}) or {})
    x_cfg = dict(cfg["x"])
    panels = list(cfg.get("panels", []) or [])
    series_cfgs = list(cfg.get("series", []) or [{"label": "", "where": {}}])

    x_key = str(x_cfg["key"])
    x_values = list(x_cfg["values"])
    x_mapping = x_cfg.get("mapping", None)

    plot_cfg = dict(cfg.get("plot", {}) or {})
    figsize = plot_cfg.get("figsize", [8.0, 2.7])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])

    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))

    rows_for_csv = []

    rc = {
        "font.size": font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_font_size,
        "ytick.labelsize": tick_font_size,
        "legend.fontsize": font_size,
    }

    with plt.rc_context(rc):
        fig, axs = plt.subplots(
            1,
            len(panels),
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )
        axs = axs[0]

        for ax, panel_cfg in zip(axs, panels):
            metric_entries = _panel_metric_entries(panel_cfg)

            # Validate metrics early.
            for metric_cfg in metric_entries:
                resolve_metric_column(df, str(metric_cfg["metric"]), stat="mean")

            last_metric = str(metric_entries[-1]["metric"])

            for metric_cfg in metric_entries:
                metric = str(metric_cfg["metric"])

                for series_cfg in series_cfgs:
                    series_label = str(series_cfg.get("label", ""))
                    where_series = dict(series_cfg.get("where", {}) or {})

                    xs = []
                    ys = []
                    es = []

                    for xv in x_values:
                        where = dict(base_filter)
                        where.update(where_series)
                        where[x_key] = xv

                        row = lookup_one(df, where)
                        mean, std = metric_mean_std(row, metric)

                        x_plot = _x_value(xv, x_mapping)
                        xs.append(x_plot)
                        ys.append(mean)
                        es.append(std)

                        rows_for_csv.append({
                            "panel": panel_cfg.get("label", panel_cfg.get("title", metric)),
                            "metric": metric,
                            "curve": metric_cfg.get("label", metric),
                            "series": series_label,
                            x_key: xv,
                            "x_plot": x_plot,
                            "mean": mean,
                            "std": std,
                        })

                    label = _curve_label(
                        metric_cfg,
                        series_cfg,
                        n_metrics=len(metric_entries),
                        n_series=len(series_cfgs),
                    )

                    ax.errorbar(
                        xs,
                        ys,
                        yerr=es if bool(_style_value(metric_cfg, series_cfg, plot_cfg, "show_std", True)) else None,
                        label=label,
                        marker=_style_value(metric_cfg, series_cfg, plot_cfg, "marker", "o"),
                        linestyle=_style_value(metric_cfg, series_cfg, plot_cfg, "linestyle", "-"),
                        linewidth=float(_style_value(metric_cfg, series_cfg, plot_cfg, "linewidth", 1.5)),
                        capsize=float(_style_value(metric_cfg, series_cfg, plot_cfg, "capsize", 2.0)),
                        color=_style_value(metric_cfg, series_cfg, plot_cfg, "color", None),
                    )

                    for ref_cfg in list(panel_cfg.get("reference_points", []) or []):
                        ref_metric = str(ref_cfg.get("metric", last_metric))
                        where = dict(base_filter)
                        where.update(dict(ref_cfg.get("where", {}) or {}))
                        row = lookup_one(df, where)
                        mean, std = metric_mean_std(row, ref_metric)

                        x_plot = ref_cfg.get("x", None)
                        if x_plot is None:
                            x_raw = ref_cfg.get(x_key, where.get(x_key, None))
                            if x_raw is None:
                                raise ValueError(f"Reference point needs either 'x' or {x_key!r}: {ref_cfg}")
                            x_plot = _x_value(x_raw, x_mapping)

                        ax.errorbar(
                            [x_plot],
                            [mean],
                            yerr=[std] if bool(ref_cfg.get("show_std", plot_cfg.get("show_std", True))) else None,
                            label=ref_cfg.get("label", None),
                            marker=ref_cfg.get("marker", "D"),
                            linestyle=ref_cfg.get("linestyle", "none"),
                            linewidth=float(ref_cfg.get("linewidth", plot_cfg.get("linewidth", 1.5))),
                            capsize=float(ref_cfg.get("capsize", plot_cfg.get("capsize", 2.0))),
                            color=ref_cfg.get("color", None),
                        )

                        rows_for_csv.append({
                            "panel": panel_cfg.get("label", panel_cfg.get("title", ref_metric)),
                            "metric": ref_metric,
                            "curve": ref_cfg.get("label", "reference"),
                            "series": "reference",
                            x_key: where.get(x_key, ""),
                            "x_plot": x_plot,
                            "mean": mean,
                            "std": std,
                        })


            ax.set_xlabel(x_cfg.get("label", x_key))
            ax.set_ylabel(panel_cfg.get("ylabel", panel_cfg.get("label", last_metric)))
            ax.set_title(panel_cfg.get("title", panel_cfg.get("label", last_metric)))

            if plot_cfg.get("xscale", None):
                ax.set_xscale(str(plot_cfg["xscale"]))
            if panel_cfg.get("yscale", plot_cfg.get("yscale", None)):
                ax.set_yscale(str(panel_cfg.get("yscale", plot_cfg.get("yscale"))))

            if panel_cfg.get("ylim", None) is not None:
                ax.set_ylim(*panel_cfg["ylim"])
            if plot_cfg.get("xlim", None) is not None:
                ax.set_xlim(*plot_cfg["xlim"])

            _apply_x_ticks_and_guides(ax, x_values, x_mapping, x_cfg, plot_cfg)

            ax.grid(True, alpha=float(plot_cfg.get("grid_alpha", 0.25)))

            if bool(panel_cfg.get("zero_line", plot_cfg.get("zero_line", False))):
                ax.axhline(
                    0.0,
                    color=plot_cfg.get("zero_line_color", "black"),
                    linewidth=float(plot_cfg.get("zero_line_width", 0.8)),
                    linestyle=plot_cfg.get("zero_line_style", ":"),
                    alpha=float(plot_cfg.get("zero_line_alpha", 0.8)),
                )

        if bool(plot_cfg.get("legend", True)):
            handles, labels = axs[-1].get_legend_handles_labels()
            if handles:
                axs[-1].legend(
                    loc=plot_cfg.get("legend_loc", "best"),
                    frameon=bool(plot_cfg.get("legend_frame", False)),
                    ncol=int(plot_cfg.get("legend_ncol", 1)),
                )

        if plot_cfg.get("title", None):
            fig.suptitle(str(plot_cfg["title"]))

        out_paths = []
        for ext in formats:
            ext = str(ext).lstrip(".")
            path = (out_dir / name).with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))

        plt.close(fig)

    csv_path = out_dir / f"{name}_data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["panel", "metric", "curve", "series", x_key, "x_plot", "mean", "std"],
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    manifest = {
        "name": name,
        "spec_path": str(spec_path),
        "grouped_csv": str(grouped_csv),
        "output_dir": str(out_dir),
        "out_paths": out_paths,
        "csv": str(csv_path),
        "x": x_cfg,
        "panels": panels,
        "series": series_cfgs,
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    return manifest