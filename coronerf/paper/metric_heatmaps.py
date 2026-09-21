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
from ._aggregate_utils import load_grouped_table, lookup_one, resolve_metric_column
from .shell_panels import _resolve_path

logger = logging.getLogger("coroNeRF.paper.metric_heatmaps")

'''
creating field heatmaps over 2 different generalized sweep axes
for example, in a noise ablation, over view count and noise multiplier
'''


def _format_cell(v: float, fmt_small: str = "{:.3f}", fmt_large: str = "{:.2f}") -> str:
    if not np.isfinite(v):
        return "—"
    if abs(v) < 0.1:
        return fmt_small.format(v)
    return fmt_large.format(v)


def generate_metric_heatmaps(
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

    x_cfg = dict(cfg["x"])
    y_cfg = dict(cfg["y"])
    panels = list(cfg.get("panels", []) or [])
    if not panels:
        raise ValueError("figure.panels must be non-empty")

    x_key = str(x_cfg["key"])
    y_key = str(y_cfg["key"])
    x_values = list(x_cfg["values"])
    y_values = list(y_cfg["values"])

    x_labels = list(x_cfg.get("labels", x_values))
    y_labels = list(y_cfg.get("labels", y_values))

    base_filter = dict(cfg.get("filter", {}) or {})

    rows_for_csv = []

    plot_cfg = dict(cfg.get("plot", {}) or {})
    figsize = plot_cfg.get("figsize", [7.2, 2.9])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])
    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))
    cmap_default = plot_cfg.get("cmap", "magma_r")

    n_panels = len(panels)

    rc = {
        "font.size": font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_font_size,
        "ytick.labelsize": tick_font_size,
    }

    plot_manifest = {
        "panels": [],
    }

    with plt.rc_context(rc):
        fig, axs = plt.subplots(
            1,
            n_panels,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )
        axs = axs[0]

        for ax, panel_cfg in zip(axs, panels):
            metric = str(panel_cfg["metric"])
            metric_col = resolve_metric_column(df, metric, stat=str(panel_cfg.get("stat", "mean")))

            Z = np.full((len(y_values), len(x_values)), np.nan, dtype=float)

            for iy, yval in enumerate(y_values):
                for ix, xval in enumerate(x_values):
                    where = dict(base_filter)
                    where[x_key] = yval if y_key == x_key else xval
                    where[y_key] = yval

                    # Correct x assignment when keys are distinct.
                    where[x_key] = xval

                    row = lookup_one(df, where)
                    val = float(row[metric_col])
                    Z[iy, ix] = val

                    rows_for_csv.append({
                        "panel": panel_cfg.get("label", metric),
                        "metric": metric_col,
                        x_key: xval,
                        y_key: yval,
                        "value": val,
                    })

            vmax = panel_cfg.get("vmax", plot_cfg.get("vmax", None))
            vmin = panel_cfg.get("vmin", plot_cfg.get("vmin", None))
            if vmax is None:
                vmax = float(np.nanpercentile(Z, 99.0))
            if vmin is None:
                vmin = float(np.nanmin(Z))

            Z_plot = np.array(Z, copy=True)
            if bool(panel_cfg.get("clip", plot_cfg.get("clip", True))):
                Z_plot = np.clip(Z_plot, float(vmin), float(vmax))

            im = ax.imshow(
                Z_plot,
                origin="lower",
                aspect=plot_cfg.get("aspect", "auto"),
                cmap=panel_cfg.get("cmap", cmap_default),
                vmin=float(vmin),
                vmax=float(vmax),
            )

            ax.set_xticks(np.arange(len(x_values)))
            ax.set_xticklabels([str(x) for x in x_labels])
            ax.set_yticks(np.arange(len(y_values)))
            ax.set_yticklabels([str(y) for y in y_labels])
            ax.set_xlabel(x_cfg.get("label", x_key))
            ax.set_ylabel(y_cfg.get("label", y_key))
            ax.set_title(panel_cfg.get("title", panel_cfg.get("label", metric)))

            if bool(plot_cfg.get("annotate", True)):
                auto_contrast = bool(plot_cfg.get("annot_auto_contrast", False))
                fixed_color = plot_cfg.get("annot_color", "white")
                dark_c  = plot_cfg.get("annot_color_dark", "black")    # text on light cells
                light_c = plot_cfg.get("annot_color_light", "white")   # text on dark cells
                lum_thresh = float(plot_cfg.get("annot_lum_threshold", 0.6))
                for iy in range(len(y_values)):
                    for ix in range(len(x_values)):
                        txt = _format_cell(
                            Z[iy, ix],
                            fmt_small=plot_cfg.get("annot_fmt_small", "{:.3f}"),
                            fmt_large=plot_cfg.get("annot_fmt_large", "{:.2f}"),
                        )
                        if auto_contrast and np.isfinite(Z_plot[iy, ix]):
                            r, g, b, _ = im.cmap(im.norm(Z_plot[iy, ix]))     # this cell's displayed colour
                            lum = 0.299 * r + 0.587 * g + 0.114 * b            # perceived luminance in [0,1]
                            cell_color = dark_c if lum > lum_thresh else light_c
                        else:
                            cell_color = fixed_color
                        ax.text(
                            ix,
                            iy,
                            txt,
                            ha="center",
                            va="center",
                            color=cell_color,
                            fontsize=float(plot_cfg.get("annot_font_size", tick_font_size)),
                        )

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=float(plot_cfg.get("colorbar_fraction", 0.046)),
                pad=float(plot_cfg.get("colorbar_pad", 0.04)),
            )
            cbar.set_label(panel_cfg.get("colorbar_label", panel_cfg.get("label", metric)))

            plot_manifest["panels"].append({
                "metric": metric,
                "metric_col": metric_col,
                "vmin": float(vmin),
                "vmax": float(vmax),
            })

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
        writer = csv.DictWriter(f, fieldnames=["panel", "metric", x_key, y_key, "value"])
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
        "y": y_cfg,
        **plot_manifest,
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    return manifest