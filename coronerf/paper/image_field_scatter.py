from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from .shell_panels import _resolve_path

logger = logging.getLogger("coroNeRF.paper.image_field_scatter")


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v

def _experiment_entries(bench_cfg: dict) -> list[dict]:
    entries = bench_cfg.get("experiments", None)
    include = bench_cfg.get("include_experiments", None)

    if entries is None and include is not None:
        entries = include

    if entries is None:
        return [{"name": "*", "label": "*"}]

    out = []
    for e in entries:
        if isinstance(e, str):
            out.append({"name": e, "label": e})
        else:
            out.append(dict(e))
    return out

def _row_matches_experiment(row: dict, exp_name: str) -> bool:
    if exp_name == "*":
        return True
    return str(row.get("experiment_name")) == str(exp_name)

def _coerce_for_compare(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return str(x)

def _values_match(a, b, atol: float = 1.0e-9) -> bool:
    aa = _coerce_for_compare(a)
    bb = _coerce_for_compare(b)

    if aa is None or bb is None:
        return aa is bb

    if isinstance(aa, float) and isinstance(bb, float):
        return abs(aa - bb) <= atol

    return str(aa) == str(bb)

def _row_matches_where(row: dict, where: dict | None) -> bool:
    """
    Match a summary row against optional filters.

    This supports both exact keys and sweep-prefixed keys, e.g.
      train_num_views
      sweep.load_data_kwargs.train_num_views
    depending on what exists in the summary rows.
    """
    if not where:
        return True

    for key, expected in dict(where).items():
        key = str(key)

        if key in row:
            actual = row.get(key)
        elif f"sweep.{key}" in row:
            actual = row.get(f"sweep.{key}")
        else:
            return False

        if not _values_match(actual, expected):
            return False

    return True

class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "?"

def _format_record_template(template: str, rec: dict) -> str:
    """
    Format labels using row fields.

    Example:
      condition_label: "Noise x25, {train_num_views} views"
    """
    return str(template).format_map(_SafeFormatDict(rec))

def _collect_records(cfg: dict, spec_root: Path) -> list[dict]:
    records = []

    for bench_cfg in list(cfg.get("benchmarks", []) or []):
        bench_id = str(bench_cfg.get("id", bench_cfg.get("name", "benchmark")))
        study = str(bench_cfg.get("study", bench_id))
        bench_dir = _resolve_path(bench_cfg["benchmark_dir"], spec_root)

        rows = collect_run_summaries(bench_dir)
        rows = [r for r in rows if r.get("status") == "completed"]

        for exp_cfg in _experiment_entries(bench_cfg):
            exp_name = str(exp_cfg.get("name", "*"))

            exp_rows = [r for r in rows if _row_matches_experiment(r, exp_name)]
            where = exp_cfg.get("where", None) or exp_cfg.get("filters", None)
            exp_rows = [r for r in exp_rows if _row_matches_where(r, where)]

            for row in exp_rows:
                rec = dict(row)
                rec["paper_benchmark_id"] = bench_id
                rec["paper_study"] = study
                rec["paper_benchmark_dir"] = str(bench_dir)
                rec["paper_experiment_name"] = str(row.get("experiment_name", exp_name))
                rec["paper_label"] = str(exp_cfg.get("label", row.get("experiment_name", exp_name)))

                if "condition_label" in exp_cfg:
                    rec["paper_condition"] = _format_record_template(exp_cfg["condition_label"], rec)
                elif "label_template" in exp_cfg:
                    rec["paper_condition"] = _format_record_template(exp_cfg["label_template"], rec)
                else:
                    rec["paper_condition"] = rec["paper_label"]

                # Optional style/group metadata.
                for key, val in exp_cfg.items():
                    if key not in {"name", "label"}:
                        rec[f"paper_{key}"] = val

                # Benchmark-level defaults can fill missing metadata.
                for key in ["group", "color_key", "marker_key"]:
                    if f"paper_{key}" not in rec and key in bench_cfg:
                        rec[f"paper_{key}"] = bench_cfg[key]

                records.append(rec)

    return records

def _build_value_map(records: list[dict], key: str, explicit: dict | None, fallback_values: list[Any]) -> dict:
    if explicit:
        return {str(k): v for k, v in explicit.items()}

    vals = []
    for r in records:
        v = r.get(key, None)
        if v is None:
            continue
        sv = str(v)
        if sv not in vals:
            vals.append(sv)

    out = {}
    for i, v in enumerate(vals):
        out[v] = fallback_values[i % len(fallback_values)]
    return out

def _write_records_csv(records: list[dict], path: Path, x_metric: str, panels: list[dict]) -> None:
    fields = [
        "paper_benchmark_id",
        "paper_study",
        "paper_experiment_name",
        "paper_label",
        "paper_condition",
        "paper_group",
        "train_num_views",
        "seed",
        "run_name",
        x_metric,
    ]
    for p in panels:
        fields.append(str(p["y_metric"]))

    # Keep unique order.
    seen = set()
    fields = [f for f in fields if not (f in seen or seen.add(f))]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

def _style_key(rec, key):
    val = rec.get(key, None)
    if val is None:
        return "unknown"
    try:
        if isinstance(val, float) and np.isnan(val):
            return "unknown"
    except Exception:
        pass
    return str(val)

def _legend_display_label(legend_key: str, label_map: dict | None) -> str:
    """Map an internal legend key to a paper-readable display label."""
    if not label_map:
        return str(legend_key)

    label_map = {str(k): str(v) for k, v in dict(label_map).items()}
    return label_map.get(str(legend_key), str(legend_key))

def _scatter_into_ax(
    ax, records, x_metric, y_metric, *,
    color_by, marker_by, legend_by,
    color_map, marker_map, legend_label_map,
    style_cfg, plot_cfg, used_labels,
    xlabel=None, ylabel=None, title=None, yscale=None, ylim=None,
    xscale=None, xlim=None,
):
    """Draw one image-vs-field scatter panel into a given axes. Extracted
    verbatim from generate_image_field_scatter's per-panel loop; `used_labels`
    is shared across panels for legend de-duplication (unchanged behavior)."""
    for rec in records:
        x = _safe_float(rec.get(x_metric, None))
        y = _safe_float(rec.get(y_metric, None))
        if x is None or y is None:
            continue

        color_key = _style_key(rec, color_by)
        marker_key = _style_key(rec, marker_by)
        legend_key = _style_key(rec, legend_by)
        if legend_key == "unknown":
            legend_key = str(rec.get("paper_label", "unknown"))

        if color_key not in color_map:
            logger.warning("No color mapping for %s=%r; using C0", color_by, color_key)
        if marker_key not in marker_map:
            logger.warning("No marker mapping for %s=%r; using o", marker_by, marker_key)

        label = (
            _legend_display_label(legend_key, legend_label_map)
            if legend_key not in used_labels else None
        )
        used_labels.add(legend_key)

        ax.scatter(
            x, y,
            s=float(style_cfg.get("size", 34)),
            alpha=float(style_cfg.get("alpha", 0.85)),
            linewidth=float(style_cfg.get("linewidth", 0.6)),
            edgecolor=style_cfg.get("edgecolor", "black"),
            color=color_map.get(color_key, "C0"),
            marker=marker_map.get(marker_key, "o"),
            label=label,
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(str(title))
    _xscale = xscale if xscale is not None else plot_cfg.get("xscale", None)
    if _xscale:
        ax.set_xscale(str(_xscale))
    if yscale:
        ax.set_yscale(str(yscale))
    if ylim is not None:
        ax.set_ylim(*ylim)
    _xlim = xlim if xlim is not None else plot_cfg.get("xlim", None)
    if _xlim is not None:
        ax.set_xlim(*_xlim)
    ax.grid(True, alpha=float(plot_cfg.get("grid_alpha", 0.25)))

# my old density + temperature side-by side scatter

def generate_image_field_scatter(
    spec_path: Union[str, Path],
    output_dir_override: Union[str, Path, None] = None,
) -> dict:
    spec_path = Path(spec_path).resolve()
    spec_root = spec_path.parent

    raw = load_yaml(spec_path)
    cfg = raw.get("figure", raw)

    name = str(cfg.get("name", spec_path.stem))

    if output_dir_override is not None:
        out_dir = _resolve_path(output_dir_override, Path.cwd())
    else:
        out_dir = _resolve_path(cfg.get("output_dir", "../../paper_outputs/figures/scatter"), spec_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_metric = str(cfg.get("x_metric", "final/val_loss"))
    panels = list(cfg.get("panels", []) or [])
    if not panels:
        panels = [
            {
                "y_metric": "final/ne_shell_inner/shell_mean_abs_dlog_mean",
                "ylabel": "$n_e$ inner mean abs. dlog",
                "title": "$n_e$",
            },
            {
                "y_metric": "final/temp_shell_inner/shell_mean_abs_dlog_mean",
                "ylabel": "$T$ inner mean abs. dlog",
                "title": "$T$",
            },
        ]

    records = _collect_records(cfg, spec_root)
    logger.info("Collected %d records for image-field scatter", len(records))
    if not records:
        raise ValueError("No completed run records found for scatter figure.")

    csv_path = out_dir / f"{name}_data.csv"
    _write_records_csv(records, csv_path, x_metric=x_metric, panels=panels)

    plot_cfg = dict(cfg.get("plot", {}) or {})
    style_cfg = dict(cfg.get("style", {}) or {})

    figsize = plot_cfg.get("figsize", [7.2, 3.0])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])

    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))

    color_by = str(style_cfg.get("color_by", "paper_label"))
    marker_by = str(style_cfg.get("marker_by", "paper_label"))
    legend_by = str(style_cfg.get("legend_by", "paper_label"))
    legend_label_map = style_cfg.get("legend_labels", None)

    color_map = _build_value_map(
        records,
        color_by,
        style_cfg.get("colors", None),
        fallback_values=[
            "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
        ],
    )

    marker_map = _build_value_map(
        records,
        marker_by,
        style_cfg.get("markers", None),
        fallback_values=["o", "s", "^", "D", "P", "X", "v", "<", ">"],
    )

    n_panels = len(panels)

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
        fig, axs = plt.subplots(
            1,
            n_panels,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )
        axs = axs[0]

        used_labels = set()

        for ax, panel in zip(axs, panels):
            _scatter_into_ax(
                ax, records,
                x_metric=x_metric,
                y_metric=str(panel["y_metric"]),
                color_by=color_by, marker_by=marker_by, legend_by=legend_by,
                color_map=color_map, marker_map=marker_map,
                legend_label_map=legend_label_map,
                style_cfg=style_cfg, plot_cfg=plot_cfg, used_labels=used_labels,
                xlabel=plot_cfg.get("xlabel", x_metric),
                ylabel=panel.get("ylabel", str(panel["y_metric"])),
                title=panel.get("title", None),
                yscale=panel.get("yscale", plot_cfg.get("yscale", None)),
                ylim=panel.get("ylim", None),
            )

        # Legend placement.
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            legend_mode = str(plot_cfg.get("legend_mode", "figure")).lower()

            if legend_mode in {"figure", "outside"}:
                fig.legend(
                    handles,
                    labels,
                    loc=plot_cfg.get("legend_loc", "center right"),
                    bbox_to_anchor=tuple(plot_cfg.get("legend_bbox_to_anchor", [1.02, 0.5])),
                    frameon=bool(plot_cfg.get("legend_frame", False)),
                    ncol = int(plot_cfg.get('legend_ncol', 4)),
                )

            elif legend_mode in {"axis", "inside", "subplot"}:
                legend_axis = int(plot_cfg.get("legend_axis", 0))
                if legend_axis < 0:
                    legend_axis = len(axs) + legend_axis
                if legend_axis < 0 or legend_axis >= len(axs):
                    raise IndexError(
                        f"plot.legend_axis={plot_cfg.get('legend_axis')} is out of range "
                        f"for {len(axs)} subplot(s)"
                    )

                axs[legend_axis].legend(
                    handles,
                    labels,
                    loc=plot_cfg.get("legend_loc", "upper right"),
                    frameon=bool(plot_cfg.get("legend_frame", True)),
                    fontsize=plot_cfg.get("legend_font_size", None),
                    title=plot_cfg.get("legend_title", None),
                )

            elif legend_mode in {"none", "off", "false"}:
                pass

            else:
                raise ValueError(
                    f"Unknown plot.legend_mode={legend_mode!r}; "
                    "use one of: figure, outside, axis, inside, subplot, none"
                )

        title = plot_cfg.get("title", None)
        if title:
            fig.suptitle(str(title))

        for ext in formats:
            ext = str(ext).lstrip(".")
            path = (out_dir / name).with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))

        plt.close(fig)

    manifest = {
        "name": name,
        "spec_path": str(spec_path),
        "output_dir": str(out_dir),
        "csv": str(csv_path),
        "out_paths": out_paths,
        "x_metric": x_metric,
        "panels": panels,
        "num_records": len(records),
        "style": {
            "color_by": color_by,
            "marker_by": marker_by,
            "legend_by": legend_by,
            "color_map": color_map,
            "marker_map": marker_map,
            "legend_label_map": legend_label_map,
        },
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("Saved image-field scatter manifest: %s", out_dir / f"{name}_manifest.json")
    return manifest

#===============================================================================================================
# new image scatter grid: rows are field metrics, cols are image metrics, for the main paper
def _write_records_csv_grid(records: list[dict], path: Path, columns: list[dict], rows: list[dict]) -> None:
    fields = [
        "paper_benchmark_id", "paper_study", "paper_experiment_name",
        "paper_label", "paper_condition", "paper_group",
        "train_num_views", "seed", "run_name",
    ]
    for c in columns:
        fields.append(str(c["x_metric"]))
    for r in rows:
        fields.append(str(r["y_metric"]))
    seen = set()
    fields = [f for f in fields if not (f in seen or seen.add(f))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def generate_image_field_grid(
    spec_path: Union[str, Path],
    output_dir_override: Union[str, Path, None] = None,
) -> dict:
    """Generic image-vs-field grid: columns = image-space metrics (x), rows = field-space metrics (y).
    Every panel is the same scatter as generate_image_field_scatter; only the layout differs."""
    spec_path = Path(spec_path).resolve()
    spec_root = spec_path.parent

    raw = load_yaml(spec_path)
    cfg = raw.get("figure", raw)
    name = str(cfg.get("name", spec_path.stem))

    if output_dir_override is not None:
        out_dir = _resolve_path(output_dir_override, Path.cwd())
    else:
        out_dir = _resolve_path(cfg.get("output_dir", "../../paper_outputs/figures/scatter"), spec_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # columns = image-space metrics (x); rows = field-space metrics (y)
    columns = list(cfg.get("columns", []) or [])
    rows = list(cfg.get("rows", []) or [])
    if not columns:
        columns = [{"x_metric": "final/val_loss", "xlabel": "held-out image loss", "title": None}]
    if not rows:
        rows = [
            {"y_metric": "final/ne_shell_inner/shell_mean_abs_dlog_mean", "ylabel": r"$\mathrm{MAE}_{\mathrm{inner}}(\log_{10} n_e)$"},
            {"y_metric": "final/temp_shell_inner/shell_mean_abs_dlog_mean", "ylabel": r"$\mathrm{MAE}_{\mathrm{inner}}(\log_{10} T)$"},
        ]

    records = _collect_records(cfg, spec_root)
    logger.info("Collected %d records for image-field grid", len(records))
    if not records:
        raise ValueError("No completed run records found for grid figure.")

    csv_path = out_dir / f"{name}_data.csv"
    _write_records_csv_grid(records, csv_path, columns, rows)

    plot_cfg = dict(cfg.get("plot", {}) or {})
    style_cfg = dict(cfg.get("style", {}) or {})

    ncols, nrows = len(columns), len(rows)
    figsize = plot_cfg.get("figsize", [3.8 * ncols, 3.0 * nrows])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])

    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))

    color_by = str(style_cfg.get("color_by", "paper_label"))
    marker_by = str(style_cfg.get("marker_by", "paper_label"))
    legend_by = str(style_cfg.get("legend_by", "paper_label"))
    legend_label_map = style_cfg.get("legend_labels", None)

    color_map = _build_value_map(
        records, color_by, style_cfg.get("colors", None),
        fallback_values=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
    )
    marker_map = _build_value_map(
        records, marker_by, style_cfg.get("markers", None),
        fallback_values=["o", "s", "^", "D", "P", "X", "v", "<", ">"],
    )

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
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
        used_labels = set()

        for i, row in enumerate(rows):
            for j, col in enumerate(columns):
                _scatter_into_ax(
                    axs[i][j], records,
                    x_metric=str(col["x_metric"]),
                    y_metric=str(row["y_metric"]),
                    color_by=color_by, marker_by=marker_by, legend_by=legend_by,
                    color_map=color_map, marker_map=marker_map,
                    legend_label_map=legend_label_map,
                    style_cfg=style_cfg, plot_cfg=plot_cfg, used_labels=used_labels,
                    xlabel=(col.get("xlabel", str(col["x_metric"])) if i == nrows - 1 else None),
                    ylabel=(row.get("ylabel", str(row["y_metric"])) if j == 0 else None),
                    title=(col.get("title", None) if i == 0 else None),
                    yscale=row.get("yscale", plot_cfg.get("yscale", None)),
                    ylim=row.get("ylim", None),
                    xscale=col.get("xscale", None),
                    xlim=col.get("xlim", None),
                )

        # one legend, de-duplicated across every panel (labels drawn once via shared used_labels)
        handles, labels, seen = [], [], set()
        for ax in axs.flat:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in seen:
                    seen.add(ll); handles.append(hh); labels.append(ll)

        if handles:
            legend_mode = str(plot_cfg.get("legend_mode", "figure")).lower()
            if legend_mode in {"figure", "outside"}:
                fig.legend(
                    handles, labels,
                    loc=plot_cfg.get("legend_loc", "center right"),
                    bbox_to_anchor=tuple(plot_cfg.get("legend_bbox_to_anchor", [1.02, 0.5])),
                    frameon=bool(plot_cfg.get("legend_frame", False)),
                    ncol=int(plot_cfg.get("legend_ncol", 1)),
                )
            elif legend_mode in {"axis", "inside", "subplot"}:
                flat = list(axs.flat)
                ai = int(plot_cfg.get("legend_axis", 0))
                if ai < 0:
                    ai = len(flat) + ai
                if ai < 0 or ai >= len(flat):
                    raise IndexError(f"plot.legend_axis={plot_cfg.get('legend_axis')} out of range for {len(flat)} panels")
                flat[ai].legend(
                    handles, labels,
                    loc=plot_cfg.get("legend_loc", "upper right"),
                    frameon=bool(plot_cfg.get("legend_frame", True)),
                    fontsize=plot_cfg.get("legend_font_size", None),
                    title=plot_cfg.get("legend_title", None),
                    ncol=int(plot_cfg.get("legend_ncol", 1)),
                )
            elif legend_mode in {"none", "off", "false"}:
                pass
            else:
                raise ValueError(f"Unknown plot.legend_mode={legend_mode!r}; use figure, outside, axis, inside, subplot, none")

        title = plot_cfg.get("title", None)
        if title:
            fig.suptitle(str(title))

        for ext in formats:
            ext = str(ext).lstrip(".")
            path = (out_dir / name).with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))
        plt.close(fig)

    manifest = {
        "name": name, "spec_path": str(spec_path), "output_dir": str(out_dir),
        "csv": str(csv_path), "out_paths": out_paths,
        "columns": columns, "rows": rows, "num_records": len(records),
        "style": {
            "color_by": color_by, "marker_by": marker_by, "legend_by": legend_by,
            "color_map": color_map, "marker_map": marker_map, "legend_label_map": legend_label_map,
        },
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("Saved image-field grid manifest: %s", out_dir / f"{name}_manifest.json")
    return manifest
