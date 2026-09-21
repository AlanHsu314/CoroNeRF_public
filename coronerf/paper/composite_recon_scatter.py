# coronerf/paper/composite_recon_scatter.py
'''
joint scatter and shell panel plot, for proposals that don't have much space to show plots

'''
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from .shell_panels import _resolve_path, _select_representative_row, _longitude_order, _slug
from .joint_shell_panels import (
    _load_products_for_selected_run,
    _reorder_product,
    _resolve_quantity_style,
    _draw_shell_cell,
    _add_shell_colorbar,
)
from .image_field_scatter import (
    _collect_records,
    _build_value_map,
    _write_records_csv,
    _scatter_into_ax,
)

logger = logging.getLogger("coroNeRF.paper.composite_recon_scatter")

_DEFAULT_SCATTER_YLABEL = {
    "ne": r"$\mathrm{MAE}_{\mathrm{inner}}(\log_{10}n_e)$",
    "temp": r"$\mathrm{MAE}_{\mathrm{inner}}(\log_{10}T)$",
}


def _plot_composite(
    products, quantity_cfgs, lon_deg, lat_deg,
    records, x_metric, y_by_field, scatter_fields,
    color_by, marker_by, legend_by,
    color_map, marker_map, legend_label_map,
    style_cfg, plot_cfg, out_base,
) -> dict:
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])
    figsize = plot_cfg.get("figsize", [10.0, 5.0])
    width_ratios = plot_cfg.get("width_ratios", [2, 1])

    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))
    aspect = plot_cfg.get("aspect", "auto")
    interpolation = plot_cfg.get("interpolation", "nearest")

    extent = [
        float(np.nanmin(lon_deg)), float(np.nanmax(lon_deg)),
        float(np.nanmin(lat_deg)), float(np.nanmax(lat_deg)),
    ]
    base_ylabel = str(plot_cfg.get("ylabel", "Latitude [deg]"))
    xlabel_text = plot_cfg.get("xlabel_maps", "Longitude [deg]")

    rc = {
        "font.size": font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_font_size,
        "ytick.labelsize": tick_font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": title_font_size + 1,
    }

    n_rows = len(quantity_cfgs)
    plot_info = {"rows": []}

    scatter_ylabels = dict(plot_cfg.get("scatter_ylabels", {}) or {})
    scatter_titles = dict(plot_cfg.get("scatter_titles", {}) or {})
    share_x = bool(plot_cfg.get("scatter_share_x", True))
    legend_axis = int(plot_cfg.get("legend_axis", 0))

    with plt.rc_context(rc):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        outer = fig.add_gridspec(1, 2, width_ratios=width_ratios)

        # ---- LEFT: 2x2 maps (GT, Residual) x quantities ----
        left = outer[0, 0].subgridspec(n_rows, 2)
        for i, qcfg in enumerate(quantity_cfgs):
            prod = products[str(qcfg["quantity"])]
            gt, pred, diff = prod["gt"], prod["pred"], prod["diff"]
            style = _resolve_quantity_style(qcfg, gt, pred, diff, plot_cfg)

            col0_ylabel = (
                f"{style['row_label']}\n{base_ylabel}"
                if style["row_label"].strip() else base_ylabel
            )
            cell_xlabel = xlabel_text if i == n_rows - 1 else None

            ax_gt = fig.add_subplot(left[i, 0])
            ax_res = fig.add_subplot(left[i, 1])

            im_gt = _draw_shell_cell(
                ax_gt, gt, style["field_cmap"], style["vmin"], style["vmax"], extent,
                title=("Ground truth" if i == 0 else None),
                ylabel=col0_ylabel, xlabel=cell_xlabel,
                interpolation=interpolation, aspect=aspect,
            )
            im_res = _draw_shell_cell(
                ax_res, diff, style["residual_cmap"], -style["dlim"], style["dlim"], extent,
                title=("Residual" if i == 0 else None),
                ylabel=None, xlabel=cell_xlabel,
                interpolation=interpolation, aspect=aspect,
            )
            _add_shell_colorbar(fig, im_gt, ax_gt, style["field_label"], plot_cfg)
            _add_shell_colorbar(fig, im_res, ax_res, style["residual_label"], plot_cfg)

            plot_info["rows"].append({
                "quantity": style["quantity"],
                "field_vmin": style["vmin"],
                "field_vmax": style["vmax"],
                "residual_vlim": style["dlim"],
            })

        # ---- RIGHT: one scatter per field, stacked vertically (shared x) ----
        n_sc = len(scatter_fields)
        right = outer[0, 1].subgridspec(n_sc, 1)
        used_labels = set()
        sc_axes = []
        for k, field in enumerate(scatter_fields):
            shared = sc_axes[0] if (share_x and sc_axes) else None
            ax_sc = fig.add_subplot(right[k, 0], sharex=shared)
            is_last = (k == n_sc - 1)
            _scatter_into_ax(
                ax_sc, records, x_metric=x_metric, y_metric=str(y_by_field[field]),
                color_by=color_by, marker_by=marker_by, legend_by=legend_by,
                color_map=color_map, marker_map=marker_map, legend_label_map=legend_label_map,
                style_cfg=style_cfg, plot_cfg=plot_cfg, used_labels=used_labels,
                xlabel=(plot_cfg.get("scatter_xlabel", x_metric) if is_last else None),
                ylabel=scatter_ylabels.get(field, _DEFAULT_SCATTER_YLABEL.get(field, str(y_by_field[field]))),
                title=scatter_titles.get(field, None),
                yscale=plot_cfg.get("scatter_yscale", plot_cfg.get("yscale", None)),
                ylim=plot_cfg.get("scatter_ylim", None),
            )
            if share_x and not is_last:
                ax_sc.tick_params(labelbottom=False)
            sc_axes.append(ax_sc)

        # legend: handles live on the first scatter drawn; place on chosen panel
        if sc_axes and str(plot_cfg.get("legend_mode", "axis")).lower() not in ("none", "off", "false"):
            handles, labels = sc_axes[0].get_legend_handles_labels()
            if handles:
                idx = legend_axis if legend_axis >= 0 else len(sc_axes) + legend_axis
                idx = max(0, min(idx, len(sc_axes) - 1))
                sc_axes[idx].legend(
                    handles, labels,
                    loc=plot_cfg.get("legend_loc", "best"),
                    frameon=bool(plot_cfg.get("legend_frame", True)),
                    fontsize=plot_cfg.get("legend_font_size", None),
                    title=plot_cfg.get("legend_title", None),
                )

        # optional free-text note (default off; kept for flexibility)
        note = plot_cfg.get("scatter_note", None)
        if note and sc_axes:
            sc_axes[-1].text(
                float(plot_cfg.get("scatter_note_x", 0.97)),
                float(plot_cfg.get("scatter_note_y", 0.03)),
                str(note), transform=sc_axes[-1].transAxes, ha="right", va="bottom",
                fontsize=plot_cfg.get("scatter_note_fontsize", tick_font_size), style="italic",
            )

        title = plot_cfg.get("title", None)
        if title:
            fig.suptitle(str(title))

        out_paths = []
        for ext in formats:
            ext = str(ext).lstrip(".")
            path = out_base.with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))
        plt.close(fig)

    plot_info["out_paths"] = out_paths
    return plot_info


def generate_composite_recon_scatter(
    spec_path: Union[str, Path],
    device_override: str | None = None,
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
        out_dir = _resolve_path(cfg.get("output_dir", "../../paper_outputs/figures/composite"), spec_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_override if device_override is not None else cfg.get("device", None)
    path_remap = cfg.get("path_remap", None)

    # ---------------- MAPS (reuse joint-shell data path) ----------------
    maps_cfg = dict(cfg.get("maps", {}) or {})
    benchmark_dir = _resolve_path(maps_cfg["benchmark_dir"], spec_root)
    radius = float(maps_cfg.get("radius", 1.5))

    quantity_cfgs = list(maps_cfg.get("quantities", []) or [])
    if not quantity_cfgs:
        quantity_cfgs = [{"quantity": "ne"}, {"quantity": "temp"}]
    quantities = [str(q["quantity"]) for q in quantity_cfgs]

    default_metric = str(maps_cfg.get("selection_metric", "final/temp_shell_inner/shell_mean_abs_dlog_mean"))
    default_policy = str(maps_cfg.get("selection_policy", "best"))
    default_mode = str(maps_cfg.get("select_mode", "min"))

    rows = collect_run_summaries(benchmark_dir)
    rows = [r for r in rows if r.get("status") == "completed"]
    if not rows:
        raise ValueError(f"No completed run summaries found in {benchmark_dir}")

    run_cfg = dict(maps_cfg.get("run", {}) or {})
    for key in ["experiment", "run_name", "run_dir", "seed", "selection_policy", "selection_metric", "select_mode"]:
        if key in maps_cfg and key not in run_cfg:
            run_cfg[key] = maps_cfg[key]

    row = _select_representative_row(
        rows=rows, benchmark_dir=benchmark_dir, spec_root=spec_root,
        panel_cfg=run_cfg, default_metric=default_metric,
        default_policy=default_policy, default_mode=default_mode,
    )

    products_raw, run_dir = _load_products_for_selected_run(
        row=row, benchmark_dir=benchmark_dir, quantities=quantities,
        radius=radius, device=device, path_remap=path_remap,
    )

    plot_cfg = dict(cfg.get("plot", {}) or {})
    first_product = products_raw[quantities[0]]
    lon_mode = str(plot_cfg.get("longitude_mode", "zero_360"))
    lon_plot, lon_order = _longitude_order(first_product.lon_deg, lon_mode)
    lat_plot = np.asarray(first_product.lat_deg, dtype=float)
    products = {q: _reorder_product(prod, lon_order) for q, prod in products_raw.items()}

    # ---------------- SCATTER (reuse aggregation) ----------------
    scatter_cfg = dict(cfg.get("scatter", {}) or {})
    x_metric = str(scatter_cfg.get("x_metric", "final/val_loss"))
    
    y_by_field = {
        "ne": str(scatter_cfg.get("y_metric_ne", "final/ne_shell_inner/shell_mean_abs_dlog_mean")),
        "temp": str(scatter_cfg.get("y_metric_temp", "final/temp_shell_inner/shell_mean_abs_dlog_mean")),
    }
    # Accept scatter_fields: [ne, temp] (preferred) or scatter_field: ne (back-compat).
    _fields = scatter_cfg.get("scatter_fields", None)
    if _fields is None:
        _fields = [scatter_cfg.get("scatter_field", "ne")]
    if not isinstance(_fields, list):
        _fields = [_fields]
    scatter_fields = [str(f) for f in _fields]
    for f in scatter_fields:
        if f not in y_by_field:
            raise ValueError(f"scatter fields must be in {list(y_by_field)}, got {f!r}")

    records = _collect_records({"benchmarks": scatter_cfg.get("benchmarks", [])}, spec_root)
    if not records:
        raise ValueError("No completed run records found for scatter panel.")

    style_cfg = dict(scatter_cfg.get("style", {}) or {})
    color_by = str(style_cfg.get("color_by", "paper_label"))
    marker_by = str(style_cfg.get("marker_by", "paper_label"))
    legend_by = str(style_cfg.get("legend_by", "paper_label"))
    legend_label_map = style_cfg.get("legend_labels", None)

    color_map = _build_value_map(records, color_by, style_cfg.get("colors", None),
                                 fallback_values=["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"])
    marker_map = _build_value_map(records, marker_by, style_cfg.get("markers", None),
                                  fallback_values=["o","s","^","D","P","X","v","<",">"])

    # ---------------- COMPOSE + SAVE ----------------
    out_base = out_dir / name
    plot_info = _plot_composite(
        products=products, quantity_cfgs=quantity_cfgs,
        lon_deg=lon_plot, lat_deg=lat_plot,
        records=records, x_metric=x_metric, y_by_field=y_by_field, scatter_fields=scatter_fields,
        color_by=color_by, marker_by=marker_by, legend_by=legend_by,
        color_map=color_map, marker_map=marker_map, legend_label_map=legend_label_map,
        style_cfg=style_cfg, plot_cfg=plot_cfg, out_base=out_base,
    )

    npz_payload = {"lon_deg": lon_plot, "lat_deg": lat_plot}
    for q, prod in products.items():
        key = _slug(q)
        npz_payload[f"{key}_gt_log"] = prod["gt"]
        npz_payload[f"{key}_pred_log"] = prod["pred"]
        npz_payload[f"{key}_diff_log"] = prod["diff"]
    npz_path = out_dir / f"{name}_arrays.npz"
    np.savez_compressed(npz_path, **npz_payload)

    csv_path = out_dir / f"{name}_scatter_data.csv"
    _write_records_csv(records, csv_path, x_metric=x_metric,
                       panels=[{"y_metric": y_by_field[f]} for f in scatter_fields])

    manifest = {
        "name": name, "spec_path": str(spec_path),
        "benchmark_dir": str(benchmark_dir), "output_dir": str(out_dir),
        "run_dir": str(run_dir), "run_name": row.get("run_name"),
        "experiment": row.get("experiment_name"), "seed": row.get("seed"),
        "radius_requested": radius, "radius_actual": float(first_product.r),
        "scatter_x_metric": x_metric,
        "scatter_fields": scatter_fields,
        "scatter_y_metrics": [y_by_field[f] for f in scatter_fields],
        "device": device, "path_remap": path_remap,
        "arrays_npz": str(npz_path), "scatter_csv": str(csv_path),
        **plot_info,
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("Saved composite figure manifest: %s", out_dir / f"{name}_manifest.json")
    return manifest