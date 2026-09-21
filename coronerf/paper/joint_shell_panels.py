from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

'''
-standard builds state used in condition compare
-load in useful helpers from shell panel plotting
'''


from ..artifacts.scalar_fields import build_scalar_shell_product, get_scalar_field_spec
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..benchmark.condition_compare import _build_state_for_analysis, _release_state
from .shell_panels import (
    _finite_percentile,
    _longitude_order,
    _resolve_path,
    _resolve_run_dir,
    _select_representative_row,
    _slug,
)

logger = logging.getLogger("coroNeRF.paper.joint_shell_panels")


def _get_field_limits(gt: np.ndarray, pred: np.ndarray, cfg: dict) -> tuple[float, float]:
    if cfg.get("vmin", None) is not None and cfg.get("vmax", None) is not None:
        return float(cfg["vmin"]), float(cfg["vmax"])

    pct = cfg.get("percentile", [1.0, 99.0])
    source = str(cfg.get("source", "gt"))

    if source == "gt":
        vals = gt
    elif source == "all":
        vals = np.concatenate([gt.reshape(-1), pred.reshape(-1)])
    else:
        raise ValueError("field_color.source must be 'gt' or 'all'")

    return _finite_percentile(vals, pct)

def _get_residual_limit(diff: np.ndarray, cfg: dict) -> float:
    if cfg.get("vlim", None) is not None:
        return float(cfg["vlim"])

    vals = np.abs(np.asarray(diff).reshape(-1))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0

    pct = float(cfg.get("percentile", 99.0))
    lim = float(np.nanpercentile(vals, pct))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(vals)) if vals.size else 1.0
    return max(lim, 1.0e-6)

def _reorder_product(product, lon_order: np.ndarray) -> dict:
    return {
        "quantity": product.quantity,
        "r": float(product.r),
        "gt": np.asarray(product.gt_log)[:, lon_order],
        "pred": np.asarray(product.pred_log)[:, lon_order],
        "diff": np.asarray(product.diff_log)[:, lon_order],
    }

def _resolve_quantity_style(qcfg: dict, gt, pred, diff, plot_cfg: dict) -> dict:
    """Resolve cmaps, color limits, and labels for one quantity row.
    Extracted verbatim from _plot_joint_shell_grid so the 3-col grid and the
    composite figure share identical styling."""
    quantity = str(qcfg["quantity"])
    spec = get_scalar_field_spec(quantity)

    field_cfg = dict(qcfg.get("field_color", {}) or {})
    resid_cfg = dict(qcfg.get("residual_color", {}) or {})

    vmin, vmax = _get_field_limits(gt, pred, field_cfg)
    dlim = _get_residual_limit(diff, resid_cfg)

    field_cmap = qcfg.get("field_cmap", spec.cmap)
    residual_cmap = qcfg.get("residual_cmap", plot_cfg.get("residual_cmap", "coolwarm"))

    row_label = qcfg.get("row_label", spec.display_name)
    row_label = "" if row_label is None else str(row_label)

    field_label = field_cfg.get("label", spec.log_label)
    residual_label = resid_cfg.get("label", rf"$\Delta$ {spec.log_label}")

    return {
        "quantity": quantity,
        "field_cmap": field_cmap,
        "residual_cmap": residual_cmap,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "dlim": float(dlim),
        "field_label": field_label,
        "residual_label": residual_label,
        "row_label": row_label,
    }

def _draw_shell_cell(ax, arr, cmap, vmin, vmax, extent, *,
                     title=None, xlabel=None, ylabel=None,
                     interpolation="nearest", aspect="auto"):
    """Draw one shell panel (imshow) into a given axes; return the image.
    Labels/title only set when provided (unset => matplotlib default '', which
    is what the original grid produced for non-edge cells)."""
    im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect=aspect, interpolation=interpolation)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return im

def _add_shell_colorbar(fig, im, ax, label, plot_cfg: dict):
    """Add a colorbar. Default (right/vertical) reproduces the original call
    byte-for-byte; colorbar_location: bottom|top switches to horizontal."""
    location = str(plot_cfg.get("colorbar_location", "right")).lower()
    fraction = float(plot_cfg.get("colorbar_fraction", 0.035))
    pad = float(plot_cfg.get("colorbar_pad", 0.02))
    if location in ("bottom", "top"):
        cbar = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad,
                            orientation="horizontal", location=location)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cbar.set_label(label)
    return cbar

def _load_products_for_selected_run(
    row: dict,
    benchmark_dir: Path,
    quantities: list[str],
    radius: float,
    device: str | None,
    path_remap: dict | None,
) -> tuple[dict, Path]:
    run_dir = _resolve_run_dir(row, benchmark_dir)
    if run_dir is None:
        raise FileNotFoundError(f"Could not resolve run_dir for selected row: {row}")

    logger.info("Loading selected run for joint shell figure: %s", run_dir)

    state = _build_state_for_analysis(
        run_dir,
        device_override=device,
        path_remap=path_remap,
    )

    try:
        products = {
            q: build_scalar_shell_product(state, quantity=q, r_target=float(radius))
            for q in quantities
        }
    finally:
        _release_state(state)

    return products, run_dir

def _plot_joint_shell_grid(
    products: dict,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    quantity_cfgs: list[dict],
    out_base: Path,
    plot_cfg: dict,
) -> dict:
    n_rows = len(quantity_cfgs)
    n_cols = 3

    figsize = plot_cfg.get("figsize", [8.2, 5.4])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])

    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))
    aspect = plot_cfg.get("aspect", "auto")
    interpolation = plot_cfg.get("interpolation", "nearest")

    extent = [
        float(np.nanmin(lon_deg)), float(np.nanmax(lon_deg)),
        float(np.nanmin(lat_deg)), float(np.nanmax(lat_deg)),
    ]

    rc = {
        "font.size": font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_font_size,
        "ytick.labelsize": tick_font_size,
        "figure.titlesize": title_font_size + 1,
    }

    plot_info = {"rows": []}
    base_ylabel = str(plot_cfg.get("ylabel", "Latitude [deg]"))
    xlabel_text = plot_cfg.get("xlabel", "Longitude [deg]")

    with plt.rc_context(rc):
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True,
        )

        fig.set_constrained_layout_pads(
            wspace=plot_cfg.get("wspace", 0.02), hspace=plot_cfg.get("hspace", 0.02),
            w_pad=plot_cfg.get("w_pad", 0.02),   h_pad=plot_cfg.get("h_pad", 0.02),
        )

        for i, qcfg in enumerate(quantity_cfgs):
            prod = products[str(qcfg["quantity"])]
            gt, pred, diff = prod["gt"], prod["pred"], prod["diff"]

            style = _resolve_quantity_style(qcfg, gt, pred, diff, plot_cfg)
            if style["row_label"].strip():
                col0_ylabel = f"{style['row_label']}\n{base_ylabel}"
            else:
                col0_ylabel = base_ylabel

            panels = [
                ("Ground truth", gt,   style["field_cmap"],    style["vmin"],  style["vmax"]),
                ("Prediction",   pred, style["field_cmap"],    style["vmin"],  style["vmax"]),
                ("Residual",     diff, style["residual_cmap"], -style["dlim"], style["dlim"]),
            ]

            ims = []
            for j, (title, arr, cmap, lo, hi) in enumerate(panels):
                im = _draw_shell_cell(
                    axs[i, j], arr, cmap, lo, hi, extent,
                    title=(title if i == 0 else None),
                    ylabel=(col0_ylabel if j == 0 else None),
                    xlabel=(xlabel_text if i == n_rows - 1 else None),
                    interpolation=interpolation, aspect=aspect,
                )
                ims.append(im)

            _add_shell_colorbar(fig, ims[1], [axs[i, 0], axs[i, 1]], style["field_label"], plot_cfg)
            _add_shell_colorbar(fig, ims[2], axs[i, 2], style["residual_label"], plot_cfg)

            plot_info["rows"].append({
                "quantity": style["quantity"],
                "field_vmin": style["vmin"],
                "field_vmax": style["vmax"],
                "residual_vlim": style["dlim"],
            })

        if plot_cfg.get("pack_ticks", False):
            for rr in range(n_rows):
                for cc in range(n_cols):
                    ax = axs[rr, cc]
                    if cc != 0:            ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
                    if rr != n_rows - 1:   ax.set_xticklabels([]); ax.tick_params(axis="x", length=0)

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

def generate_joint_shell_figure(
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

    device = device_override
    if device is None:
        device = cfg.get("device", None)

    radius = float(cfg.get("radius", 1.5))

    quantity_cfgs = list(cfg.get("quantities", []) or [])
    if not quantity_cfgs:
        quantity_cfgs = [
            {"quantity": "ne"},
            {"quantity": "temp"},
        ]
    quantities = [str(q["quantity"]) for q in quantity_cfgs]

    default_metric = str(cfg.get("selection_metric", "final/temp_shell_inner/shell_mean_abs_dlog_mean"))
    default_policy = str(cfg.get("selection_policy", "best"))
    default_mode = str(cfg.get("select_mode", "min"))

    rows = collect_run_summaries(benchmark_dir)
    rows = [r for r in rows if r.get("status") == "completed"]
    if not rows:
        raise ValueError(f"No completed run summaries found in {benchmark_dir}")

    run_cfg = dict(cfg.get("run", {}) or {})
    # Also allow experiment/run_name/run_dir at top level for convenience.
    for key in ["experiment", "run_name", "run_dir", "seed", "selection_policy", "selection_metric", "select_mode"]:
        if key in cfg and key not in run_cfg:
            run_cfg[key] = cfg[key]

    row = _select_representative_row(
        rows=rows,
        benchmark_dir=benchmark_dir,
        spec_root=spec_root,
        panel_cfg=run_cfg,
        default_metric=default_metric,
        default_policy=default_policy,
        default_mode=default_mode,
    )

    products_raw, run_dir = _load_products_for_selected_run(
        row=row,
        benchmark_dir=benchmark_dir,
        quantities=quantities,
        radius=radius,
        device=device,
        path_remap=path_remap,
    )

    first_product = products_raw[quantities[0]]
    plot_cfg = dict(cfg.get("plot", {}) or {})
    lon_mode = str(plot_cfg.get("longitude_mode", "zero_360"))
    lon_plot, lon_order = _longitude_order(first_product.lon_deg, lon_mode)
    lat_plot = np.asarray(first_product.lat_deg, dtype=float)

    products = {
        q: _reorder_product(prod, lon_order)
        for q, prod in products_raw.items()
    }

    out_base = out_dir / name
    plot_info = _plot_joint_shell_grid(
        products=products,
        lon_deg=lon_plot,
        lat_deg=lat_plot,
        quantity_cfgs=quantity_cfgs,
        out_base=out_base,
        plot_cfg=plot_cfg,
    )

    npz_payload = {
        "lon_deg": lon_plot,
        "lat_deg": lat_plot,
    }
    for q, prod in products.items():
        key = _slug(q)
        npz_payload[f"{key}_gt_log"] = prod["gt"]
        npz_payload[f"{key}_pred_log"] = prod["pred"]
        npz_payload[f"{key}_diff_log"] = prod["diff"]

    npz_path = out_dir / f"{name}_arrays.npz"
    np.savez_compressed(npz_path, **npz_payload)

    manifest = {
        "name": name,
        "spec_path": str(spec_path),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(out_dir),
        "run_dir": str(run_dir),
        "run_name": row.get("run_name"),
        "experiment": row.get("experiment_name"),
        "seed": row.get("seed"),
        "radius_requested": radius,
        "radius_actual": float(first_product.r),
        "selection_metric": default_metric,
        "selection_metric_value": None if row.get(default_metric) is None else float(row.get(default_metric)),
        "selection_policy": default_policy,
        "select_mode": default_mode,
        "device": device,
        "path_remap": path_remap,
        "arrays_npz": str(npz_path),
        **plot_info,
    }

    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("Saved joint shell figure manifest: %s", out_dir / f"{name}_manifest.json")
    return manifest