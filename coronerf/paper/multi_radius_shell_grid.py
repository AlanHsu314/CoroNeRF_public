from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..artifacts.scalar_fields import build_scalar_shell_product, get_scalar_field_spec
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..benchmark.condition_compare import _build_state_for_analysis, _release_state
from .joint_shell_panels import _get_field_limits, _get_residual_limit
from .shell_panels import (
    _longitude_order,
    _resolve_path,
    _resolve_run_dir,
    _select_representative_row,
    _slug,
)

logger = logging.getLogger("coroNeRF.paper.multi_radius_shell_grid")


def _default_rows() -> list[dict]:
    return [
        {"quantity": "ne", "mode": "gt", "row_label": r"$n_e$ GT"},
        {"quantity": "ne", "mode": "pred", "row_label": r"$n_e$ pred."},
        {"quantity": "ne", "mode": "residual", "row_label": r"$n_e$ residual"},
        {"quantity": "temp", "mode": "gt", "row_label": r"$T$ GT"},
        {"quantity": "temp", "mode": "pred", "row_label": r"$T$ pred."},
        {"quantity": "temp", "mode": "residual", "row_label": r"$T$ residual"},
    ]


def _mode(row_cfg: dict) -> str:
    mode = str(row_cfg.get("mode", row_cfg.get("display", "pred"))).lower()
    aliases = {
        "prediction": "pred",
        "field": "pred",
        "ground_truth": "gt",
        "truth": "gt",
        "diff": "residual",
        "resid": "residual",
        "error": "residual",
        "abs_resid": "abs_residual",
        "abs_error": "abs_residual",
    }
    return aliases.get(mode, mode)


def _product_array(product, mode: str, lon_order: np.ndarray) -> np.ndarray:
    if mode == "gt":
        return np.asarray(product.gt_log)[:, lon_order]
    if mode == "pred":
        return np.asarray(product.pred_log)[:, lon_order]
    if mode == "residual":
        return np.asarray(product.diff_log)[:, lon_order]
    if mode == "abs_residual":
        return np.abs(np.asarray(product.diff_log)[:, lon_order])
    raise ValueError(f"Unknown row mode={mode!r}; expected gt, pred, residual, or abs_residual")


def _positive_limits(arrays: list[np.ndarray], cfg: dict) -> tuple[float, float]:
    if cfg.get("vmin", None) is not None and cfg.get("vmax", None) is not None:
        return float(cfg["vmin"]), float(cfg["vmax"])
    vals = np.concatenate([np.asarray(a).reshape(-1) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    pct = cfg.get("percentile", [1.0, 99.0])
    if isinstance(pct, (int, float)):
        pct = [0.0, float(pct)]
    lo = float(np.nanpercentile(vals, float(pct[0])))
    hi = float(np.nanpercentile(vals, float(pct[1])))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def _row_limits_and_cmap(
    row_cfg: dict,
    products_by_radius: dict[float, dict],
    lon_order: np.ndarray,
    plot_cfg: dict,
) -> tuple[float, float, str, str]:
    quantity = str(row_cfg["quantity"])
    spec = get_scalar_field_spec(quantity)
    mode = _mode(row_cfg)

    arrays = [_product_array(products_by_radius[r][quantity], mode, lon_order) for r in products_by_radius]

    if mode in {"gt", "pred"}:
        field_cfg = dict(row_cfg.get("field_color", {}) or {})
        # For field rows, use both GT and prediction over all requested radii so
        # GT and pred are comparable even when shown as separate rows.
        gt_all = []
        pred_all = []
        for r in products_by_radius:
            prod = products_by_radius[r][quantity]
            gt_all.append(np.asarray(prod.gt_log)[:, lon_order])
            pred_all.append(np.asarray(prod.pred_log)[:, lon_order])
        gt = np.concatenate([x.reshape(-1) for x in gt_all])
        pred = np.concatenate([x.reshape(-1) for x in pred_all])
        vmin, vmax = _get_field_limits(gt, pred, field_cfg)
        cmap = str(row_cfg.get("field_cmap", spec.cmap))
        label = str(field_cfg.get("label", spec.log_label))
        return float(vmin), float(vmax), cmap, label

    if mode == "residual":
        resid_cfg = dict(row_cfg.get("residual_color", {}) or {})
        diff = np.concatenate([np.asarray(a).reshape(-1) for a in arrays])
        dlim = _get_residual_limit(diff, resid_cfg)
        cmap = str(row_cfg.get("residual_cmap", plot_cfg.get("residual_cmap", "coolwarm")))
        label = str(resid_cfg.get("label", rf"$\Delta$ {spec.log_label}"))
        return -float(dlim), float(dlim), cmap, label

    if mode == "abs_residual":
        resid_cfg = dict(row_cfg.get("residual_color", {}) or {})
        vmin, vmax = _positive_limits(arrays, resid_cfg)
        cmap = str(row_cfg.get("residual_cmap", plot_cfg.get("abs_residual_cmap", "magma")))
        label = str(resid_cfg.get("label", rf"$|\Delta|$ {spec.log_label}"))
        return float(vmin), float(vmax), cmap, label

    raise ValueError(f"Unhandled row mode={mode!r}")


def _load_products_for_radii(
    row: dict,
    benchmark_dir: Path,
    quantities: list[str],
    radii: list[float],
    device: str | None,
    path_remap: dict | None,
) -> tuple[dict[float, dict], Path]:
    run_dir = _resolve_run_dir(row, benchmark_dir)
    if run_dir is None:
        raise FileNotFoundError(f"Could not resolve run_dir for selected row: {row}")

    logger.info("Loading selected run for multi-radius shell figure: %s", run_dir)
    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=path_remap)
    products: dict[float, dict] = {}
    try:
        for r in radii:
            products[float(r)] = {
                q: build_scalar_shell_product(state, quantity=q, r_target=float(r))
                for q in quantities
            }
    finally:
        _release_state(state)

    return products, run_dir


def _plot_multi_radius_shell_grid(
    products_by_radius: dict[float, dict],
    row_cfgs: list[dict],
    out_base: Path,
    plot_cfg: dict,
) -> tuple[dict, dict]:
    radii = list(products_by_radius.keys())
    first_prod = next(iter(next(iter(products_by_radius.values())).values()))
    lon_mode = str(plot_cfg.get("longitude_mode", "zero_360"))
    lon_plot, lon_order = _longitude_order(first_prod.lon_deg, lon_mode)
    lat_plot = np.asarray(first_prod.lat_deg, dtype=float)

    n_rows = len(row_cfgs)
    n_cols = len(radii)

    figsize = plot_cfg.get("figsize", [8.0, 8.6])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])

    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))

    extent = [
        float(np.nanmin(lon_plot)),
        float(np.nanmax(lon_plot)),
        float(np.nanmin(lat_plot)),
        float(np.nanmax(lat_plot)),
    ]

    rc = {
        "font.size": font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_font_size,
        "ytick.labelsize": tick_font_size,
        "figure.titlesize": title_font_size + 1,
    }

    npz_payload = {"lon_deg": lon_plot, "lat_deg": lat_plot}
    plot_info = {"rows": [], "radii": []}

    with plt.rc_context(rc):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, constrained_layout=True)

        fig.set_constrained_layout_pads(
            wspace=plot_cfg.get("wspace", 0.02), hspace=plot_cfg.get("hspace", 0.02),
            w_pad=plot_cfg.get("w_pad", 0.02),   h_pad=plot_cfg.get("h_pad", 0.02),
        )

        for i, row_cfg in enumerate(row_cfgs):
            quantity = str(row_cfg["quantity"])
            mode = _mode(row_cfg)
            vmin, vmax, cmap, cbar_label = _row_limits_and_cmap(row_cfg, products_by_radius, lon_order, plot_cfg)

            im = None
            for j, r in enumerate(radii):
                prod = products_by_radius[r][quantity]
                arr = _product_array(prod, mode, lon_order)
                npz_payload[f"{_slug(quantity)}_{_slug(mode)}_r{j}"] = arr

                ax = axs[i, j]
                im = ax.imshow(
                    arr,
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect=plot_cfg.get("aspect", "auto"),
                    interpolation=plot_cfg.get("interpolation", "nearest"),
                )

                if i == 0:
                    title = row_cfg.get("column_title", None)
                    if title is None:
                        title = plot_cfg.get("column_title_format", r"$r={r:.2f}\,R_\odot$").format(r=float(prod.r))
                    ax.set_title(str(title))

                if j == 0:
                    base_ylabel = str(plot_cfg.get("ylabel", "Latitude [deg]"))
                    row_label = row_cfg.get("row_label", None)
                    if row_label is None:
                        spec = get_scalar_field_spec(quantity)
                        row_label = f"{spec.display_name} {mode}"
                    row_label = "" if row_label is None else str(row_label)
                    ax.set_ylabel(f"{row_label}\n{base_ylabel}" if row_label.strip() else base_ylabel)
                else:
                    ax.set_ylabel("")

                if i == n_rows - 1:
                    ax.set_xlabel(plot_cfg.get("xlabel", "Longitude [deg]"))
                else:
                    ax.set_xlabel("")

            cbar = fig.colorbar(
                im,
                ax=axs[i, :].ravel().tolist(),
                fraction=float(plot_cfg.get("colorbar_fraction", 0.035)),
                pad=float(plot_cfg.get("colorbar_pad", 0.02)),
            )
            cbar.set_label(cbar_label)

            plot_info["rows"].append({
                "quantity": quantity,
                "mode": mode,
                "vmin": float(vmin),
                "vmax": float(vmax),
                "cmap": cmap,
                "label": cbar_label,
            })

        if plot_cfg.get("pack_ticks", False):
            for rr in range(n_rows):
                for cc in range(n_cols):
                    ax = axs[rr, cc]
                    if cc != 0:            ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
                    if rr != n_rows - 1:   ax.set_xticklabels([]); ax.tick_params(axis="x", length=0)

        if plot_cfg.get("title", None):
            fig.suptitle(str(plot_cfg["title"]))

        out_paths = []
        for ext in formats:
            ext = str(ext).lstrip(".")
            path = out_base.with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))

        plt.close(fig)

    for r in radii:
        any_prod = next(iter(products_by_radius[r].values()))
        plot_info["radii"].append({"requested": float(r), "actual": float(any_prod.r)})
    plot_info["out_paths"] = out_paths
    return plot_info, npz_payload


def generate_multi_radius_shell_grid(
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

    radii = [float(x) for x in cfg.get("radii", [1.2, 1.5, 1.9])]
    row_cfgs = list(cfg.get("rows", []) or [])
    if not row_cfgs:
        row_cfgs = _default_rows()

    quantities = list(dict.fromkeys(str(r["quantity"]) for r in row_cfgs))

    default_metric = str(cfg.get("selection_metric", "final/temp_shell_inner/shell_mean_abs_dlog_mean"))
    default_policy = str(cfg.get("selection_policy", "best"))
    default_mode = str(cfg.get("select_mode", "min"))

    rows = collect_run_summaries(benchmark_dir)
    rows = [r for r in rows if r.get("status") == "completed"]
    if not rows:
        raise ValueError(f"No completed run summaries found in {benchmark_dir}")

    run_cfg = dict(cfg.get("run", {}) or {})
    for key in ["experiment", "run_name", "run_dir", "seed", "selection_policy", "selection_metric", "select_mode", "where"]:
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

    products_by_radius, run_dir = _load_products_for_radii(
        row=row,
        benchmark_dir=benchmark_dir,
        quantities=quantities,
        radii=radii,
        device=device,
        path_remap=path_remap,
    )

    plot_info, npz_payload = _plot_multi_radius_shell_grid(
        products_by_radius=products_by_radius,
        row_cfgs=row_cfgs,
        out_base=out_dir / name,
        plot_cfg=dict(cfg.get("plot", {}) or {}),
    )

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
    logger.info("Saved multi-radius shell manifest: %s", out_dir / f"{name}_manifest.json")
    return manifest
