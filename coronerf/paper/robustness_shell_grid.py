from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..artifacts.scalar_fields import build_scalar_shell_product, get_scalar_field_spec
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..benchmark.condition_compare import _build_state_for_analysis, _release_state
from .joint_shell_panels import _get_field_limits
from .shell_panels import _longitude_order, _resolve_path, _resolve_run_dir, _select_representative_row, _slug

logger = logging.getLogger("coroNeRF.paper.robustness_shell_grid")


'''
IMPORTANT: same plotting format as shell_panels.py

generic shell reconstruction script, each column is a specific run
for example, in noise ablation, we show robustness with different columns being different noise mult
'''

def _load_products(row, benchmark_dir, quantities, radius, device, path_remap):
    run_dir = _resolve_run_dir(row, benchmark_dir)
    if run_dir is None:
        raise FileNotFoundError(f"Could not resolve run_dir for row: {row}")

    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=path_remap)
    try:
        products = {
            q: build_scalar_shell_product(state, quantity=q, r_target=float(radius))
            for q in quantities
        }
    finally:
        _release_state(state)

    return products, run_dir

def _reorder(arr: np.ndarray, lon_order: np.ndarray) -> np.ndarray:
    return np.asarray(arr)[:, lon_order]

def _quantity_mode(qcfg: dict) -> str:
    mode = str(qcfg.get("mode", qcfg.get("display", "field"))).lower()
    aliases = {
        "pred": "field",
        "prediction": "field",
        "diff": "residual",
        "error": "residual",
        "resid": "residual",
    }
    return aliases.get(mode, mode)

def _plot_array_for_mode(product, col_cfg: dict, mode: str, lon_order: np.ndarray) -> np.ndarray:
    source = str(col_cfg.get("source", "pred"))

    if mode == "field":
        if source == "gt":
            return _reorder(product.gt_log, lon_order)
        return _reorder(product.pred_log, lon_order)

    if mode == "residual":
        if source == "gt":
            return np.zeros_like(_reorder(product.gt_log, lon_order))
        return _reorder(product.diff_log, lon_order)

    if mode == "abs_residual":
        if source == "gt":
            return np.zeros_like(_reorder(product.gt_log, lon_order))
        return np.abs(_reorder(product.diff_log, lon_order))

    if mode == "gt":
        return _reorder(product.gt_log, lon_order)

    raise ValueError(
        "Unknown quantity mode/display "
        f"{mode!r}; expected field, residual, abs_residual, or gt"
    )

def _symmetric_limit(arrays: list[np.ndarray], cfg: dict) -> float:
    if cfg.get("vlim", None) is not None:
        return float(cfg["vlim"])

    vals = np.concatenate([np.abs(np.asarray(x).reshape(-1)) for x in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0

    pct = float(cfg.get("percentile", 99.0))
    lim = float(np.nanpercentile(vals, pct))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(vals)) if vals.size else 1.0
    return max(lim, 1.0e-6)

def _positive_limit(arrays: list[np.ndarray], cfg: dict) -> tuple[float, float]:
    if cfg.get("vmin", None) is not None and cfg.get("vmax", None) is not None:
        return float(cfg["vmin"]), float(cfg["vmax"])

    vals = np.concatenate([np.asarray(x).reshape(-1) for x in arrays])
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

def generate_robustness_shell_grid(
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
    radius = float(cfg.get("radius", 1.5))

    quantity_cfgs = list(cfg.get("quantities", []) or [{"quantity": "ne"}, {"quantity": "temp"}])
    # Preserve order while avoiding duplicate loads when the same quantity appears in multiple rows, for example field + residual.
    quantities = list(dict.fromkeys(str(q["quantity"]) for q in quantity_cfgs))

    columns = list(cfg.get("columns", []) or [])
    if not columns:
        raise ValueError("figure.columns must be non-empty")

    default_metric = str(cfg.get("selection_metric", "final/temp_shell_inner/shell_mean_abs_dlog_mean"))
    default_policy = str(cfg.get("selection_policy", "best"))
    default_mode = str(cfg.get("select_mode", "min"))

    rows = collect_run_summaries(benchmark_dir)
    rows = [r for r in rows if r.get("status") == "completed"]

    selected = []
    products_by_col = {}
    run_dirs_by_col = {}

    first_non_gt_product = None

    for j, col_cfg in enumerate(columns):
        if str(col_cfg.get("source", "pred")) == "gt":
            selected.append({"source": "gt", "label": col_cfg.get("label", "Ground truth")})
            continue

        row = _select_representative_row(
            rows=rows,
            benchmark_dir=benchmark_dir,
            spec_root=spec_root,
            panel_cfg=col_cfg,
            default_metric=col_cfg.get("selection_metric", default_metric),
            default_policy=col_cfg.get("selection_policy", default_policy),
            default_mode=col_cfg.get("select_mode", default_mode),
        )

        products, run_dir = _load_products(
            row=row,
            benchmark_dir=benchmark_dir,
            quantities=quantities,
            radius=radius,
            device=device,
            path_remap=path_remap,
        )

        products_by_col[j] = products
        run_dirs_by_col[j] = run_dir
        selected.append({
            "source": "pred",
            "label": col_cfg.get("label", row.get("experiment_name")),
            "experiment": row.get("experiment_name"),
            "run_name": row.get("run_name"),
            "run_dir": str(run_dir),
            "seed": row.get("seed"),
            "where": col_cfg.get("where", None),
        })

        if first_non_gt_product is None:
            first_non_gt_product = next(iter(products.values()))

    if first_non_gt_product is None:
        raise ValueError("At least one non-GT column is required")

    plot_cfg = dict(cfg.get("plot", {}) or {})
    lon_mode = str(plot_cfg.get("longitude_mode", "zero_360"))
    lon_plot, lon_order = _longitude_order(first_non_gt_product.lon_deg, lon_mode)
    lat_plot = np.asarray(first_non_gt_product.lat_deg, dtype=float)

    # Fill GT columns using the first loaded product.
    for j, col_cfg in enumerate(columns):
        if str(col_cfg.get("source", "pred")) == "gt":
            products_by_col[j] = {
                q: products_by_col[next(k for k in products_by_col.keys())][q]
                for q in quantities
            }

    n_rows = len(quantity_cfgs)
    n_cols = len(columns)

    figsize = plot_cfg.get("figsize", [10.2, 4.8])
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
    }

    npz_payload = {
        "lon_deg": lon_plot,
        "lat_deg": lat_plot,
    }

    with plt.rc_context(rc):
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(
            wspace=plot_cfg.get("wspace", 0.02), hspace=plot_cfg.get("hspace", 0.02),
             w_pad=plot_cfg.get("w_pad", 0.02),   h_pad=plot_cfg.get("h_pad", 0.02),
        )

        for i, qcfg in enumerate(quantity_cfgs):
            quantity = str(qcfg["quantity"])
            spec = get_scalar_field_spec(quantity)
            mode = _quantity_mode(qcfg)

            row_label = qcfg.get("row_label", spec.display_name)
            row_label = "" if row_label is None else str(row_label)

            arrays = []
            for j, col_cfg in enumerate(columns):
                prod = products_by_col[j][quantity]
                arrays.append(_plot_array_for_mode(prod, col_cfg, mode, lon_order))

            if mode == "field":
                gt = _reorder(products_by_col[0][quantity].gt_log, lon_order)
                field_cfg = dict(qcfg.get("field_color", {}) or {})
                vmin, vmax = _get_field_limits(
                    gt,
                    np.concatenate([x.reshape(1, -1) for x in arrays]).reshape(-1),
                    field_cfg,
                )
                cmap = qcfg.get("field_cmap", spec.cmap)
                cbar_label = field_cfg.get("label", spec.log_label)

            elif mode == "residual":
                resid_cfg = dict(qcfg.get("residual_color", {}) or {})
                dlim = _symmetric_limit(arrays, resid_cfg)
                vmin, vmax = -dlim, dlim
                cmap = qcfg.get("residual_cmap", plot_cfg.get("residual_cmap", "coolwarm"))
                cbar_label = resid_cfg.get("label", rf"$\Delta$ {spec.log_label}")

            elif mode == "abs_residual":
                resid_cfg = dict(qcfg.get("residual_color", {}) or {})
                vmin, vmax = _positive_limit(arrays, resid_cfg)
                cmap = qcfg.get("residual_cmap", plot_cfg.get("abs_residual_cmap", "magma"))
                cbar_label = resid_cfg.get("label", rf"$|\Delta|$ {spec.log_label}")

            elif mode == "gt":
                field_cfg = dict(qcfg.get("field_color", {}) or {})
                vmin, vmax = _positive_limit(arrays, field_cfg)
                cmap = qcfg.get("field_cmap", spec.cmap)
                cbar_label = field_cfg.get("label", spec.log_label)

            else:
                raise ValueError(f"Unhandled quantity mode: {mode}")

            im = None
            for j, col_cfg in enumerate(columns):
                arr = arrays[j]
                npz_payload[f"{_slug(quantity)}_{_slug(mode)}_{j}_{_slug(selected[j]['label'])}"] = arr

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
                    ax.set_title(str(col_cfg.get("label", selected[j]["label"])))

                if j == 0:
                    base_ylabel = str(plot_cfg.get("ylabel", "Latitude [deg]"))
                    if row_label.strip():
                        ax.set_ylabel(f"{row_label}\n{base_ylabel}")
                    else:
                        ax.set_ylabel(base_ylabel)
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
            path = (out_dir / name).with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))

        plt.close(fig)

    npz_path = out_dir / f"{name}_arrays.npz"
    np.savez_compressed(npz_path, **npz_payload)

    manifest = {
        "name": name,
        "spec_path": str(spec_path),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(out_dir),
        "radius": radius,
        "device": device,
        "path_remap": path_remap,
        "columns": selected,
        "arrays_npz": str(npz_path),
        "out_paths": out_paths,
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    return manifest