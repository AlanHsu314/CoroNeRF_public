from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..benchmark.condition_compare import _build_state_for_analysis, _release_state
from ..artifacts.scalar_fields import build_scalar_shell_product, get_scalar_field_spec, build_scalar_slice_product

logger = logging.getLogger("coroNeRF.paper.shell_panels")


def _resolve_path(path: Union[str, Path], root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (root / path).resolve()

def _slug(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "panel"

def _resolve_run_dir(row: dict, benchmark_dir: Path) -> Path | None:
    run_dir_raw = row.get("run_dir")
    run_name = row.get("run_name")

    if run_dir_raw:
        p = Path(run_dir_raw)
        if p.exists():
            return p

    if run_name:
        p = benchmark_dir / "runs" / str(run_name)
        if p.exists():
            return p

    return None

def _metric_value(row: dict, key: str) -> float:
    val = row.get(key, None)
    if val is None:
        raise KeyError(f"Metric {key!r} missing from run {row.get('run_name')}")
    return float(val)

def _seed_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if str(a) == str(b):
        return True
    try:
        return int(a) == int(b)
    except Exception:
        pass
    try:
        return int(str(a), 0) == int(str(b), 0)
    except Exception:
        return False
    
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
    Match a run-summary row against a dictionary of required values.

    Example:
      where:
        sweep.load_data_kwargs.train_num_views: 300

    The helper also accepts alternatives:
      train_num_views
      load_data_kwargs.train_num_views
    as long as that exact key exists in the row.
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

def _select_representative_row(
    rows: list[dict],
    benchmark_dir: Path,
    spec_root: Path,
    panel_cfg: dict,
    default_metric: str,
    default_policy: str,
    default_mode: str,
) -> dict:
    """
    Select one completed run for a panel.

    Supported panel selectors:
      - run_dir
      - run_name
      - experiment + selection_policy
      - experiment + seed
    """
    if panel_cfg.get("run_dir", None) is not None:
        run_dir = _resolve_path(panel_cfg["run_dir"], spec_root)
        if not run_dir.exists():
            raise FileNotFoundError(f"Panel run_dir does not exist: {run_dir}")
        return {
            "status": "completed",
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "experiment_name": panel_cfg.get("experiment", panel_cfg.get("label", run_dir.name)),
        }

    if panel_cfg.get("run_name", None) is not None:
        run_name = str(panel_cfg["run_name"])
        matches = [
            r for r in rows
            if r.get("status") == "completed" and str(r.get("run_name")) == run_name
        ]
        if not matches:
            raise ValueError(f"No completed run found with run_name={run_name}")
        return matches[0]

    exp = panel_cfg.get("experiment", None)
    if exp is None:
        raise ValueError("Non-GT panels require one of: run_dir, run_name, experiment")

    matches = [
        r for r in rows
        if r.get("status") == "completed" and str(r.get("experiment_name")) == str(exp)
    ]

    # patch, where can match with sweep...train_num_views or just train_num_views
    where = panel_cfg.get("where", None) or panel_cfg.get("filters", None)
    matches = [r for r in matches if _row_matches_where(r, where)]
    
    if not matches:
        raise ValueError(f"No completed runs found for experiment={exp}")

    seed = panel_cfg.get("seed", None)
    if seed is not None:
        seed_matches = [r for r in matches if _seed_equal(r.get("seed"), seed)]
        if not seed_matches:
            raise ValueError(f"No completed run found for experiment={exp}, seed={seed}")
        return seed_matches[0]

    metric_key = panel_cfg.get("selection_metric", default_metric)
    policy = panel_cfg.get("selection_policy", default_policy)
    mode = panel_cfg.get("select_mode", default_mode)

    if policy in {"best", "median"}:
        metric_matches = [r for r in matches if r.get(metric_key) is not None]
        if not metric_matches:
            raise ValueError(
                f"No completed runs for experiment={exp} contain metric={metric_key}"
            )

        if policy == "best":
            if mode == "min":
                return min(metric_matches, key=lambda r: _metric_value(r, metric_key))
            if mode == "max":
                return max(metric_matches, key=lambda r: _metric_value(r, metric_key))
            raise ValueError(f"Unknown select_mode={mode}")

        vals = np.asarray([_metric_value(r, metric_key) for r in metric_matches], dtype=float)
        med = float(np.median(vals))
        return min(metric_matches, key=lambda r: abs(_metric_value(r, metric_key) - med))

    if policy == "first":
        return sorted(matches, key=lambda r: str(r.get("run_name")))[0]

    raise ValueError(f"Unknown selection_policy={policy}")

def _load_shell_product_for_row(
    row: dict,
    benchmark_dir: Path,
    quantity: str,
    radius: float,
    device_override: str | None,
    path_remap: dict | None = None,
    slice_cfg: dict | None = None,
):
    run_dir = _resolve_run_dir(row, benchmark_dir)
    if run_dir is None:
        raise FileNotFoundError(f"Could not resolve run dir for row={row}")

    logger.info(
        "Loading run for shell figure | experiment=%s | run=%s | run_dir=%s",
        row.get("experiment_name"),
        row.get("run_name"),
        run_dir,
    )

    state = _build_state_for_analysis(run_dir, 
                                      device_override=device_override,
                                      path_remap=path_remap,)
    try:
        if slice_cfg:
            product = build_scalar_slice_product(state, quantity=quantity, lon_target=float(slice_cfg.get("longitude", 0.0)),
                                                 r_min=slice_cfg.get("r_min", 1.0), r_max=slice_cfg.get("r_max", 3.0),
                                                 n_r=slice_cfg.get("n_r", 100))
        else:
            product = build_scalar_shell_product(state, quantity=quantity, r_target=float(radius))
    finally:
        _release_state(state)

    return product, run_dir

def _longitude_order(lon_deg: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    lon_deg = np.asarray(lon_deg, dtype=float)

    if mode in {"minus180_180", "-180_180", "180"}:
        lon_plot = ((lon_deg + 180.0) % 360.0) - 180.0
        order = np.argsort(lon_plot)
        return lon_plot[order], order

    if mode in {"zero_360", "0_360", "native"}:
        return lon_deg, np.arange(lon_deg.size)

    raise ValueError(f"Unknown longitude_mode={mode}")

def _finite_percentile(vals: np.ndarray, pct: list[float], fallback=(0.0, 1.0)) -> tuple[float, float]:
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(fallback[0]), float(fallback[1])

    lo = float(np.nanpercentile(vals, float(pct[0])))
    hi = float(np.nanpercentile(vals, float(pct[1])))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))

    if lo == hi:
        hi = lo + 1.0

    return lo, hi

def _get_field_limits(
    panels: list[dict],
    gt_array: np.ndarray,
    plot_cfg: dict,
) -> tuple[float, float]:
    field_cfg = dict(plot_cfg.get("field_color", {}) or {})

    if field_cfg.get("vmin", None) is not None and field_cfg.get("vmax", None) is not None:
        return float(field_cfg["vmin"]), float(field_cfg["vmax"])

    pct = field_cfg.get("percentile", [1.0, 99.0])
    source = field_cfg.get("source", "gt")

    if source == "gt":
        vals = gt_array
    elif source == "all":
        vals = np.concatenate([p["field"].reshape(-1) for p in panels])
    else:
        raise ValueError("plot.field_color.source must be 'gt' or 'all'")

    return _finite_percentile(vals, pct)

def _get_residual_limit(
    residual_arrays: list[np.ndarray],
    plot_cfg: dict,
) -> float:
    resid_cfg = dict(plot_cfg.get("residual_color", {}) or {})

    if resid_cfg.get("vlim", None) is not None:
        return float(resid_cfg["vlim"])

    if not residual_arrays:
        return 1.0

    vals = np.concatenate([np.abs(x).reshape(-1) for x in residual_arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0

    pct = float(resid_cfg.get("percentile", 99.0))
    lim = float(np.nanpercentile(vals, pct))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(vals)) if vals.size else 1.0
    return max(lim, 1.0e-6)

def _format_axes(ax, show_xlabel: bool, show_ylabel: bool, plot_cfg: dict) -> None:
    if show_xlabel:
        ax.set_xlabel(plot_cfg.get("xlabel", "Longitude [deg]"))
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(plot_cfg.get("ylabel", "Latitude [deg]"))
    else:
        ax.set_ylabel("")

    tick_font_size = plot_cfg.get("tick_font_size", None)
    if tick_font_size is not None:
        ax.tick_params(labelsize=float(tick_font_size))

def _plot_shell_panel_grid(
    panels: list[dict],
    gt_array: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    quantity: str,
    radius: float,
    out_base: Path,
    plot_cfg: dict,
    residual_cfg: dict,
) -> dict:
    spec = get_scalar_field_spec(quantity)

    residual_enabled = bool(residual_cfg.get("enabled", True))
    n_cols = len(panels)
    n_rows = 2 if residual_enabled else 1

    figsize = plot_cfg.get("figsize", None)
    if figsize is None:
        figsize = [3.0 * n_cols + 1.0, 5.2 if residual_enabled else 2.8]

    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", font_size + 1))
    dpi = int(plot_cfg.get("dpi", 300))

    field_cmap = plot_cfg.get("field_cmap", spec.cmap)
    residual_cmap = plot_cfg.get("residual_cmap", "coolwarm")
    aspect = plot_cfg.get("aspect", "auto")

    vmin, vmax = _get_field_limits(panels, gt_array, plot_cfg)

    residual_arrays = [
        p["field"] - gt_array
        for p in panels
        if p.get("source") != "gt"
    ]
    dlim = _get_residual_limit(residual_arrays, plot_cfg)

    extent = [
        float(np.nanmin(lon_deg)),
        float(np.nanmax(lon_deg)),
        float(np.nanmin(lat_deg)),
        float(np.nanmax(lat_deg)),
    ]

    rc = {
        "font.size": font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "figure.titlesize": title_font_size + 1,
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

        im_field = None
        im_resid = None

        for j, panel in enumerate(panels):
            ax = axs[0, j]
            im_field = ax.imshow(
                panel["field"],
                origin="lower",
                extent=extent,
                cmap=field_cmap,
                vmin=vmin,
                vmax=vmax,
                aspect=aspect,
                interpolation=plot_cfg.get("interpolation", "nearest"),
            )
            ax.set_title(panel["label"])
            _format_axes(
                ax,
                show_xlabel=(not residual_enabled),
                show_ylabel=(j == 0),
                plot_cfg=plot_cfg,
            )

        if residual_enabled:
            for j, panel in enumerate(panels):
                ax = axs[1, j]

                if panel.get("source") == "gt" and not bool(residual_cfg.get("show_gt_zero", False)):
                    ax.axis("off")
                    continue

                resid = panel["field"] - gt_array
                im_resid = ax.imshow(
                    resid,
                    origin="lower",
                    extent=extent,
                    cmap=residual_cmap,
                    vmin=-dlim,
                    vmax=dlim,
                    aspect=aspect,
                    interpolation=plot_cfg.get("interpolation", "nearest"),
                )

                if panel.get("source") == "gt":
                    ax.set_title("GT residual")
                else:
                    ax.set_title(panel.get("residual_label", "Residual"))

                _format_axes(
                    ax,
                    show_xlabel=True,
                    show_ylabel=(j == 0),
                    plot_cfg=plot_cfg,
                )

        if plot_cfg.get("pack_ticks", False):
            for rr in range(n_rows):
                for cc in range(n_cols):
                    ax = axs[rr, cc]
                    if cc != 0:            ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
                    if rr != n_rows - 1:   ax.set_xticklabels([]); ax.tick_params(axis="x", length=0)

        fig_title = plot_cfg.get("title", None)
        if fig_title:
            fig.suptitle(str(fig_title))

        field_cbar = fig.colorbar(
            im_field,
            ax=axs[0, :].ravel().tolist(),
            fraction=float(plot_cfg.get("colorbar_fraction", 0.035)),
            pad=float(plot_cfg.get("colorbar_pad", 0.02)),
        )
        field_cbar.set_label(
            plot_cfg.get("field_color", {}).get("label", spec.log_label)
        )

        if residual_enabled and im_resid is not None:
            resid_cbar = fig.colorbar(
                im_resid,
                ax=axs[1, :].ravel().tolist(),
                fraction=float(plot_cfg.get("colorbar_fraction", 0.035)),
                pad=float(plot_cfg.get("colorbar_pad", 0.02)),
            )
            resid_cbar.set_label(
                plot_cfg.get("residual_color", {}).get(
                    "label",
                    rf"$\Delta$ {spec.log_label}",
                )
            )

        formats = plot_cfg.get("formats", ["png", "pdf"])
        out_paths = []
        for ext in formats:
            ext = str(ext).lstrip(".")
            path = out_base.with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))

        plt.close(fig)

    return {
        "out_paths": out_paths,
        "field_vmin": float(vmin),
        "field_vmax": float(vmax),
        "residual_vlim": float(dlim),
        "radius_actual": float(radius),
    }

def generate_shell_panel_figure(
    spec_path: Union[str, Path],
    device_override: str | None = None,
    output_dir_override: Union[str, Path, None] = None,
) -> dict:
    spec_path = Path(spec_path).resolve()
    spec_root = spec_path.parent

    raw = load_yaml(spec_path)
    cfg = raw.get("figure", raw)

    path_remap = cfg.get("path_remap", None)

    name = str(cfg.get("name", spec_path.stem))
    benchmark_dir = _resolve_path(cfg["benchmark_dir"], spec_root)

    if output_dir_override is not None:
        out_dir = _resolve_path(output_dir_override, Path.cwd())
    else:
        out_dir = _resolve_path(cfg.get("output_dir", benchmark_dir / "paper_figures"), spec_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    quantity = str(cfg.get("quantity", "ne"))
    radius = float(cfg.get("radius", 1.5))
    slice_cfg = dict(cfg.get("slice", {}) or {})               # {longitude, r_min, r_max, n_r} -> meridional slice mode

    device = device_override
    if device is None:
        device = cfg.get("device", None)

    default_metric = str(
        cfg.get(
            "selection_metric",
            f"final/{quantity}_shell_inner/shell_mean_abs_dlog_mean",
        )
    )
    default_policy = str(cfg.get("selection_policy", "median"))
    default_mode = str(cfg.get("select_mode", "min"))

    rows = collect_run_summaries(benchmark_dir)
    rows = [r for r in rows if r.get("status") == "completed"]
    if not rows:
        raise ValueError(f"No completed run summaries found in {benchmark_dir}")

    panel_cfgs = list(cfg.get("panels", []) or [])
    if not panel_cfgs:
        raise ValueError("figure.panels must contain at least one panel")

    products_by_idx: dict[int, Any] = {}
    selected_rows_by_idx: dict[int, dict] = {}
    run_dirs_by_idx: dict[int, Path] = {}

    # Load all non-GT products first. The first one supplies the GT shell.
    first_product = None
    for i, panel_cfg in enumerate(panel_cfgs):
        if str(panel_cfg.get("source", "pred")) == "gt":
            continue

        row = _select_representative_row(
            rows=rows,
            benchmark_dir=benchmark_dir,
            spec_root=spec_root,
            panel_cfg=panel_cfg,
            default_metric=default_metric,
            default_policy=default_policy,
            default_mode=default_mode,
        )
        product, run_dir = _load_shell_product_for_row(
            row=row,
            benchmark_dir=benchmark_dir,
            quantity=quantity,
            radius=radius,
            device_override=device,
            path_remap=path_remap,
            slice_cfg = slice_cfg,
        )

        products_by_idx[i] = product
        selected_rows_by_idx[i] = row
        run_dirs_by_idx[i] = run_dir

        if first_product is None:
            first_product = product

    if first_product is None:
        raise ValueError("At least one non-GT panel is required so the script can load the GT field")

    # longitude handling
    plot_cfg = dict(cfg.get("plot", {}) or {})
    if slice_cfg:                                                     # slice: x = r (monotonic, no reorder)
        x_plot = np.asarray(first_product.r_axis, dtype=float)
        lon_order = np.arange(x_plot.size)
        plot_cfg.setdefault("xlabel", r"$r\ [R_\odot]$")
        x_meta = float(slice_cfg.get("longitude", 0.0))
    else:                                                            # shell: x = longitude (reordered)
        lon_mode = str(plot_cfg.get("longitude_mode", "minus180_180"))
        x_plot, lon_order = _longitude_order(first_product.lon_deg, lon_mode)
        x_meta = float(first_product.r)
    lat_plot = np.asarray(first_product.lat_deg, dtype=float)
    def reorder(arr): return np.asarray(arr)[:, lon_order]           # identity for slice
    gt_array = reorder(first_product.gt_log)

    panels = []
    manifest_panels = []

    for i, panel_cfg in enumerate(panel_cfgs):
        source = str(panel_cfg.get("source", "pred"))
        label = str(panel_cfg.get("label", panel_cfg.get("experiment", source)))

        if source == "gt":
            field = gt_array
            row = None
            run_dir = None
        else:
            product = products_by_idx[i]
            field = reorder(product.pred_log)
            row = selected_rows_by_idx[i]
            run_dir = run_dirs_by_idx[i]

        panel = {
            "label": label,
            "source": source,
            "field": field,
            "residual_label": panel_cfg.get("residual_label", rf"{label} $-$ GT"),
        }
        panels.append(panel)

        manifest_panels.append({
            "label": label,
            "source": source,
            "experiment": panel_cfg.get("experiment", None),
            "run_name": None if row is None else row.get("run_name"),
            "run_dir": None if run_dir is None else str(run_dir),
            "seed": None if row is None else row.get("seed"),
            "selection_metric": None if row is None else default_metric,
            "selection_metric_value": None if row is None or row.get(default_metric) is None else float(row.get(default_metric)),
        })

    out_base = out_dir / name
    residual_cfg = dict(cfg.get("residual", {}) or {})
    plot_info = _plot_shell_panel_grid(
        panels=panels,
        gt_array=gt_array,
        lon_deg=x_plot,
        lat_deg=lat_plot,
        quantity=quantity,
        radius=x_meta,
        out_base=out_base,
        plot_cfg=plot_cfg,
        residual_cfg=residual_cfg,
    )

    # Save plotted arrays for reproducibility.
    npz_payload = {
        "lon_deg": x_plot,
        "lat_deg": lat_plot,
        "gt_log": gt_array,
    }
    for i, panel in enumerate(panels):
        key = _slug(panel["label"])
        npz_payload[f"field_{i}_{key}"] = panel["field"]
        if panel.get("source") != "gt":
            npz_payload[f"residual_{i}_{key}"] = panel["field"] - gt_array

    npz_path = out_dir / f"{name}_arrays.npz"
    np.savez_compressed(npz_path, **npz_payload)

    manifest = {
        "name": name,
        "spec_path": str(spec_path),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(out_dir),
        "quantity": quantity,
        "path_remap": path_remap,
        "radius_requested": float(radius),
        "radius_actual": x_meta,
        "device": device,
        "selection_metric": default_metric,
        "selection_policy": default_policy,
        "select_mode": default_mode,
        "panels": manifest_panels,
        "arrays_npz": str(npz_path),
        **plot_info,
    }

    save_json(manifest, out_dir / f"{name}_manifest.json")
    logger.info("Saved paper shell figure manifest: %s", out_dir / f"{name}_manifest.json")
    return manifest