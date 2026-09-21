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
from ..benchmark.condition_compare import (
    _build_state_for_analysis,
    _release_state,
    _get_comparison_observation_cube,
    _render_selected_views,
)
from .shell_panels import _resolve_path, _resolve_run_dir, _select_representative_row, _slug

logger = logging.getLogger("coroNeRF.paper.observation_denoising")

'''
generic script to compare observations between runs
but, script is made for showing denoising capabilities for observation noise
'''

def _apply_image_mask(arr: np.ndarray, mask: np.ndarray | None, mask_value=np.nan) -> np.ndarray:
    if mask is None:
        return arr
    out = np.array(arr, copy=True)
    out[~mask.astype(bool)] = mask_value
    return out

def _display_transform(arr: np.ndarray, cfg: dict, channel_index: int) -> np.ndarray:
    mode = str(cfg.get("mode", "asinh"))

    if mode == "linear":
        return arr

    if mode == "log_pos":
        eps = float(cfg.get("eps", 1.0e-12))
        return np.log10(np.maximum(arr, eps))

    if mode == "asinh":
        scale = cfg.get("scale", None)
        scales = cfg.get("scale_by_channel", None)

        if scale is None:
            if scales is None:
                raise ValueError("display.mode=asinh requires scale or scale_by_channel")
            scale = float(scales[int(channel_index)])
        else:
            scale = float(scale)

        return np.arcsinh(arr / scale)

    raise ValueError(f"Unknown display transform mode={mode}")

def _select_row(rows, benchmark_dir, spec_root, panel_cfg, default_metric, default_policy, default_mode):
    return _select_representative_row(
        rows=rows,
        benchmark_dir=benchmark_dir,
        spec_root=spec_root,
        panel_cfg=panel_cfg,
        default_metric=panel_cfg.get("selection_metric", default_metric),
        default_policy=panel_cfg.get("selection_policy", default_policy),
        default_mode=panel_cfg.get("select_mode", default_mode),
    )

def _load_panel_image(
    row: dict,
    benchmark_dir: Path,
    panel_cfg: dict,
    view_index: int,
    channel_index: int,
    device: str | None,
    path_remap: dict | None,
    chunk_size: int,
) -> tuple[np.ndarray, Path]:
    run_dir = _resolve_run_dir(row, benchmark_dir)
    if run_dir is None:
        raise FileNotFoundError(f"Could not resolve run_dir for row: {row}")

    source = str(panel_cfg.get("source", "train_obs"))

    state = _build_state_for_analysis(run_dir, device_override=device, path_remap=path_remap)
    try:
        if source in {"train_obs", "obs", "target"}:
            cube = _get_comparison_observation_cube(state, "train")
            img = cube[int(view_index), ..., int(channel_index)]

        elif source == "test_obs":
            cube = _get_comparison_observation_cube(state, "test")
            img = cube[int(view_index), ..., int(channel_index)]

        elif source in {"train_pred", "pred"}:
            cube = _render_selected_views(
                state,
                split="train",
                view_indices=[int(view_index)],
                chunk_size=int(chunk_size),
            )
            img = cube[0, ..., int(channel_index)]

        elif source == "test_pred":
            cube = _render_selected_views(
                state,
                split="test",
                view_indices=[int(view_index)],
                chunk_size=int(chunk_size),
            )
            img = cube[0, ..., int(channel_index)]

        else:
            raise ValueError(f"Unknown panel source={source}")

    finally:
        _release_state(state)

    return np.asarray(img, dtype=np.float32), run_dir

def generate_observation_denoising_figure(
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

    view_index = int(cfg.get("view_index", 0))
    channel_index = int(cfg.get("channel_index", 0))
    chunk_size = int(cfg.get("chunk_size", 8192))

    default_metric = str(cfg.get("selection_metric", "final/temp_shell_inner/shell_mean_abs_dlog_mean"))
    default_policy = str(cfg.get("selection_policy", "best"))
    default_mode = str(cfg.get("select_mode", "min"))

    rows = collect_run_summaries(benchmark_dir)
    rows = [r for r in rows if r.get("status") == "completed"]

    panels_cfg = list(cfg.get("panels", []) or [])
    if not panels_cfg:
        raise ValueError("figure.panels must be non-empty")

    images = []
    panel_manifest = []

    for panel_cfg in panels_cfg:
        row = _select_row(
            rows=rows,
            benchmark_dir=benchmark_dir,
            spec_root=spec_root,
            panel_cfg=panel_cfg,
            default_metric=default_metric,
            default_policy=default_policy,
            default_mode=default_mode,
        )

        img, run_dir = _load_panel_image(
            row=row,
            benchmark_dir=benchmark_dir,
            panel_cfg=panel_cfg,
            view_index=int(panel_cfg.get("view_index", view_index)),
            channel_index=int(panel_cfg.get("channel_index", channel_index)),
            device=device,
            path_remap=path_remap,
            chunk_size=chunk_size,
        )

        images.append(img)
        panel_manifest.append({
            "label": panel_cfg.get("label", row.get("experiment_name")),
            "source": panel_cfg.get("source", "train_obs"),
            "experiment": row.get("experiment_name"),
            "run_name": row.get("run_name"),
            "run_dir": str(run_dir),
            "seed": row.get("seed"),
            "where": panel_cfg.get("where", None),
        })

    # apply photosphere mask
    mask_cfg = dict(cfg.get("mask", {}) or {})
    mask_enabled = bool(mask_cfg.get("enabled", True))
    mask_source = str(mask_cfg.get("source", "first_panel"))

    mask = None
    if mask_enabled:
        if mask_source == "finite_first_panel":
            mask = np.isfinite(images[0])
        elif mask_source == "positive_first_panel":
            mask = images[0] > 0
        elif mask_source == "nonzero_first_panel":
            mask = images[0] != 0
        else:
            raise ValueError(
                f"Unknown mask.source={mask_source!r}; use "
                "finite_first_panel, positive_first_panel, or nonzero_first_panel"
            )

        images = [
            _apply_image_mask(img, mask, mask_value=np.nan)
            for img in images
        ]
    
    
    display_cfg = dict(cfg.get("display", {}) or {})
    disp_images = [
        _display_transform(img, display_cfg, channel_index=channel_index)
        for img in images
    ]

    finite_vals = np.concatenate([
        x[np.isfinite(x)].reshape(-1)
        for x in disp_images
        if np.any(np.isfinite(x))
    ])

    if display_cfg.get("vmin", None) is not None and display_cfg.get("vmax", None) is not None:
        vmin = float(display_cfg["vmin"])
        vmax = float(display_cfg["vmax"])
    else:
        pct = display_cfg.get("percentile", [1.0, 99.0])
        vmin = float(np.nanpercentile(finite_vals, float(pct[0])))
        vmax = float(np.nanpercentile(finite_vals, float(pct[1])))

    plot_cfg = dict(cfg.get("plot", {}) or {})
    figsize = plot_cfg.get("figsize", [10.0, 2.6])
    dpi = int(plot_cfg.get("dpi", 300))
    formats = plot_cfg.get("formats", ["png", "pdf"])
    font_size = float(plot_cfg.get("font_size", 8))
    title_font_size = float(plot_cfg.get("title_font_size", 9))
    tick_font_size = float(plot_cfg.get("tick_font_size", 7))

    cmap = plt.get_cmap(plot_cfg.get("cmap", "coolwarm")).copy()
    cmap.set_bad(plot_cfg.get("bad_color", "black"))

    rc = {
        "font.size": font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_font_size,
        "ytick.labelsize": tick_font_size,
    }

    with plt.rc_context(rc):
        fig, axs = plt.subplots(
            1,
            len(disp_images),
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )
        axs = axs[0]

        im = None
        for ax, panel, disp in zip(axs, panel_manifest, disp_images):
            im = ax.imshow(
                disp,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=plot_cfg.get("interpolation", "nearest"),
            )
            ax.set_title(str(panel["label"]))
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(
            im,
            ax=axs.ravel().tolist(),
            fraction=float(plot_cfg.get("colorbar_fraction", 0.035)),
            pad=float(plot_cfg.get("colorbar_pad", 0.02)),
        )
        cbar.set_label(display_cfg.get("label", display_cfg.get("mode", "display")))

        if plot_cfg.get("title", None):
            fig.suptitle(str(plot_cfg["title"]))

        out_paths = []
        for ext in formats:
            ext = str(ext).lstrip(".")
            path = (out_dir / name).with_suffix(f".{ext}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            out_paths.append(str(path))

        plt.close(fig)

    npz_payload = {}
    for i, (raw_img, disp_img, panel) in enumerate(zip(images, disp_images, panel_manifest)):
        key = f"{i}_{_slug(panel['label'])}"
        npz_payload[f"raw_{key}"] = raw_img
        npz_payload[f"display_{key}"] = disp_img

    npz_path = out_dir / f"{name}_arrays.npz"
    np.savez_compressed(npz_path, **npz_payload)

    manifest = {
        "name": name,
        "spec_path": str(spec_path),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(out_dir),
        "device": device,
        "view_index": view_index,
        "channel_index": channel_index,
        "display": display_cfg,
        "vmin": vmin,
        "vmax": vmax,
        "panels": panel_manifest,
        "arrays_npz": str(npz_path),
        "out_paths": out_paths,
    }
    save_json(manifest, out_dir / f"{name}_manifest.json")
    return manifest