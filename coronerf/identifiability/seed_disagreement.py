##########################################################
### Piece A: seed-disagreement field-uncertainty diagnostic
###
### Does cross-seed ensemble disagreement (a GT-free signal)
### track the true 3D field reconstruction error?
###
### For one benchmark CONDITION (an experiment + optional `where`
### filter, whose multiple seeds form the ensemble) this builds:
###   - per-quantity (ne, temp) shell panels: GT | ensemble mean |
###     |mean - GT| (true error) | cross-seed std (disagreement)
###   - one held-out image-space residual panel (ensemble render - clean)
###   - a quantitative scatter/calibration of disagreement vs true error
###
### Mirrors the paper-figure conventions in coronerf/util/figure.py.
##########################################################

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, save_json
from ..benchmark.condition_compare import (
    _build_state_for_analysis,
    _release_state,
    _render_selected_views,
    _get_comparison_observation_cube,
    _mask_image_cube,
    _choose_view_indices,
    _channel_labels,
    _resolve_run_dir,
)
from ..artifacts.scalar_fields import (
    build_scalar_shell_product, # fixed r, shell
    build_scalar_slice_product, # fixed lon, meridional slice
    get_scalar_field_spec,
    scalar_field_available,
)
from ..benchmark.select import _collect_condition_seed_rows
from ..util.figure import (
    _resolve_path, _slug, _savefig, _cmap_with_bad, _qty_tex, _reorder_cols,
    _correlations, _longitude_order, _finite_percentile, diagnostics_dir,
)

logger = logging.getLogger("coroNeRF.identifiability.seed_disagreement")

__all__ = ["generate_seed_disagreement_figure"]


# condition/seed selection now lives in ..benchmark.select (_collect_condition_seed_rows)


# ----------------------------------------------------------------------
# ensemble accumulation (one checkpoint load per seed)
# ----------------------------------------------------------------------

def _accumulate_ensemble(
    seed_rows: list[dict],
    benchmark_dir: Path,
    quantities: list[str],
    radii: list[float],
    image_cfg: dict,
    device: str | None,
    path_remap: dict | None,
    slice_cfg: dict | None = None,
) -> tuple[dict, dict, dict]:
    """
    Loop over seeds once. For each seed: collect per-(quantity, radius)
    predicted shell fields, optionally a fixed-longitude (lat, r) meridional
    slice per quantity, and (optionally) render held-out image views.
    Returns (field, slice_field, img). slice_field is {} when slice_cfg is None.
    """
    field: dict[str, dict[float, dict]] = {
        q: {float(r): {"gt": None, "lon": None, "lat": None, "preds": []} for r in radii}
        for q in quantities
    }
    # one meridional slice per quantity (fixed longitude), grid is (lat, r)
    slice_field: dict[str, dict] = (
        {q: {"gt": None, "lat": None, "r": None, "preds": []} for q in quantities}
        if slice_cfg else {}
    )
    img: dict[str, Any] = {
        "renders": [],
        "clean": None,
        "view_indices": None,
        "channel_labels": None,
    }

    image_enabled = bool(image_cfg.get("enabled", True))
    split = str(image_cfg.get("split", "test"))
    avail_quantities: list[str] | None = None

    for row in seed_rows:
        run_dir = _resolve_run_dir(row, benchmark_dir)
        if run_dir is None:
            raise FileNotFoundError(f"Could not resolve run dir for row={row}")

        state = _build_state_for_analysis(
            run_dir, device_override=device, path_remap=path_remap
        )
        try:
            avail = [q for q in quantities if scalar_field_available(state, q)]
            if avail_quantities is None:
                avail_quantities = avail

            # ---- field shells ----
            for q in avail:
                for r in radii:
                    product = build_scalar_shell_product(state, quantity=q, r_target=float(r))
                    slot = field[q][float(r)]
                    if slot["gt"] is None:
                        slot["gt"] = product.gt_log            # identical across seeds
                        slot["lon"] = product.lon_deg
                        slot["lat"] = product.lat_deg
                    slot["preds"].append(product.pred_log)

            # ---- fixed-longitude meridional slice (lat, r), one per quantity ----
            if slice_cfg:
                for q in avail:
                    sp = build_scalar_slice_product(
                        state, quantity=q,
                        lon_target=float(slice_cfg.get("longitude", 0.0)),
                        r_min=slice_cfg.get("r_min"), r_max=slice_cfg.get("r_max"),
                        n_r=slice_cfg.get("n_r"),
                    )
                    sl = slice_field[q]
                    if sl["gt"] is None:
                        sl["gt"] = sp.gt_log                   # identical across seeds
                        sl["lat"] = sp.lat_deg
                        sl["r"] = sp.r_axis
                    sl["preds"].append(sp.pred_log)

            # ---- held-out image residual ----
            if image_enabled:
                if split == "test":
                    v_total = int(state.test_data["imgs"].shape[0])
                    mask_2d = state.test_data["mask"]
                else:
                    v_total = int(state.train_data["imgs"].shape[0])
                    mask_2d = state.train_data["mask"]

                if int(image_cfg.get("num_views", 1)) == 1 and image_cfg.get("view_index") is not None:
                    view_indices = [int(image_cfg["view_index"])]
                else:
                    view_indices = _choose_view_indices(
                        v_total,
                        int(image_cfg.get("num_views", 1)),
                        str(image_cfg.get("view_strategy", "evenly_spaced")),
                    )

                render = _render_selected_views(
                    state, split, view_indices,
                    chunk_size=int(image_cfg.get("chunk_size", 8192)),
                )  # (len(view_indices), H, W, C), coronal-masked with NaN
                img["renders"].append(render)

                if img["clean"] is None:
                    clean_full = _get_comparison_observation_cube(state, split)  # (V,H,W,C)
                    clean = np.asarray(clean_full)[view_indices]
                    clean = _mask_image_cube(clean, mask_2d, fill_value=np.nan)
                    img["clean"] = clean
                    img["view_indices"] = view_indices
                    img["channel_labels"] = _channel_labels(state)
        finally:
            _release_state(state)

    # keep only quantities that were actually available
    avail_quantities = avail_quantities or []
    field = {q: field[q] for q in quantities if q in avail_quantities}
    if slice_cfg:
        slice_field = {q: slice_field[q] for q in quantities if q in avail_quantities}
    if not field:
        raise ValueError(
            f"None of the requested quantities {quantities} are available for this run "
            f"(reconstruction_target may be 'ne' only)."
        )
    return field, slice_field, img


# ----------------------------------------------------------------------
# reductions
# ----------------------------------------------------------------------

def _spread(preds: np.ndarray, kind: str) -> np.ndarray:
    """Cross-seed dispersion statistic over axis 0. preds: (S, P, L)."""
    s = preds.shape[0]
    if s < 2:
        return np.zeros(preds.shape[1:], dtype=float)
    if kind == "std":
        return np.nanstd(preds, axis=0, ddof=1)
    if kind == "mad":  # median absolute deviation (scaled to ~std)
        med = np.nanmedian(preds, axis=0)
        return 1.4826 * np.nanmedian(np.abs(preds - med[None]), axis=0)
    if kind == "iqr":
        q75, q25 = np.nanpercentile(preds, [75, 25], axis=0)
        return q75 - q25
    raise ValueError(f"Unknown disagreement kind={kind}")

def _reduce_field(field: dict, error_mode: str, disagreement: str) -> dict:
    out: dict[str, dict[float, dict]] = {}
    for q, by_r in field.items():
        out[q] = {}
        for r, slot in by_r.items():
            if not slot["preds"]:
                continue
            preds = np.stack(slot["preds"], axis=0).astype(float)  # (S,P,L)
            gt = np.asarray(slot["gt"], dtype=float)               # (P,L)

            ens_mean = np.nanmean(preds, axis=0)
            disag = _spread(preds, disagreement)

            bias = np.abs(ens_mean - gt)                                  # |E[pred] - GT| : error of the AVERAGED field
            rmse = np.sqrt(np.nanmean((preds - gt[None]) ** 2, axis=0))   # sqrt(bias^2 + variance) : total single-seed error
            if error_mode in ("ensemble_mean_abs", "bias"):
                true_err = bias
            elif error_mode == "mean_seed_abs":
                true_err = np.nanmean(np.abs(preds - gt[None]), axis=0)
            elif error_mode == "rmse":
                true_err = rmse
            else:
                raise ValueError(f"Unknown error_mode={error_mode} (use: bias | rmse | mean_seed_abs)")

            out[q][float(r)] = {
                "gt": gt,
                "ens_mean": ens_mean,
                "true_err": true_err,
                "bias": bias,
                "rmse": rmse,
                "disag": disag,
                "lon": np.asarray(slot["lon"], dtype=float),
                "lat": np.asarray(slot["lat"], dtype=float),
                "n_seeds": int(preds.shape[0]),
            }
    return out

def _reduce_slice(slice_field: dict, error_mode: str, disagreement: str) -> dict:
    """Slice analog of _reduce_field. Each quantity has ONE (lat, r) grid (fixed longitude)."""
    out: dict[str, dict] = {}
    for q, slot in slice_field.items():
        if not slot["preds"]:
            continue
        preds = np.stack(slot["preds"], axis=0).astype(float)  # (S, P, R)
        gt = np.asarray(slot["gt"], dtype=float)               # (P, R)

        ens_mean = np.nanmean(preds, axis=0)
        disag = _spread(preds, disagreement)

        bias = np.abs(ens_mean - gt)
        rmse = np.sqrt(np.nanmean((preds - gt[None]) ** 2, axis=0))
        if error_mode in ("ensemble_mean_abs", "bias"):
            true_err = bias
        elif error_mode == "mean_seed_abs":
            true_err = np.nanmean(np.abs(preds - gt[None]), axis=0)
        elif error_mode == "rmse":
            true_err = rmse
        else:
            raise ValueError(f"Unknown error_mode={error_mode} (use: bias | rmse | mean_seed_abs)")

        out[q] = {
            "gt": gt,
            "ens_mean": ens_mean,
            "true_err": true_err,
            "bias": bias,
            "rmse": rmse,
            "disag": disag,
            "lat": np.asarray(slot["lat"], dtype=float),
            "r": np.asarray(slot["r"], dtype=float),
            "n_seeds": int(preds.shape[0]),
        }
    return out

def _reduce_image(img: dict, residual_mode: str) -> dict | None:
    if not img["renders"]:
        return None
    renders = np.stack(img["renders"], axis=0).astype(float)  # (S, V, H, W, C)
    ens_render = np.nanmean(renders, axis=0)                   # (V, H, W, C)
    resid = ens_render - np.asarray(img["clean"], dtype=float)
    if residual_mode == "abs":
        resid = np.abs(resid)
    elif residual_mode != "signed":
        raise ValueError(f"Unknown residual_mode={residual_mode}")
    return {
        "residual": resid,
        "ens_render": ens_render,
        "clean": np.asarray(img["clean"], dtype=float),
        "view_indices": img["view_indices"],
        "channel_labels": img["channel_labels"],
    }

# ----------------------------------------------------------------------
# small plotting helpers
# ----------------------------------------------------------------------

def _nearest_radius_key(by_r: dict, r_target: float) -> float:
    keys = list(by_r.keys())
    return min(keys, key=lambda k: abs(float(k) - float(r_target)))

# (_reorder_cols, _cmap_with_bad, _qty_tex now imported from ..util.figure)

def _upper_limit(arr: np.ndarray, pct: float, vmin: float = 0.0) -> tuple[float, float]:
    vals = np.asarray(arr)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(vmin), float(vmin) + 1.0
    hi = float(np.nanpercentile(np.abs(vals), float(pct)))
    if not np.isfinite(hi) or hi <= vmin:
        hi = float(np.nanmax(np.abs(vals))) if vals.size else vmin + 1.0
    return float(vmin), max(hi, vmin + 1e-12)


def _sym_limit(arr: np.ndarray, pct: float) -> float:
    vals = np.asarray(arr)
    vals = np.abs(vals[np.isfinite(vals)])
    if vals.size == 0:
        return 1.0
    lim = float(np.nanpercentile(vals, float(pct)))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(vals)) if vals.size else 1.0
    return max(lim, 1e-12)


def _apply_display(arr: np.ndarray, display_cfg: dict, channel_index: int) -> np.ndarray:
    mode = str(display_cfg.get("mode", "asinh"))
    sbc = display_cfg.get("scale_by_channel", None)
    s = 1.0
    if sbc is not None and 0 <= channel_index < len(sbc):
        s = float(sbc[channel_index])
    s = s if s > 0 else 1.0

    if mode == "linear":
        return arr
    if mode == "asinh":
        return np.arcsinh(arr / s)
    if mode == "log_pos":
        eps = float(display_cfg.get("eps", 1e-12))
        return np.log10(np.maximum(arr, eps))
    raise ValueError(f"Unknown display mode={mode}")


# (_savefig now imported from ..util.figure)


# ----------------------------------------------------------------------
# main figure: 2x4 field grid (+ big image residual panel)
# ----------------------------------------------------------------------

def _plot_field_and_image(
    reduced: dict,
    image_red: dict | None,
    primary_radius: float,
    plot_cfg: dict,
    image_cfg: dict,
    out_base: Path,
) -> dict:
    formats = list(plot_cfg.get("formats", ["png", "pdf"]))
    dpi = int(plot_cfg.get("dpi", 300))
    bad_color = str(plot_cfg.get("bad_color", "black"))
    show_cbars = bool(plot_cfg.get("show_colorbars", True))

    quantities = list(reduced.keys())
    n_rows = len(quantities)

    layout = str(image_cfg.get("layout", "big_plus_grid"))
    have_image = (image_red is not None) and (layout in ("big_plus_grid", "big_only"))

    col_labels = list(plot_cfg.get(
        "col_labels", ["Ground truth", "Ensemble mean", "|mean - GT|", "Cross-seed std"]
    ))
    row_labels = dict(plot_cfg.get("row_labels", {}))

    field_cmap_cfg = dict(plot_cfg.get("field_cmap", {}) or {})
    error_cmap = _cmap_with_bad(plot_cfg.get("error_cmap", "magma"), bad_color)
    disag_cmap = _cmap_with_bad(plot_cfg.get("disagreement_cmap", "cividis"), bad_color)

    field_color = dict(plot_cfg.get("field_color", {}) or {})
    error_color = dict(plot_cfg.get("error_color", {}) or {})
    disag_color = dict(plot_cfg.get("disagreement_color", {}) or {})
    lon_mode = str(plot_cfg.get("longitude_mode", "zero_360"))
    aspect = plot_cfg.get("aspect", "auto")
    interp = plot_cfg.get("interpolation", "nearest")

    rc = {
        "font.size": float(plot_cfg.get("font_size", 8)),
        "axes.titlesize": float(plot_cfg.get("title_font_size", 9)),
        "axes.labelsize": float(plot_cfg.get("font_size", 8)),
        "xtick.labelsize": float(plot_cfg.get("tick_font_size", 7)),
        "ytick.labelsize": float(plot_cfg.get("tick_font_size", 7)),
    }

    n_field_cols = 4
    n_total_cols = n_field_cols + (1 if have_image else 0)
    width_ratios = [1.0] * n_field_cols + ([1.45] if have_image else [])
    figsize = plot_cfg.get("figsize", [4.0 * n_total_cols, 3.0 * n_rows])

    panel_info: dict[str, Any] = {"radii_used": {}}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(n_rows, n_total_cols, width_ratios=width_ratios)

        for ri, q in enumerate(quantities):
            spec = get_scalar_field_spec(q)
            r_key = _nearest_radius_key(reduced[q], primary_radius)
            panel_info["radii_used"][q] = float(r_key)
            d = reduced[q][r_key]

            lon_plot, order = _longitude_order(d["lon"], lon_mode)
            lat = np.asarray(d["lat"], dtype=float)
            extent = [float(np.nanmin(lon_plot)), float(np.nanmax(lon_plot)),
                      float(np.nanmin(lat)), float(np.nanmax(lat))]

            gt = _reorder_cols(d["gt"], order)
            mean = _reorder_cols(d["ens_mean"], order)
            err = _reorder_cols(d["true_err"], order)
            disag = _reorder_cols(d["disag"], order)

            # --- correlation at this shell, annotated on the disagreement panel ---
            _xc = disag.reshape(-1); _yc = err.reshape(-1)
            _ok = np.isfinite(_xc) & np.isfinite(_yc)
            _cr = _correlations(_xc[_ok], _yc[_ok])
            _sp, _pe = _cr["spearman"], _cr["pearson"]
            _corr_txt = "$\\rho_{\\epsilon}^{(c)}$=" + ("n/a" if _sp is None else f"{_sp:.2f}")
            #if _pe is not None:
            #    _corr_txt += f", r={_pe:.2f}"

            fcmap = _cmap_with_bad(field_cmap_cfg.get(q, spec.cmap), bad_color)

            # field limits (shared by GT + mean)
            if field_color.get("source", "gt") == "all":
                fvals = np.concatenate([gt.reshape(-1), mean.reshape(-1)])
            else:
                fvals = gt.reshape(-1)
            vmin, vmax = _finite_percentile(fvals, field_color.get("percentile", [1.0, 99.0]))
            if field_color.get("vmin") is not None:
                vmin = float(field_color["vmin"])
            if field_color.get("vmax") is not None:
                vmax = float(field_color["vmax"])

            evmin, evmax = _upper_limit(err, float(error_color.get("percentile", 99.0)),
                                        float(error_color.get("vmin", 0.0)))
            svmin, svmax = _upper_limit(disag, float(disag_color.get("percentile", 99.0)),
                                        float(disag_color.get("vmin", 0.0)))

            arrays = [gt, mean, err, disag]
            cmaps = [fcmap, fcmap, error_cmap, disag_cmap]
            lims = [(vmin, vmax), (vmin, vmax), (evmin, evmax), (svmin, svmax)]

            ims = []
            row_axes = []
            for ci in range(n_field_cols):
                ax = fig.add_subplot(gs[ri, ci])
                row_axes.append(ax)
                im = ax.imshow(
                    arrays[ci], origin="lower", extent=extent, cmap=cmaps[ci],
                    vmin=lims[ci][0], vmax=lims[ci][1], aspect=aspect, interpolation=interp,
                )
                ims.append(im)

                if ci == 3:  # disagreement panel
                    ax.text(0.03, 0.97, _corr_txt, transform=ax.transAxes, va="top", ha="left",
                            fontsize=8, color="white",
                            bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="none"))

                if ri == 0 and ci < len(col_labels):
                    ax.set_title(col_labels[ci])
                if ci == 0:
                    ax.set_ylabel(row_labels.get(q, _qty_tex(q, "field")))
                ax.set_xticks([])
                ax.set_yticks([])

            if show_cbars:
                # attach to THIS row's axes explicitly (fig.axes shifts as colorbars are added)
                fig.colorbar(ims[1], ax=row_axes[1], fraction=0.046, pad=0.02,
                             label=field_color.get("label", _qty_tex(q, "field")))
                fig.colorbar(ims[2], ax=row_axes[2], fraction=0.046, pad=0.02,
                             label=error_color.get("label", _qty_tex(q, "err")))
                fig.colorbar(ims[3], ax=row_axes[3], fraction=0.046, pad=0.02,
                             label=disag_color.get("label", _qty_tex(q, "std")))

        # big held-out image residual panel
        if have_image:
            ax_img = fig.add_subplot(gs[:, n_field_cols])
            ch = int(image_cfg.get("big_channel_index", 0))
            display_cfg = dict(image_cfg.get("display", {}) or {})
            resid = image_red["residual"][0, ..., ch]  # first selected view
            disp = _apply_display(resid, display_cfg, ch)

            resid_mode = str(image_cfg.get("residual_mode", "signed"))
            img_cmap = _cmap_with_bad(plot_cfg.get("image_resid_cmap", "coolwarm"), bad_color)
            if resid_mode == "abs":
                ivmin, ivmax = _upper_limit(disp, float(display_cfg.get("percentile", [1.0, 99.0])[-1]
                                                        if isinstance(display_cfg.get("percentile"), (list, tuple))
                                                        else display_cfg.get("percentile", 99.0)), 0.0)
            else:
                lim = _sym_limit(disp, float(display_cfg.get("percentile", [1.0, 99.0])[-1]
                                             if isinstance(display_cfg.get("percentile"), (list, tuple))
                                             else display_cfg.get("percentile", 99.0)))
                ivmin, ivmax = -lim, lim

            im = ax_img.imshow(disp, cmap=img_cmap, vmin=ivmin, vmax=ivmax, interpolation=interp)
            labels = image_red["channel_labels"] or []
            ch_label = labels[ch] if ch < len(labels) else f"ch {ch}"
            ax_img.set_title(f"Held-out image residual\n{ch_label}")
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if show_cbars:
                fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.02,
                             label=display_cfg.get("label", "render - clean"))

        title = plot_cfg.get("title", None)
        if title:
            fig.suptitle(str(title))

        out_paths = _savefig(fig, out_base, formats, dpi)
        plt.close(fig)

    panel_info["out_paths"] = out_paths
    return panel_info


# ----------------------------------------------------------------------
# appendix: per-channel held-out image residual grid
# ----------------------------------------------------------------------

def _plot_image_channel_grid(
    image_red: dict,
    image_cfg: dict,
    plot_cfg: dict,
    appendix_cfg: dict,
    out_base: Path,
) -> dict:
    formats = list(plot_cfg.get("formats", ["png", "pdf"]))
    dpi = int(plot_cfg.get("dpi", 300))
    bad_color = str(plot_cfg.get("bad_color", "black"))
    display_cfg = dict(image_cfg.get("display", {}) or {})
    resid_mode = str(image_cfg.get("residual_mode", "signed"))
    img_cmap = _cmap_with_bad(plot_cfg.get("image_resid_cmap", "coolwarm"), bad_color)

    resid = image_red["residual"][0]  # (H, W, C), first selected view
    n_ch = resid.shape[-1]
    labels = image_red["channel_labels"] or [f"ch {i}" for i in range(n_ch)]

    grid_shape = appendix_cfg.get("grid_shape", "auto")
    if grid_shape == "auto":
        ncol = int(np.ceil(np.sqrt(n_ch)))
        nrow = int(np.ceil(n_ch / ncol))
    else:
        nrow, ncol = int(grid_shape[0]), int(grid_shape[1])

    disp_all = [_apply_display(resid[..., c], display_cfg, c) for c in range(n_ch)]
    pct = display_cfg.get("percentile", 99.0)
    pct = pct[-1] if isinstance(pct, (list, tuple)) else pct
    if resid_mode == "abs":
        vmin, vmax = 0.0, max(_sym_limit(np.stack(disp_all), float(pct)), 1e-12)
    else:
        lim = _sym_limit(np.stack(disp_all), float(pct))
        vmin, vmax = -lim, lim

    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False,
                            layout="constrained")
    im = None
    for c in range(nrow * ncol):
        ax = axs[c // ncol, c % ncol]
        if c < n_ch:
            im = ax.imshow(disp_all[c], cmap=img_cmap, vmin=vmin, vmax=vmax,
                           interpolation=plot_cfg.get("interpolation", "nearest"))
            ax.set_title(labels[c])
        ax.set_xticks([])
        ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02,
                     label=display_cfg.get("label", "render - clean"))

    out_paths = _savefig(fig, out_base, formats, dpi)
    plt.close(fig)
    return {"out_paths": out_paths, "n_channels": int(n_ch)}


# ----------------------------------------------------------------------
# the quantitative test: disagreement vs true error
# ----------------------------------------------------------------------

# (_correlations now imported from ..util.figure)


def _plot_diagnostic(reduced: dict, diag_cfg: dict, plot_cfg: dict, out_base: Path) -> dict:
    formats = list(plot_cfg.get("formats", ["png", "pdf"]))
    dpi = int(plot_cfg.get("dpi", 300))
    pool = str(diag_cfg.get("pool", "per_quantity"))
    n_bins = int(diag_cfg.get("calibration_bins", 20))
    alpha = float(diag_cfg.get("scatter_alpha", 0.05))
    max_pts = int(diag_cfg.get("scatter_max_points", 200000))
    rng = np.random.default_rng(0)

    def _pool(qs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for q in qs:
            for _, d in reduced[q].items():
                x = np.asarray(d["disag"]).reshape(-1)
                y = np.asarray(d["true_err"]).reshape(-1)
                ok = np.isfinite(x) & np.isfinite(y)
                xs.append(x[ok]); ys.append(y[ok])
        if not xs:
            return np.array([]), np.array([])
        return np.concatenate(xs), np.concatenate(ys)

    groups = {q: [q] for q in reduced} if pool == "per_quantity" else {"combined": list(reduced.keys())}

    stats: dict[str, dict] = {}
    n_panels = len(groups)
    fig, axs = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.4), squeeze=False)
    axs = axs[0]

    for ax, (gname, qs) in zip(axs, groups.items()):
        x, y = _pool(qs)
        corr = _correlations(x, y)
        stats[gname] = corr

        if x.size > max_pts:
            idx = rng.choice(x.size, size=max_pts, replace=False)
            xp, yp = x[idx], y[idx]
        else:
            xp, yp = x, y

        ax.scatter(xp, yp, s=2, alpha=alpha, rasterized=True)

        # calibration: mean true error per disagreement quantile-bin
        if x.size >= n_bins * 5:
            edges = np.nanpercentile(x, np.linspace(0, 100, n_bins + 1))
            edges = np.unique(edges)
            cx, cy = [], []
            for i in range(len(edges) - 1):
                m = (x >= edges[i]) & (x <= edges[i + 1])
                if np.any(m):
                    cx.append(float(np.mean(x[m]))); cy.append(float(np.mean(y[m])))
            ax.plot(cx, cy, "-o", color="k", ms=3, lw=1.2, label="calibration")
            lim = max(float(np.nanmax(x)) if x.size else 1.0, float(np.nanmax(y)) if y.size else 1.0)
            ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
            ax.legend(loc="best", fontsize=7)

        sp = corr["spearman"]
        pe = corr["pearson"]
        ax.set_title(f"{gname}  |  Spearman={sp:.3f}  Pearson={pe:.3f}"
                     if sp is not None else gname)
        ax.set_xlabel("cross-seed disagreement (dex)")
        ax.set_ylabel("true field error |Δlog| (dex)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(diag_cfg.get("title", "Seed disagreement vs true field error"))
    fig.tight_layout()
    out_paths = _savefig(fig, out_base, formats, dpi)
    plt.close(fig)
    return {"out_paths": out_paths, "correlations": stats}


# ----------------------------------------------------------------------
# orchestrator (public entry point)
# ----------------------------------------------------------------------

def _seed_disagreement_core(*, benchmark_dir, out_dir, device, path_remap, cond_cfg,
                            fields_cfg, image_cfg, diag_cfg, plot_cfg, appendix_cfg,
                            save_hero=True, rows_summ=None) -> dict:
    """Accumulate one ensemble condition -> reduce -> save arrays.npz (+ optional hero) -> manifest.
    Writes FIXED filenames into out_dir (one folder per condition)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    quantities = [str(q) for q in fields_cfg.get("quantities", ["ne", "temp"])]
    radii = [float(r) for r in fields_cfg.get("radii", [1.5])]
    primary_radius = float(fields_cfg.get("primary_radius", radii[0]))
    radii_all = sorted(set(radii) | {primary_radius})

    rows = rows_summ if rows_summ is not None else collect_run_summaries(benchmark_dir)
    seed_rows = _collect_condition_seed_rows(rows, cond_cfg)


    slice_cfg = dict(fields_cfg.get("slice", {}) or {})
    field, slice_field, img = _accumulate_ensemble(
        seed_rows=seed_rows, benchmark_dir=benchmark_dir,
        quantities=quantities, radii=radii_all,
        image_cfg=image_cfg, device=device, path_remap=path_remap,
        slice_cfg=(slice_cfg or None))
    err_mode = str(fields_cfg.get("error_mode", "ensemble_mean_abs"))
    disagree = str(fields_cfg.get("disagreement", "std"))
    reduced = _reduce_field(field, error_mode=err_mode, disagreement=disagree)
    reduced_slice = _reduce_slice(slice_field, error_mode=err_mode, disagreement=disagree) if slice_cfg else {}

    
    image_red = (_reduce_image(img, str(image_cfg.get("residual_mode", "signed")))
                 if image_cfg.get("enabled", False) else None)

    fig_info = (_plot_field_and_image(reduced, image_red, primary_radius, plot_cfg, image_cfg, out_dir / "hero")
                if save_hero else {})
    diag_info = (_plot_diagnostic(reduced, diag_cfg, plot_cfg, out_dir / "diagnostic")
                 if diag_cfg.get("enabled", False) else {})
    appendix_info = (_plot_image_channel_grid(image_red, image_cfg, plot_cfg, appendix_cfg, out_dir / "image_channels")
                     if (image_red is not None and appendix_cfg.get("per_channel_image_grid", True)) else {})

    # create and save payload
    npz_payload: dict[str, np.ndarray] = {}
    for q in reduced:
        r_key = _nearest_radius_key(reduced[q], primary_radius); d = reduced[q][r_key]; sl = _slug(q)
        npz_payload[f"{sl}_lon_deg"] = d["lon"]; npz_payload[f"{sl}_lat_deg"] = d["lat"]
        npz_payload[f"{sl}_gt"] = d["gt"]; npz_payload[f"{sl}_ens_mean"] = d["ens_mean"]
        npz_payload[f"{sl}_true_err"] = d["true_err"]; npz_payload[f"{sl}_disagreement"] = d["disag"]
    for q in reduced_slice:
        d = reduced_slice[q]; sl = _slug(q)
        npz_payload[f"{sl}_slice_lat_deg"] = d["lat"]; npz_payload[f"{sl}_slice_r_axis"] = d["r"]
        npz_payload[f"{sl}_slice_gt"] = d["gt"]; npz_payload[f"{sl}_slice_ens_mean"] = d["ens_mean"]
        npz_payload[f"{sl}_slice_true_err"] = d["true_err"]; npz_payload[f"{sl}_slice_disagreement"] = d["disag"]
    if reduced_slice:
        npz_payload["slice_longitude"] = np.asarray(float(slice_cfg.get("longitude", 0.0)))
    if image_red is not None:
        npz_payload["image_residual"] = image_red["residual"]
    npz_path = out_dir / "arrays.npz"
    np.savez_compressed(npz_path, **npz_payload)

    manifest = {"benchmark_dir": str(benchmark_dir), "output_dir": str(out_dir), "device": device,
                "path_remap": path_remap, "condition": cond_cfg, "n_seeds": len(seed_rows),
                "seed_run_names": [r.get("run_name") for r in seed_rows], "seeds": [r.get("seed") for r in seed_rows],
                "quantities": list(reduced.keys()), "radii": radii_all, "primary_radius": primary_radius,
                "slice": (slice_cfg or None), "slice_quantities": list(reduced_slice.keys()),
                "error_mode": fields_cfg.get("error_mode", "ensemble_mean_abs"),
                "disagreement": fields_cfg.get("disagreement", "std"),
                "figure": fig_info, "diagnostic": diag_info, "appendix": appendix_info,
                "arrays_npz": str(npz_path)}
    save_json(manifest, out_dir / "manifest.json")
    logger.info("seed-disagreement: %s (n_seeds=%d) -> %s", cond_cfg.get("experiment", "?"), len(seed_rows), out_dir)
    return manifest

def generate_seed_disagreement_figure(spec_path, device_override=None, output_dir_override=None) -> dict:
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    raw = load_yaml(spec_path); cfg = raw.get("figure", raw)
    name = str(cfg.get("name", spec_path.stem))
    benchmark_dir = _resolve_path(cfg["benchmark_dir"], spec_root)
    device = device_override if device_override is not None else cfg.get("device")
    sd_root = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg.get("output_dir", diagnostics_dir(benchmark_dir, "seed_disagreement")), spec_root))
    return _seed_disagreement_core(
        benchmark_dir=benchmark_dir, out_dir=sd_root / name, device=device,
        path_remap=cfg.get("path_remap"), cond_cfg=dict(cfg.get("condition", {}) or {}),
        fields_cfg=dict(cfg.get("fields", {}) or {}), image_cfg=dict(cfg.get("image_residual", {}) or {}),
        diag_cfg=dict(cfg.get("diagnostic", {}) or {}), plot_cfg=dict(cfg.get("plot", {}) or {}),
        appendix_cfg=dict(cfg.get("appendix", {}) or {}), save_hero=True)

def generate_seed_disagreement_batch(spec_path, device_override=None, output_dir_override=None) -> dict:
    """Loop over conditions (experiment families) in ONE benchmark, generating the sigma_ens arrays
    for each into <seed_disagreement>/<id>/. Same accumulation as `hero`, batched + reorganized."""
    spec_path = Path(spec_path).resolve(); spec_root = spec_path.parent
    cfg = load_yaml(spec_path).get("seed_disagreement_batch", {})
    name = str(cfg.get("name", spec_path.stem))
    benchmark_dir = _resolve_path(cfg["benchmark_dir"], spec_root)
    device = device_override if device_override is not None else cfg.get("device")
    path_remap = cfg.get("path_remap")
    sd_root = (_resolve_path(output_dir_override, Path.cwd()) if output_dir_override
               else _resolve_path(cfg.get("output_dir", diagnostics_dir(benchmark_dir, "seed_disagreement")), spec_root))
    sd_root.mkdir(parents=True, exist_ok=True)

    # shared defaults (batch = npz + cheap hero; no image residual / scatter by default)
    fields_cfg = dict(cfg.get("fields", {}) or {})
    image_cfg = dict(cfg.get("image_residual", {"enabled": False}) or {})
    diag_cfg = dict(cfg.get("diagnostic", {"enabled": False}) or {})
    plot_cfg = dict(cfg.get("plot", {}) or {})
    appendix_cfg = dict(cfg.get("appendix", {}) or {})
    save_hero = bool(cfg.get("save_hero", True))

    rows_summ = collect_run_summaries(benchmark_dir)            # scan the benchmark ONCE
    batch = {"name": name, "benchmark_dir": str(benchmark_dir), "output_dir": str(sd_root), "conditions": []}
    _SEL = ("experiment", "where", "filters", "run_names", "run_dirs", "min_seeds", "max_seeds")
    for cond in cfg.get("conditions", []):
        cid = str(cond["id"])
        cond_cfg = {k: cond[k] for k in _SEL if k in cond}
        try:
            man = _seed_disagreement_core(
                benchmark_dir=benchmark_dir, out_dir=sd_root / cid, device=device, path_remap=path_remap,
                cond_cfg=cond_cfg, fields_cfg=fields_cfg, image_cfg=image_cfg, diag_cfg=diag_cfg,
                plot_cfg=plot_cfg, appendix_cfg=appendix_cfg, save_hero=save_hero, rows_summ=rows_summ)
            batch["conditions"].append({"id": cid, "status": "ok", "n_seeds": man["n_seeds"],
                                        "arrays_npz": man["arrays_npz"], "dir": str(sd_root / cid)})
        except Exception as e:
            logger.warning("batch: condition '%s' FAILED: %s", cid, e)
            batch["conditions"].append({"id": cid, "status": "failed", "error": str(e)})
    save_json(batch, sd_root / "batch_manifest.json")
    logger.info("seed-disagreement batch '%s' -> %s (%d conditions)", name, sd_root, len(batch["conditions"]))
    return batch


# def generate_seed_disagreement_figure(
#     spec_path: Union[str, Path],
#     device_override: str | None = None,
#     output_dir_override: Union[str, Path, None] = None,
# ) -> dict:
#     spec_path = Path(spec_path).resolve()
#     spec_root = spec_path.parent

#     raw = load_yaml(spec_path)
#     cfg = raw.get("figure", raw)

#     path_remap = cfg.get("path_remap", None)
#     name = str(cfg.get("name", spec_path.stem))
#     benchmark_dir = _resolve_path(cfg["benchmark_dir"], spec_root)

#     if output_dir_override is not None:
#         out_dir = _resolve_path(output_dir_override, Path.cwd())
#     else:
#         out_dir = _resolve_path(cfg.get("output_dir", diagnostics_dir(benchmark_dir, "seed_disagreement")), spec_root)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     device = device_override if device_override is not None else cfg.get("device", None)

#     cond_cfg = dict(cfg.get("condition", {}) or {})
#     fields_cfg = dict(cfg.get("fields", {}) or {})
#     image_cfg = dict(cfg.get("image_residual", {}) or {})
#     diag_cfg = dict(cfg.get("diagnostic", {}) or {})
#     plot_cfg = dict(cfg.get("plot", {}) or {})
#     appendix_cfg = dict(cfg.get("appendix", {}) or {})

#     quantities = [str(q) for q in fields_cfg.get("quantities", ["ne", "temp"])]
#     radii = [float(r) for r in fields_cfg.get("radii", [1.5])]
#     primary_radius = float(fields_cfg.get("primary_radius", radii[0]))
#     radii_all = sorted(set([float(r) for r in radii]) | {primary_radius})

#     rows = collect_run_summaries(benchmark_dir)
#     seed_rows = _collect_condition_seed_rows(rows, cond_cfg)

#     field, img = _accumulate_ensemble(
#         seed_rows=seed_rows,
#         benchmark_dir=benchmark_dir,
#         quantities=quantities,
#         radii=radii_all,
#         image_cfg=image_cfg,
#         device=device,
#         path_remap=path_remap,
#     )

#     reduced = _reduce_field(
#         field,
#         error_mode=str(fields_cfg.get("error_mode", "ensemble_mean_abs")),
#         disagreement=str(fields_cfg.get("disagreement", "std")),
#     )
#     image_red = _reduce_image(img, str(image_cfg.get("residual_mode", "signed"))) \
#         if image_cfg.get("enabled", True) else None

#     out_base = out_dir / name
#     fig_info = _plot_field_and_image(
#         reduced, image_red, primary_radius, plot_cfg, image_cfg, out_base
#     )

#     diag_info = {}
#     if diag_cfg.get("enabled", True):
#         diag_info = _plot_diagnostic(reduced, diag_cfg, plot_cfg, out_dir / f"{name}_diagnostic")

#     appendix_info = {}
#     if image_red is not None and appendix_cfg.get("per_channel_image_grid", True):
#         appendix_info = _plot_image_channel_grid(
#             image_red, image_cfg, plot_cfg, appendix_cfg, out_dir / f"{name}_image_channels"
#         )

#     # save plotted arrays for reproducibility
#     npz_payload: dict[str, np.ndarray] = {}
#     for q in reduced:
#         r_key = _nearest_radius_key(reduced[q], primary_radius)
#         d = reduced[q][r_key]
#         sl = _slug(q)
#         npz_payload[f"{sl}_lon_deg"] = d["lon"]
#         npz_payload[f"{sl}_lat_deg"] = d["lat"]
#         npz_payload[f"{sl}_gt"] = d["gt"]
#         npz_payload[f"{sl}_ens_mean"] = d["ens_mean"]
#         npz_payload[f"{sl}_true_err"] = d["true_err"]
#         npz_payload[f"{sl}_disagreement"] = d["disag"]
#     if image_red is not None:
#         npz_payload["image_residual"] = image_red["residual"]
#     npz_path = out_dir / f"{name}_arrays.npz"
#     np.savez_compressed(npz_path, **npz_payload)

#     manifest = {
#         "name": name,
#         "spec_path": str(spec_path),
#         "benchmark_dir": str(benchmark_dir),
#         "output_dir": str(out_dir),
#         "device": device,
#         "path_remap": path_remap,
#         "condition": cond_cfg,
#         "n_seeds": len(seed_rows),
#         "seed_run_names": [r.get("run_name") for r in seed_rows],
#         "seeds": [r.get("seed") for r in seed_rows],
#         "quantities": list(reduced.keys()),
#         "radii": radii_all,
#         "primary_radius": primary_radius,
#         "error_mode": fields_cfg.get("error_mode", "ensemble_mean_abs"),
#         "disagreement": fields_cfg.get("disagreement", "std"),
#         "figure": fig_info,
#         "diagnostic": diag_info,
#         "appendix": appendix_info,
#         "arrays_npz": str(npz_path),
#     }
#     save_json(manifest, out_dir / f"{name}_manifest.json")
#     logger.info("Saved seed-disagreement diagnostic manifest: %s", out_dir / f"{name}_manifest.json")
#     return manifest
