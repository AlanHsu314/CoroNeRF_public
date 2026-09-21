############################################################
### class for generic scalar field artifacts
############################################################

from __future__ import annotations
from typing import Union, Optional
from dataclasses import dataclass
import logging

import numpy as np
import torch

from ..models.field_outputs import get_log_ne_from_raw, get_log_temp_from_raw
from ..geom.coords import llr_to_xyz
from ..util.torch import match_field

# helpers for the spec

@dataclass(frozen = True)
class ScalarFieldSpec:
    key: str
    display_name: str
    log_label: str
    frac_key: str
    gt_field_attr: str
    shell_dirname: str
    eval_prefix: str
    video_dirname: str
    cmap: str
    requires_target: Optional[str] = None

SCALAR_FIELD_SPECS = {
    "ne": ScalarFieldSpec(
        key="ne",
        display_name="Density",
        log_label="log10 n_e",
        frac_key="frac_ne",
        gt_field_attr="ne_field",
        shell_dirname="density_shells",
        eval_prefix="density_shell",
        video_dirname="density_shell_video",
        cmap = 'inferno',
        requires_target=None,
    ),
    "temp": ScalarFieldSpec(
        key="temp",
        display_name="Temperature",
        log_label="log10 T",
        frac_key="frac_temp",
        gt_field_attr="temp_field",
        shell_dirname="temperature_shells",
        eval_prefix="temperature_shell",
        video_dirname="temperature_shell_video",
        cmap = 'viridis',
        requires_target="ne_t",
    ),
}

def get_scalar_field_spec(quantity: str) -> ScalarFieldSpec:
    if quantity not in SCALAR_FIELD_SPECS:
        raise KeyError(f'Unknown scalar field quantity: {quantity}')
    return SCALAR_FIELD_SPECS[quantity]

def scalar_field_available(state, quantity: str) -> bool:
    '''
    wrapper for spec getter
    '''
    spec = get_scalar_field_spec(quantity)

    # pipeline state has to have the gt field if we are trying to reconstruct it
    # eventually this will need to change when we use it for observations
    if not hasattr(state, spec.gt_field_attr):
        return False
    if getattr(state, spec.gt_field_attr) is None:
        return False
    
    if spec.requires_target is None:
        return True
    
    return state.shared.get('reconstruction_target', 'ne') == spec.requires_target

def available_scalar_fields(state) -> list[str]:
    return [q for q in SCALAR_FIELD_SPECS if scalar_field_available(state, q)]

# generic products function

@dataclass
class ScalarShellProduct:
    quantity: str
    r: float
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    gt_log: np.ndarray
    pred_log: np.ndarray
    diff_log: np.ndarray
    vmin: float
    vmax: float

def _predict_scalar_log_field(state, quantity: str, x_aabb: torch.Tensor) -> torch.Tensor:
    '''
    wrapper for forward through model
    '''
    raw = state.model(x_aabb)
    target = state.shared.get('reconstruction_target', 'ne')

    if quantity == 'ne':
        return get_log_ne_from_raw(raw, target = target)
    elif quantity == 'temp':
        log_temp = get_log_temp_from_raw(raw, target = target)
        if log_temp is None:
            raise ValueError('requested temp prediction but current model does not predict temperature')
        return log_temp
    else:
        raise KeyError(f'Unknown scalar quantity: {quantity}')

def _default_shell_metric_bands(radii: list[float]) -> dict[str, tuple[Union[float, None], Union[float, None]]]:
    '''
    Default trusted shell metric bands, so stats are more physically interpretable
    '''
    if len(radii) == 0:
        return {
            'inner': (1.1, 2.0),
            'mid': (1.1, 2.6),
            'full': (None, None),
        }

    r_min = float(min(radii))
    r_max = float(max(radii))

    return {
        'inner': (r_min, min(2.0, r_max)),
        'mid': (r_min, min(2.6, r_max)),
        'full': (None, None),
    }

def build_scalar_shell_product(state, quantity: str, r_target: float) -> ScalarShellProduct:
    '''
    renders shell product and returns results
    '''
    logger = logging.getLogger('coroNeRF.build_scalar_shell_product')

    # first get spec
    spec = get_scalar_field_spec(quantity)

    if not scalar_field_available(state, quantity):
        raise ValueError(f'Scalar field {quantity} is not available for this run')
    
    # gt field, extract llr coords
    gt_field = getattr(state, spec.gt_field_attr)

    lon_axis = gt_field.lon_axis.detach().cpu()
    phi_axis = gt_field.phi_axis.detach().cpu()
    r_axis   = gt_field.r_axis.detach().cpu()

    lon_deg = np.rad2deg(lon_axis.numpy())
    lat_deg = np.rad2deg(phi_axis.numpy())

    # get r closest to requested shell
    r_diff = torch.abs(r_axis - float(r_target))
    r_idx = int(torch.argmin(r_diff))
    r_val = float(r_axis[r_idx].item())

    logger.debug(f'r_target:{r_target}, closest: {r_val}')

    # build llr array, convert to xyz array
    phi_grid, lon_grid = torch.meshgrid(phi_axis, lon_axis, indexing="ij")
    r_grid = torch.full_like(phi_grid, r_val)
    llr = torch.stack([lon_grid, phi_grid, r_grid], dim=-1).view(-1, 3)

    x_xyz = llr_to_xyz(llr)
    x_xyz = match_field(gt_field.field, x_xyz)

    # get gt values
    with torch.no_grad():
        log_gt_flat = gt_field(x_xyz)
    
    P, L = phi_grid.shape
    gt_log = log_gt_flat.view(P, L).detach().cpu().numpy()
    
    # get pred values
    x_aabb = (x_xyz / state.renderer.aabb_scale).to(
        device=state.renderer.device,
        dtype=state.renderer.dtype,
    )

    with torch.no_grad():
        log_pred_flat = _predict_scalar_log_field(state, quantity, x_aabb)

    pred_log = log_pred_flat.view(P, L).detach().cpu().numpy()

    # compute differences, and return
    diff_log = pred_log - gt_log

    # if you want color bars based on both gt and pred values
    # finite_vals = np.concatenate([
    #     gt_log[np.isfinite(gt_log)].ravel(),
    #     pred_log[np.isfinite(pred_log)].ravel(),
    # ]) if (np.isfinite(gt_log).any() or np.isfinite(pred_log).any()) else np.array([])

    # if finite_vals.size == 0:
    #     vmin, vmax = 0.0, 1.0
    # else:
    #     vmin, vmax = float(finite_vals.min()), float(finite_vals.max())

    # Use GT-only limits so the shell color scale stays fixed over training steps
    # for a given shell radius. This makes GT and Pred directly comparable over time.
    finite_gt = gt_log[np.isfinite(gt_log)].ravel()

    if finite_gt.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        # Percentiles are usually nicer than raw min/max for visualization.
        vmin = float(np.percentile(finite_gt, 1.0))
        vmax = float(np.percentile(finite_gt, 99.0))

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(finite_gt.min()), float(finite_gt.max())

    return ScalarShellProduct(
        quantity=quantity,
        r=r_val,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        gt_log=gt_log,
        pred_log=pred_log,
        diff_log=diff_log,
        vmin=vmin,
        vmax=vmax,
    )

def _eval_field_on_llr(state, gt_field, quantity, llr, out_shape):
    """Evaluate GT + prediction (log10) at LLR points -> (gt_log, pred_log, diff_log, vmin, vmax) reshaped to out_shape."""
    "part of build_scalar_shell_product(), can refactor if needed"
    x_xyz = llr_to_xyz(llr)
    x_xyz = match_field(gt_field.field, x_xyz)
    with torch.no_grad():
        gt_flat = gt_field(x_xyz)
    gt_log = gt_flat.view(*out_shape).detach().cpu().numpy()
    x_aabb = (x_xyz / state.renderer.aabb_scale).to(device=state.renderer.device, dtype=state.renderer.dtype)
    with torch.no_grad():
        pred_flat = _predict_scalar_log_field(state, quantity, x_aabb)
    pred_log = pred_flat.view(*out_shape).detach().cpu().numpy()
    diff_log = pred_log - gt_log
    finite_gt = gt_log[np.isfinite(gt_log)].ravel()
    if finite_gt.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(finite_gt, 1.0)); vmax = float(np.percentile(finite_gt, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(finite_gt.min()), float(finite_gt.max())
    return gt_log, pred_log, diff_log, vmin, vmax

# generic evaluator on products

def evaluate_scalar_shells(state, quantity: str, radii: list[float], metric_bands: Union[dict, None] = None) -> dict:
    '''
    unified scalar field evaluation on shells
    '''

    spec = get_scalar_field_spec(quantity)

    if not scalar_field_available(state, quantity):
        return {
            'available': False,
            'quantity': quantity,
            'reason': f'{quantity} not available for this run',
        }
    
    # per-row
    rows = []
    for r_target in radii:
        # grab difference products
        product = build_scalar_shell_product(state, quantity=quantity, r_target=r_target)

        gt_log = product.gt_log
        pred_log = product.pred_log

        # if there arent any good values
        ok = np.isfinite(gt_log) & np.isfinite(pred_log)
        if not np.any(ok):
            rows.append({
                "r": product.r,
                "mean_dlog": None,
                "median_dlog": None,
                "mean_abs_dlog": None,
                "rmse_dlog": None,
                f"mean_abs_{spec.frac_key}": None,
                f"median_abs_{spec.frac_key}": None,
            })
            continue

        # compute metrics
        dlog = pred_log[ok] - gt_log[ok]

        gt_lin = np.power(10.0, gt_log[ok])
        pred_lin = np.power(10.0, pred_log[ok])
        frac = np.abs(pred_lin - gt_lin) / np.maximum(np.abs(gt_lin), 1e-30)

        # append per-row
        rows.append({
            "r": product.r,
            "mean_dlog": float(np.mean(dlog)),                     # ME(log ne)
            "median_dlog": float(np.median(dlog)),                 # MedE(log ne)
            "mean_abs_dlog": float(np.mean(np.abs(dlog))),         # MAE(log ne)
            "rmse_dlog": float(np.sqrt(np.mean(dlog ** 2))),       # RMSE(log ne)
            f"mean_abs_{spec.frac_key}": float(np.mean(frac)),     # Relative MAE (log ne)
            f"median_abs_{spec.frac_key}": float(np.median(frac)), # no idea what the name for this is
        })

    # do per-metric band summarys (semi-avg over shells)
    if metric_bands is None:
        metric_bands = _default_shell_metric_bands(radii)

    from ..eval.metrics import summarize_scalar_shell_rows
    aggregates = summarize_scalar_shell_rows(rows, metric_bands, frac_key = spec.frac_key)

    return {
        "available": True,
        "quantity": quantity,
        "display_name": spec.display_name,
        "shells": rows,
        "aggregates": aggregates,
    }

#new dataclass for evaluating a slice with FIXED lon, so it is a lat-r 2D plot
@dataclass
class ScalarSliceProduct:
    quantity: str
    lon: float             # fixed longitude (deg)
    lat_deg: np.ndarray    # (P,)
    r_axis: np.ndarray     # (R,)  -- the x-axis (customizable range)
    gt_log: np.ndarray     # (P, R)
    pred_log: np.ndarray   # (P, R)
    diff_log: np.ndarray   # (P, R)
    vmin: float
    vmax: float

def build_scalar_slice_product(state, quantity: str, lon_target: float,
                               r_min: float | None = None, r_max: float | None = None,
                               n_r: int | None = None, r_axis=None) -> ScalarSliceProduct:
    """Fixed-longitude meridional slice: mesh (lat, r) at lon=lon_target. r range is customizable."""
    "note r_max should be set to within the GT field r range, or else itll extrapolate"
    spec = get_scalar_field_spec(quantity)
    if not scalar_field_available(state, quantity):
        raise ValueError(f'Scalar field {quantity} is not available for this run')
    gt_field = getattr(state, spec.gt_field_attr)
    phi_axis = gt_field.phi_axis.detach().cpu()                       # latitude (rad)
    gt_r = gt_field.r_axis.detach().cpu()
    lat_deg = np.rad2deg(phi_axis.numpy())

    if r_axis is not None:
        r_used = torch.as_tensor(np.asarray(r_axis, dtype=phi_axis.dtype))
    else:
        rmin = float(r_min) if r_min is not None else float(gt_r.min())
        rmax = float(r_max) if r_max is not None else float(gt_r.max())
        nr = int(n_r) if n_r is not None else int(gt_r.numel())
        r_used = torch.linspace(rmin, rmax, nr)

    lon_rad = float(np.deg2rad(float(lon_target)))                    # arbitrary lon; interpolator is periodic
    phi_grid, r_grid = torch.meshgrid(phi_axis, r_used, indexing="ij")   # (P, R)
    lon_grid = torch.full_like(phi_grid, lon_rad)
    llr = torch.stack([lon_grid, phi_grid, r_grid], dim=-1).view(-1, 3)

    gt_log, pred_log, diff_log, vmin, vmax = _eval_field_on_llr(state, gt_field, quantity, llr, phi_grid.shape)
    return ScalarSliceProduct(quantity, float(lon_target), lat_deg, r_used.numpy(),
                              gt_log, pred_log, diff_log, vmin, vmax)





