#######################################################
### EVALUATION METRICS (FIELD AND IMAGE SPACE)
#######################################################

from __future__ import annotations
import numpy as np
from typing import Union

def masked_mse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    '''
    pred, gt: (H, W, C) or (H, W)
    mask: (H, W) boolean, True = keep
    '''

    if pred.ndim == 2:
        pred = pred[..., None]
        gt = gt[..., None]
    
    keep = mask[..., None]
    diff = (pred - gt)**2
    return float(np.mean(diff[keep]))

def masked_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    '''
    absolute error under mask
    '''
    if pred.ndim == 2:
        pred = pred[..., None]
        gt = gt[..., None]

    keep = mask[..., None]
    diff = np.abs(pred - gt)
    return float(np.mean(diff[keep]))

def psnr_from_mse(mse: float, data_range: float) -> float:
    if mse <= 0:
        return float('inf')
    return float(10.0 * np.log10((data_range **2) / mse))

def masked_log_mse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    '''
    mse but on log images, assuming all intensities are >=0
    '''

    pred = np.maximum(pred, 0.0) # clip
    gt = np.maximum(gt, 0.0)

    log_pred = np.log10(pred + eps)
    log_gt = np.log10(gt + eps)

    return masked_mse(log_pred, log_gt, mask)

def masked_log_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> float:
    '''
    mae on log10 images
    '''
    pred = np.maximum(pred, 0.0)
    gt = np.maximum(gt, 0.0)

    log_pred = np.log10(pred + eps)
    log_gt = np.log10(gt + eps)

    return masked_mae(log_pred, log_gt, mask)

def safe_float_mean(vals: list[float]) -> Union[float, None]:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))

def safe_float_std(vals: list[float]) -> Union[float, None]:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(vals) == 0:
        return None
    return float(np.std(vals))

# might be legacy, keep for now
def summarize_shell_rows(shell_rows: list[dict], bands: dict[str, tuple[Union[float, None], Union[float, None]]]) -> dict:
    '''
    Aggregate per-shell rows into trusted radial bands.

    shell_rows:
        [
            {
                "r": ...,
                "mean_abs_dlog": ...,
                "rmse_dlog": ...,
                "mean_abs_frac_ne": ...,
                "median_abs_frac_ne": ...,
            },
            ...
        ]

    bands:
        {
            "inner": (1.1, 2.0),
            "mid":   (1.1, 2.6),
            "full":  (None, None),
        }
    '''
    out = {}

    for band_name, (r_lo, r_hi) in bands.items():
        rows_band = []
        for row in shell_rows:
            r = row.get('r', None)
            if r is None:
                continue
            if r_lo is not None and r < r_lo:
                continue
            if r_hi is not None and r > r_hi:
                continue
            rows_band.append(row)

        # per-shell summaries
        mean_dlog = [row.get('mean_dlog') for row in rows_band if row.get('mean_dlog') is not None]
        median_dlog = [row.get('median_dlog') for row in rows_band if row.get('median_dlog') is not None]
        mean_abs_dlog = [row.get('mean_abs_dlog') for row in rows_band if row.get('mean_abs_dlog') is not None]
        rmse = [row.get('rmse_dlog') for row in rows_band if row.get('rmse_dlog') is not None]
        frac_mean = [row.get('mean_abs_frac_ne') for row in rows_band if row.get('mean_abs_frac_ne') is not None]
        frac_med = [row.get('median_abs_frac_ne') for row in rows_band if row.get('median_abs_frac_ne') is not None]

        # per-band summaries
        out[band_name] = {
            'r_min': r_lo,
            'r_max': r_hi,
            'num_shells': len(rows_band),
            'shell_mean_dlog_mean': safe_float_mean(mean_dlog),
            'shell_mean_dlog_std': safe_float_std(mean_dlog),
            'shell_median_dlog_mean': safe_float_mean(median_dlog),
            'shell_median_dlog_std': safe_float_std(median_dlog),
            'shell_mean_abs_dlog_mean': safe_float_mean(mean_abs_dlog),
            'shell_mean_abs_dlog_std': safe_float_std(mean_abs_dlog),
            'shell_rmse_dlog_mean': safe_float_mean(rmse),
            'shell_rmse_dlog_std': safe_float_std(rmse),
            'shell_mean_abs_frac_ne_mean': safe_float_mean(frac_mean),
            'shell_mean_abs_frac_ne_std': safe_float_std(frac_mean),
            'shell_median_abs_frac_ne_mean': safe_float_mean(frac_med),
            'shell_median_abs_frac_ne_std': safe_float_std(frac_med),
        }

    return out

def summarize_image_metric_matrix(vals: np.ndarray) -> dict:
    '''
    vals: (V, C)
    returns per-channel + mean-over-channels summaries
    '''
    if vals.ndim != 2:
        raise ValueError(f'Expected (V, C) metric matrix, got shape {vals.shape}')

    mean_per_channel = vals.mean(axis=0)
    std_per_channel = vals.std(axis=0)
    median_per_channel = np.median(vals, axis=0)

    return {
        'mean_per_channel': mean_per_channel.tolist(),
        'std_per_channel': std_per_channel.tolist(),
        'median_per_channel': median_per_channel.tolist(),
        'mean_over_channels': float(np.mean(mean_per_channel)),
        'median_over_channels': float(np.mean(median_per_channel)),
    }

# extended to generic scalar summarizer
def summarize_scalar_shell_rows(
    shell_rows: list[dict],
    bands: dict[str, tuple[Union[float, None], Union[float, None]]],
    frac_key: str,
) -> dict:
    out = {}

    for band_name, (r_lo, r_hi) in bands.items():
        rows_band = []
        for row in shell_rows:
            r = row.get('r', None)
            if r is None:
                continue
            if r_lo is not None and r < r_lo:
                continue
            if r_hi is not None and r > r_hi:
                continue
            rows_band.append(row)
        
        # per-shell summaries
        mean_dlog = [row.get('mean_dlog') for row in rows_band if row.get('mean_dlog') is not None]
        median_dlog = [row.get('median_dlog') for row in rows_band if row.get('median_dlog') is not None]
        mean_abs_dlog = [row.get('mean_abs_dlog') for row in rows_band if row.get('mean_abs_dlog') is not None]
        rmse = [row.get('rmse_dlog') for row in rows_band if row.get('rmse_dlog') is not None]
        frac_mean = [row.get(f'mean_abs_{frac_key}') for row in rows_band if row.get(f'mean_abs_{frac_key}') is not None]
        frac_med = [row.get(f'median_abs_{frac_key}') for row in rows_band if row.get(f'median_abs_{frac_key}') is not None]

        # per-band summaries
        out[band_name] = {
            'r_min': r_lo,
            'r_max': r_hi,
            'num_shells': len(rows_band),
            'shell_mean_dlog_mean': safe_float_mean(mean_dlog),
            'shell_mean_dlog_std': safe_float_std(mean_dlog),
            'shell_median_dlog_mean': safe_float_mean(median_dlog),
            'shell_median_dlog_std': safe_float_std(median_dlog),
            'shell_mean_abs_dlog_mean': safe_float_mean(mean_abs_dlog),
            'shell_mean_abs_dlog_std': safe_float_std(mean_abs_dlog),
            'shell_rmse_dlog_mean': safe_float_mean(rmse),
            'shell_rmse_dlog_std': safe_float_std(rmse),
            f'shell_mean_abs_{frac_key}_mean': safe_float_mean(frac_mean),
            f'shell_mean_abs_{frac_key}_std': safe_float_std(frac_mean),
            f'shell_median_abs_{frac_key}_mean': safe_float_mean(frac_med),
            f'shell_median_abs_{frac_key}_std': safe_float_std(frac_med),
        }

    return out




