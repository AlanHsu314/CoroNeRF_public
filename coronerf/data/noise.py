#################################################
### observation noise helpers
#################################################

from __future__ import annotations
from typing import Union

import json
import logging
from numbers import Number
from pathlib import Path
from typing import Any

import torch
import math

logger = logging.getLogger('coroNeRF.noise')

def _as_builtin(x: Any):
    if isinstance(x, dict):
        return {str(k): _as_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_as_builtin(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().tolist()
    return x

def save_noise_json(meta: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('w') as f:
        json.dump(_as_builtin(meta), f, indent=2)

def get_observation_noise_cfg(cfg) -> dict:
    '''
    resolved observation noise cfg
    '''
    raw = dict(cfg.get('observation_noise', {}) or {})

    out = {
        'enabled': bool(raw.get('enabled', False)),
        'apply_to': raw.get('apply_to', 'train'),
        'model': raw.get('model', 'effective_shot'),
        'seed_offset': int(raw.get('seed_offset', 1000003)),
        'diagnostics': dict(raw.get('diagnostics', {}) or {}),
        'effective_shot': dict(raw.get('effective_shot', {}) or {}),
        'hetero_gaussian': dict(raw.get('hetero_gaussian', {}) or {}),
    }

    # noise diagnostics defaults
    out['diagnostics'].setdefault('enabled', True)
    out['diagnostics'].setdefault('positive_eps', 1.0e-12)
    out['diagnostics'].setdefault('percentiles', [10.0, 50.0, 90.0])
    out['diagnostics'].setdefault('max_quantile_samples', 200000)

    out['diagnostics'].setdefault('radial_bins', [
        {'name': 'low_corona', 'r_min': 1.01, 'r_max': 1.5},
        {'name': 'inner', 'r_min': 1.01, 'r_max': 2.0},
        {'name': 'mid_outer_inner', 'r_min': 1.5, 'r_max': 2.0},
    ])

    # hetero gaussian defaults
    out['hetero_gaussian'].setdefault('shot_coeff', None)
    out['hetero_gaussian'].setdefault('shot_coeff_by_global_channel', None)
    out['hetero_gaussian'].setdefault('sigma_floor', 0.0)
    out['hetero_gaussian'].setdefault('sigma_floor_by_global_channel', None)

    # checks modes
    if not out['enabled']:
        return out

    if out['apply_to'] != 'train':
        raise NotImplementedError(
            f"observation_noise.apply_to='{out['apply_to']}' is not implemented; use 'train'"
        )

    # noise model checks
    if out['model'] == 'effective_shot':
        # check alpha values
        eff = out['effective_shot']
        alpha = eff.get('alpha', None)
        alpha_by_global_channel = eff.get('alpha_by_global_channel', None)

        if alpha is None and alpha_by_global_channel is None:
            raise ValueError(
                'observation_noise.enabled=True with model=effective_shot requires '
                'either effective_shot.alpha or effective_shot.alpha_by_global_channel'
            )
    elif out['model'] == 'hetero_gaussian':
        # check coeff values
        hg = out['hetero_gaussian']
        shot_coeff = hg.get('shot_coeff', None)
        shot_coeff_by_global_channel = hg.get('shot_coeff_by_global_channel', None)

        if shot_coeff is None and shot_coeff_by_global_channel is None:
            raise ValueError(
                'observation_noise.enabled=True with model=hetero_gaussian requires '
                'either hetero_gaussian.shot_coeff or hetero_gaussian.shot_coeff_by_global_channel'
            )
    else:
        raise NotImplementedError(
            f"Unknown observation_noise.model='{out['model']}'. "
            f"Currently supported: effective_shot, hetero_gaussian"
        )

    return out

def _resolve_global_channel_ids(n_channels: int, global_channel_ids=None) -> list[int]:
    if global_channel_ids is None:
        return list(range(int(n_channels)))

    out = [int(i) for i in global_channel_ids]
    if len(out) != int(n_channels):
        raise ValueError(
            f'global_channel_ids has length {len(out)}, expected {n_channels} active channels'
        )
    return out

def resolve_effective_shot_alpha(
    noise_cfg: dict,
    n_channels: int,
    global_channel_ids=None,
) -> torch.Tensor:
    '''
    robust checks on alpha cfg values (single or per-channel)
    '''
    eff = dict(noise_cfg.get('effective_shot', {}) or {})
    alpha = eff.get('alpha', None)
    alpha_by_global_channel = eff.get('alpha_by_global_channel', None)

    if alpha is not None and alpha_by_global_channel is not None:
        raise ValueError(
            'Provide either effective_shot.alpha or effective_shot.alpha_by_global_channel, not both'
        )

    if alpha_by_global_channel is not None:
        if not isinstance(alpha_by_global_channel, (list, tuple)):
            raise TypeError(
                'effective_shot.alpha_by_global_channel must be a list/tuple of positive numbers'
            )
        ids = _resolve_global_channel_ids(n_channels, global_channel_ids)
        max_id = max(ids) if ids else -1
        if max_id >= len(alpha_by_global_channel):
            raise IndexError(
                f'effective_shot.alpha_by_global_channel has length {len(alpha_by_global_channel)}, '
                f'but active global channel id {max_id} was requested'
            )
        alpha_vals = [float(alpha_by_global_channel[i]) for i in ids]
    else:
        if isinstance(alpha, Number):
            alpha_vals = [float(alpha)] * int(n_channels)
        elif isinstance(alpha, (list, tuple)):
            alpha_vals = [float(v) for v in alpha]
            if len(alpha_vals) != int(n_channels):
                raise ValueError(
                    f'effective_shot.alpha has length {len(alpha_vals)}, expected {n_channels} active channels'
                )
        else:
            raise TypeError(
                'effective_shot.alpha must be either a positive scalar or a list/tuple '
                'matching the active channel count'
            )

    bad = [float(v) for v in alpha_vals if float(v) <= 0.0]
    if bad:
        raise ValueError(f'effective_shot alpha values must be positive, got {bad}')

    return torch.tensor(alpha_vals, dtype=torch.float32)

def _resolve_scalar_or_channel_values(
    scalar_value,
    values_by_global_channel,
    n_channels: int,
    global_channel_ids=None,
    *,
    name: str,
    allow_zero: bool,
) -> torch.Tensor:
    '''
    generic resolver for scalar/per channel resolver for the noise coeffs

    currently used my hetero_gaussian noise case
    '''
    if scalar_value is not None and values_by_global_channel is not None:
        raise ValueError(f'Provide either {name} or {name}_by_global_channel, not both')

    if values_by_global_channel is not None:
        if not isinstance(values_by_global_channel, (list, tuple)):
            raise TypeError(f'{name}_by_global_channel must be a list/tuple')
        ids = _resolve_global_channel_ids(n_channels, global_channel_ids)
        max_id = max(ids) if ids else -1
        if max_id >= len(values_by_global_channel):
            raise IndexError(
                f'{name}_by_global_channel has length {len(values_by_global_channel)}, '
                f'but active global channel id {max_id} was requested'
            )
        vals = [float(values_by_global_channel[i]) for i in ids]
    else:
        if isinstance(scalar_value, Number):
            vals = [float(scalar_value)] * int(n_channels)
        elif isinstance(scalar_value, (list, tuple)):
            vals = [float(v) for v in scalar_value]
            if len(vals) != int(n_channels):
                raise ValueError(
                    f'{name} has length {len(vals)}, expected {n_channels} active channels'
                )
        else:
            raise TypeError(
                f'{name} must be either a scalar or a list/tuple matching the active channel count'
            )

    if allow_zero:
        bad = [float(v) for v in vals if float(v) < 0.0]
        if bad:
            raise ValueError(f'{name} values must be nonnegative, got {bad}')
    else:
        bad = [float(v) for v in vals if float(v) <= 0.0]
        if bad:
            raise ValueError(f'{name} values must be positive, got {bad}')

    return torch.tensor(vals, dtype=torch.float32)

def resolve_hetero_gaussian_params(
    noise_cfg: dict,
    n_channels: int,
    global_channel_ids=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    resolves params for heterogaussian noise case
    '''
    hg = dict(noise_cfg.get('hetero_gaussian', {}) or {})

    shot_coeff = _resolve_scalar_or_channel_values(
        scalar_value=hg.get('shot_coeff', None),
        values_by_global_channel=hg.get('shot_coeff_by_global_channel', None),
        n_channels=n_channels,
        global_channel_ids=global_channel_ids,
        name='hetero_gaussian.shot_coeff',
        allow_zero=False,
    )

    sigma_floor = _resolve_scalar_or_channel_values(
        scalar_value=hg.get('sigma_floor', 0.0),
        values_by_global_channel=hg.get('sigma_floor_by_global_channel', None),
        n_channels=n_channels,
        global_channel_ids=global_channel_ids,
        name='hetero_gaussian.sigma_floor',
        allow_zero=True,
    )

    return shot_coeff, sigma_floor

def _channel_name(local_idx: int, global_idx: int, line_names=None) -> str:
    if line_names is None:
        return f'ch_{global_idx}'
    if local_idx < len(line_names):
        return str(line_names[local_idx])
    return f'ch_{global_idx}'

def _estimate_rsun_pixels_from_mask(mask_2d: torch.Tensor) -> float:
    """
    Estimate solar-disk radius in pixels from the occulted disk region (~mask).
    Assumes mask=True for kept pixels and mask=False inside the occulted disk.
    """
    mask_bool = mask_2d.bool()
    disk_mask = ~mask_bool
    n_disk = int(disk_mask.sum().item())
    if n_disk <= 0:
        raise ValueError('Could not estimate solar radius from mask: disk region is empty')
    return float((n_disk / math.pi) ** 0.5)

def _build_flat_projected_radius_from_mask(
    mask_2d: torch.Tensor,
    pix_idx: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Build projected image-plane radius (in R_sun units) for each flattened training pixel.
    The output order matches CoronaDataset.imgs_flat / pix_idx order.
    """
    rsun_px = _estimate_rsun_pixels_from_mask(mask_2d)

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij',
    )
    cy = 0.5 * (H - 1)
    cx = 0.5 * (W - 1)
    rho_2d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / rsun_px

    rho_flat_full = rho_2d.reshape(-1)
    
    # pix_idx indexes the flattened multi-view dataset (V * H * W),
    # while rho_flat_full only covers one image plane (H * W).
    # Project back to the per-view image-plane index.
    pix_idx_cpu = pix_idx.detach().cpu().long()
    pix_idx_hw = pix_idx_cpu % int(H * W)

    return rho_flat_full[pix_idx_hw].clone()

def _quantiles(values: torch.Tensor, percentiles: list[float], max_samples: Union[int, None] = None) -> list[float]:
    if values.numel() == 0:
        return [0.0 for _ in percentiles]
    
    if not percentiles:
        return []
    
    # bound if too many samples
    if max_samples is not None and int(max_samples) > 0 and values.numel() > int(max_samples):
        stride = max(1, math.ceil(values.numel() / int(max_samples)))
        values = values[::stride].contiguous()

    q = torch.tensor(
        [max(0.0, min(100.0, float(p))) / 100.0 for p in percentiles],
        dtype=values.dtype,
        device=values.device,
    )
    out = torch.quantile(values, q)
    return [float(v.item()) for v in out]

def _summarize_channel_noise_subset(
    clean_c: torch.Tensor,
    noisy_c: torch.Tensor,
    alpha_c: float,
    percentiles: list[float],
    positive_eps: float,
    max_quantile_samples: int,
) -> dict:
    '''
    summary helper for effective shot noise, will change the name later
    '''
    delta_c = noisy_c - clean_c

    positive = clean_c > positive_eps
    clean_pos = clean_c[positive]
    noisy_pos = noisy_c[positive]
    delta_pos = delta_c[positive]

    clean_q = _quantiles(clean_pos, percentiles, max_samples=max_quantile_samples)
    expected_counts_q = [float(alpha_c * q) for q in clean_q]
    theory_snr_q = [float((max(v, 0.0)) ** 0.5) for v in expected_counts_q]

    noise_rmse = float(torch.sqrt(torch.mean(delta_c.pow(2))).item()) if clean_c.numel() > 0 else 0.0
    clean_rms = float(torch.sqrt(torch.mean(clean_c.pow(2))).item()) if clean_c.numel() > 0 else 0.0
    noise_rel_rmse = float(noise_rmse / max(clean_rms, positive_eps))

    if clean_pos.numel() > 0:
        mean_abs_rel_noise = float(
            torch.mean(torch.abs(delta_pos) / clean_pos.clamp_min(positive_eps)).item()
        )
        clean_mean_pos = float(torch.mean(clean_pos).item())
        noisy_mean_pos = float(torch.mean(noisy_pos).item())
    else:
        mean_abs_rel_noise = None
        clean_mean_pos = 0.0
        noisy_mean_pos = 0.0

    frac_zero_observed = float(torch.mean((noisy_c <= positive_eps).float()).item()) if noisy_c.numel() > 0 else 0.0
    frac_negative_clean = float(torch.mean((clean_c < 0.0).float()).item()) if clean_c.numel() > 0 else 0.0

    return {
        'num_pixels': int(clean_c.numel()),
        'num_positive_pixels': int(clean_pos.numel()),
        'clean_mean_positive': clean_mean_pos,
        'noisy_mean_positive': noisy_mean_pos,
        'clean_percentiles': {
            str(int(p) if float(p).is_integer() else p): float(v)
            for p, v in zip(percentiles, clean_q)
        },
        'expected_counts_percentiles': {
            str(int(p) if float(p).is_integer() else p): float(v)
            for p, v in zip(percentiles, expected_counts_q)
        },
        'theory_snr_percentiles': {
            str(int(p) if float(p).is_integer() else p): float(v)
            for p, v in zip(percentiles, theory_snr_q)
        },
        'noise_rmse': noise_rmse,
        'noise_rel_rmse': noise_rel_rmse,
        'mean_abs_rel_noise': mean_abs_rel_noise,
        'frac_zero_observed': frac_zero_observed,
        'frac_negative_clean': frac_negative_clean,
    }

def _summarize_channel_gaussian_subset(
    clean_c: torch.Tensor,
    noisy_c: torch.Tensor,
    sigma_c: torch.Tensor,
    percentiles: list[float],
    positive_eps: float,
    max_quantile_samples: int,
) -> dict:
    '''
    summary helper for hetero_gaussian noise
    '''
    delta_c = noisy_c - clean_c

    positive = clean_c > positive_eps
    clean_pos = clean_c[positive]
    noisy_pos = noisy_c[positive]
    delta_pos = delta_c[positive]
    sigma_pos = sigma_c[positive]

    clean_q = _quantiles(clean_pos, percentiles, max_samples=max_quantile_samples)
    sigma_q = _quantiles(sigma_pos, percentiles, max_samples=max_quantile_samples)
    theory_snr_q = [
        float(q_clean / max(q_sigma, positive_eps))
        for q_clean, q_sigma in zip(clean_q, sigma_q)
    ]

    noise_rmse = float(torch.sqrt(torch.mean(delta_c.pow(2))).item()) if clean_c.numel() > 0 else 0.0
    clean_rms = float(torch.sqrt(torch.mean(clean_c.pow(2))).item()) if clean_c.numel() > 0 else 0.0
    noise_rel_rmse = float(noise_rmse / max(clean_rms, positive_eps))

    if clean_pos.numel() > 0:
        mean_abs_rel_noise = float(
            torch.mean(torch.abs(delta_pos) / clean_pos.clamp_min(positive_eps)).item()
        )
        clean_mean_pos = float(torch.mean(clean_pos).item())
        noisy_mean_pos = float(torch.mean(noisy_pos).item())
    else:
        mean_abs_rel_noise = None
        clean_mean_pos = 0.0
        noisy_mean_pos = 0.0

    frac_negative_observed = float(torch.mean((noisy_c < 0.0).float()).item()) if noisy_c.numel() > 0 else 0.0
    frac_nonpositive_observed = float(torch.mean((noisy_c <= positive_eps).float()).item()) if noisy_c.numel() > 0 else 0.0

    return {
        'num_pixels': int(clean_c.numel()),
        'num_positive_pixels': int(clean_pos.numel()),
        'clean_mean_positive': clean_mean_pos,
        'noisy_mean_positive': noisy_mean_pos,
        'clean_percentiles': {
            str(int(p) if float(p).is_integer() else p): float(v)
            for p, v in zip(percentiles, clean_q)
        },
        'sigma_percentiles': {
            str(int(p) if float(p).is_integer() else p): float(v)
            for p, v in zip(percentiles, sigma_q)
        },
        'theory_snr_percentiles': {
            str(int(p) if float(p).is_integer() else p): float(v)
            for p, v in zip(percentiles, theory_snr_q)
        },
        'noise_rmse': noise_rmse,
        'noise_rel_rmse': noise_rel_rmse,
        'mean_abs_rel_noise': mean_abs_rel_noise,
        'frac_negative_observed': frac_negative_observed,
        'frac_nonpositive_observed': frac_nonpositive_observed,
    }

def _build_effective_shot_diagnostics(
    clean_flat: torch.Tensor,
    noisy_flat: torch.Tensor,
    alpha: torch.Tensor,
    rho_flat: Union[torch.Tensor, None] = None,
    global_channel_ids=None,
    line_names=None,
    diagnostics_cfg=None,
) -> dict:
    '''
    effective shot noise diagnostics
    '''
    # cfg defaults
    diagnostics_cfg = dict(diagnostics_cfg or {})
    percentiles = [float(p) for p in diagnostics_cfg.get('percentiles', [10.0, 50.0, 90.0])]
    positive_eps = float(diagnostics_cfg.get('positive_eps', 1.0e-12))
    max_quantile_samples = diagnostics_cfg.get('max_quantile_samples', 200000)
    radial_bins_cfg = list(diagnostics_cfg.get('radial_bins', []) or [])

    # resolve channel ids just in case
    n_channels = int(clean_flat.shape[1])
    global_ids = _resolve_global_channel_ids(n_channels, global_channel_ids)

    # loop through channels
    channels = []
    for local_idx in range(n_channels):
        # resolved channel name just in case
        global_idx = int(global_ids[local_idx])
        line_name = _channel_name(local_idx, global_idx, line_names=line_names)

        clean_c = clean_flat[:, local_idx]
        noisy_c = noisy_flat[:, local_idx]
        alpha_c = float(alpha[local_idx].item())

        overall_stats = _summarize_channel_noise_subset(
            clean_c=clean_c,
            noisy_c=noisy_c,
            alpha_c=alpha_c,
            percentiles=percentiles,
            positive_eps=positive_eps,
            max_quantile_samples=max_quantile_samples,
        )

        radial_bin_stats = []
        if rho_flat is not None and radial_bins_cfg:
            for bin_cfg in radial_bins_cfg:
                name = str(bin_cfg.get('name', 'unnamed'))
                r_min = bin_cfg.get('r_min', None)
                r_max = bin_cfg.get('r_max', None)

                select = torch.ones_like(rho_flat, dtype=torch.bool)
                if r_min is not None:
                    select = select & (rho_flat >= float(r_min))
                if r_max is not None:
                    select = select & (rho_flat < float(r_max))

                clean_bin = clean_c[select]
                noisy_bin = noisy_c[select]

                bin_stats = _summarize_channel_noise_subset(
                    clean_c=clean_bin,
                    noisy_c=noisy_bin,
                    alpha_c=alpha_c,
                    percentiles=percentiles,
                    positive_eps=positive_eps,
                    max_quantile_samples=max_quantile_samples,
                )
                bin_stats['name'] = name
                bin_stats['r_min'] = None if r_min is None else float(r_min)
                bin_stats['r_max'] = None if r_max is None else float(r_max)
                radial_bin_stats.append(bin_stats)

        channels.append({
            'local_channel': int(local_idx),
            'global_channel': int(global_idx),
            'line_name': line_name,
            'alpha': alpha_c,
            **overall_stats,
            'radial_bins': radial_bin_stats,
        })

    # return diagnostic summaries
    return {
        'enabled': True,
        'model': 'effective_shot',
        'apply_to': 'train',
        'num_train_pixels': int(clean_flat.shape[0]),
        'num_active_channels': int(n_channels),
        'diagnostics_percentiles': percentiles,
        'diagnostics_max_quantile_samples': int(max_quantile_samples),
        'channels': channels,
        'diagnostics_radial_bins': radial_bins_cfg,
    }

def _build_hetero_gaussian_diagnostics(
    clean_flat: torch.Tensor,
    noisy_flat: torch.Tensor,
    sigma_flat: torch.Tensor,
    shot_coeff: torch.Tensor,
    sigma_floor: torch.Tensor,
    rho_flat: Union[torch.Tensor, None] = None,
    global_channel_ids=None,
    line_names=None,
    diagnostics_cfg=None,
) -> dict:
    '''
    heteroscedastic gaussian noise diagnostics
    '''
    diagnostics_cfg = dict(diagnostics_cfg or {})
    percentiles = [float(p) for p in diagnostics_cfg.get('percentiles', [10.0, 50.0, 90.0])]
    positive_eps = float(diagnostics_cfg.get('positive_eps', 1.0e-12))
    max_quantile_samples = diagnostics_cfg.get('max_quantile_samples', 200000)
    radial_bins_cfg = list(diagnostics_cfg.get('radial_bins', []) or [])

    n_channels = int(clean_flat.shape[1])
    global_ids = _resolve_global_channel_ids(n_channels, global_channel_ids)

    channels = []
    for local_idx in range(n_channels):
        global_idx = int(global_ids[local_idx])
        line_name = _channel_name(local_idx, global_idx, line_names=line_names)

        clean_c = clean_flat[:, local_idx]
        noisy_c = noisy_flat[:, local_idx]
        sigma_c = sigma_flat[:, local_idx]

        overall_stats = _summarize_channel_gaussian_subset(
            clean_c=clean_c,
            noisy_c=noisy_c,
            sigma_c=sigma_c,
            percentiles=percentiles,
            positive_eps=positive_eps,
            max_quantile_samples=max_quantile_samples,
        )

        radial_bin_stats = []
        if rho_flat is not None and radial_bins_cfg:
            for bin_cfg in radial_bins_cfg:
                name = str(bin_cfg.get('name', 'unnamed'))
                r_min = bin_cfg.get('r_min', None)
                r_max = bin_cfg.get('r_max', None)

                select = torch.ones_like(rho_flat, dtype=torch.bool)
                if r_min is not None:
                    select = select & (rho_flat >= float(r_min))
                if r_max is not None:
                    select = select & (rho_flat < float(r_max))

                clean_bin = clean_c[select]
                noisy_bin = noisy_c[select]
                sigma_bin = sigma_c[select]

                bin_stats = _summarize_channel_gaussian_subset(
                    clean_c=clean_bin,
                    noisy_c=noisy_bin,
                    sigma_c=sigma_bin,
                    percentiles=percentiles,
                    positive_eps=positive_eps,
                    max_quantile_samples=max_quantile_samples,
                )
                bin_stats['name'] = name
                bin_stats['r_min'] = None if r_min is None else float(r_min)
                bin_stats['r_max'] = None if r_max is None else float(r_max)
                radial_bin_stats.append(bin_stats)

        channels.append({
            'local_channel': int(local_idx),
            'global_channel': int(global_idx),
            'line_name': line_name,
            'shot_coeff': float(shot_coeff[local_idx].item()),
            'sigma_floor': float(sigma_floor[local_idx].item()),
            **overall_stats,
            'radial_bins': radial_bin_stats,
        })

    return {
        'enabled': True,
        'model': 'hetero_gaussian',
        'apply_to': 'train',
        'num_train_pixels': int(clean_flat.shape[0]),
        'num_active_channels': int(n_channels),
        'diagnostics_percentiles': percentiles,
        'diagnostics_max_quantile_samples': int(max_quantile_samples),
        'diagnostics_radial_bins': radial_bins_cfg,
        'channels': channels,
    }

def apply_effective_shot_noise_flat(
    clean_flat: torch.Tensor,
    alpha: torch.Tensor,
    seed: int,
    rho_flat: Union[torch.Tensor, None] = None,
    global_channel_ids=None,
    line_names=None,
    diagnostics_cfg=None,
) -> tuple[torch.Tensor, dict]:
    '''
    helper, apply shot noise based on alpha to given intensity pixels
    '''
    if clean_flat.ndim != 2:
        raise ValueError(f'clean_flat must have shape (N, C), got {tuple(clean_flat.shape)}')

    alpha = alpha.to(device=clean_flat.device, dtype=clean_flat.dtype)
    if alpha.ndim != 1 or alpha.shape[0] != clean_flat.shape[1]:
        raise ValueError(
            f'alpha must have shape ({clean_flat.shape[1]},), got {tuple(alpha.shape)}'
        )
    
    # just in case > 0, I_{i, c} [ergs s^-1 cm^-2 arcsec^-2]
    clean_nonneg = clean_flat.clamp_min(0.0)

    # rng
    gen = torch.Generator(device=clean_flat.device.type)
    gen.manual_seed(int(seed))

    # alpha has units [photons ergs^-1 s^1 cm^2 arcsec^2]
    # \lambda_{i, c} = I_{i, c} * \alpha_c, compute effective counts [photons]
    lam = clean_nonneg * alpha.view(1, -1)
    
    # sample noise observations K_{i, c} ~ Poisson(\lambda_{i,c})
    counts = torch.poisson(lam, generator=gen)

    # map back to original space I^{obs}_{i,c} = K_{i, c} / \alpha_c
    noisy_flat = counts / alpha.view(1, -1)

    # diagnostics, only if enabled
    diagnostics_cfg = dict(diagnostics_cfg or {})
    diagnostics_enabled = bool(diagnostics_cfg.get('enabled', True))

    if diagnostics_enabled:
        meta = _build_effective_shot_diagnostics(
            clean_flat=clean_nonneg,
            noisy_flat=noisy_flat,
            alpha=alpha,
            rho_flat = rho_flat,
            global_channel_ids=global_channel_ids,
            line_names=line_names,
            diagnostics_cfg=diagnostics_cfg,
        )
    else:
        meta = {
            'enabled': True,
            'model': 'effective_shot',
            'apply_to': 'train',
            'num_train_pixels': int(clean_flat.shape[0]),
            'num_active_channels': int(clean_flat.shape[1]),
            'diagnostics_enabled': False,
            'channels': [],
        }

    meta['seed'] = int(seed)

    return noisy_flat, meta

def apply_hetero_gaussian_noise_flat(
    clean_flat: torch.Tensor,
    shot_coeff: torch.Tensor,
    sigma_floor: torch.Tensor,
    seed: int,
    rho_flat: Union[torch.Tensor, None] = None,
    global_channel_ids=None,
    line_names=None,
    diagnostics_cfg=None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if clean_flat.ndim != 2:
        raise ValueError(f'clean_flat must have shape (N, C), got {tuple(clean_flat.shape)}')

    shot_coeff = shot_coeff.to(device=clean_flat.device, dtype=clean_flat.dtype)
    sigma_floor = sigma_floor.to(device=clean_flat.device, dtype=clean_flat.dtype)

    if shot_coeff.ndim != 1 or shot_coeff.shape[0] != clean_flat.shape[1]:
        raise ValueError(f'shot_coeff must have shape ({clean_flat.shape[1]},), got {tuple(shot_coeff.shape)}')
    if sigma_floor.ndim != 1 or sigma_floor.shape[0] != clean_flat.shape[1]:
        raise ValueError(f'sigma_floor must have shape ({clean_flat.shape[1]},), got {tuple(sigma_floor.shape)}')

    clean_nonneg = clean_flat.clamp_min(0.0)

    sigma_flat = torch.sqrt(
        torch.clamp(
            clean_nonneg * shot_coeff.view(1, -1) + sigma_floor.view(1, -1).pow(2),
            min=0.0,
        )
    )

    gen = torch.Generator(device=clean_flat.device.type)
    gen.manual_seed(int(seed))
    noise = torch.randn(
        clean_nonneg.shape,
        generator=gen,
        device=clean_flat.device,
        dtype=clean_flat.dtype,
    )

    noisy_flat = clean_nonneg + sigma_flat * noise

    diagnostics_cfg = dict(diagnostics_cfg or {})
    diagnostics_enabled = bool(diagnostics_cfg.get('enabled', True))

    if diagnostics_enabled:
        meta = _build_hetero_gaussian_diagnostics(
            clean_flat=clean_nonneg,
            noisy_flat=noisy_flat,
            sigma_flat=sigma_flat,
            shot_coeff=shot_coeff,
            sigma_floor=sigma_floor,
            rho_flat=rho_flat,
            global_channel_ids=global_channel_ids,
            line_names=line_names,
            diagnostics_cfg=diagnostics_cfg,
        )
    else:
        meta = {
            'enabled': True,
            'model': 'hetero_gaussian',
            'apply_to': 'train',
            'num_train_pixels': int(clean_flat.shape[0]),
            'num_active_channels': int(clean_flat.shape[1]),
            'diagnostics_enabled': False,
            'channels': [],
        }

    meta['seed'] = int(seed)
    return noisy_flat, sigma_flat, meta

def apply_observation_noise_flat(
    clean_flat: torch.Tensor,
    cfg,
    global_channel_ids=None,
    line_names=None,
    seed: int = 0,
    mask_2d: Union[torch.Tensor, None] = None,
    pix_idx: Union[torch.Tensor, None] = None,
    image_hw: Union[tuple[int, int], None] = None,
) -> tuple[torch.Tensor, Union[torch.Tensor, None], dict | None]:
    '''
    apply noise to all channels
    '''
    noise_cfg = get_observation_noise_cfg(cfg)
    if not noise_cfg.get('enabled', False):
        return clean_flat, None, None
    
    # case on noise model
    if noise_cfg['model'] == 'effective_shot':
        # resolve alpha
        alpha = resolve_effective_shot_alpha(
            noise_cfg=noise_cfg,
            n_channels=int(clean_flat.shape[1]),
            global_channel_ids=global_channel_ids,
        )
        # compute radius for radial bin summaries
        rho_flat = None
        if mask_2d is not None and pix_idx is not None and image_hw is not None:
            H, W = int(image_hw[0]), int(image_hw[1])
            rho_flat = _build_flat_projected_radius_from_mask(
                mask_2d=mask_2d,
                pix_idx=pix_idx,
                H=H,
                W=W,
            )
        # apply noise to all channels
        noisy_flat, meta = apply_effective_shot_noise_flat(
            clean_flat=clean_flat,
            alpha=alpha,
            seed=int(seed),
            rho_flat=rho_flat,
            global_channel_ids=global_channel_ids,
            line_names=line_names,
            diagnostics_cfg=noise_cfg.get('diagnostics', {}),
        )
        meta['effective_shot'] = {
            'alpha_active_channels': [float(v) for v in alpha.detach().cpu().tolist()]
        }
        return noisy_flat, None, meta

    if noise_cfg['model'] == 'hetero_gaussian':
        shot_coeff, sigma_floor = resolve_hetero_gaussian_params(
            noise_cfg=noise_cfg,
            n_channels=int(clean_flat.shape[1]),
            global_channel_ids=global_channel_ids,
        )

        rho_flat = None
        if mask_2d is not None and pix_idx is not None and image_hw is not None:
            H, W = int(image_hw[0]), int(image_hw[1])
            rho_flat = _build_flat_projected_radius_from_mask(
                mask_2d=mask_2d,
                pix_idx=pix_idx,
                H=H,
                W=W,
            )

        noisy_flat, sigma_flat, meta = apply_hetero_gaussian_noise_flat(
            clean_flat=clean_flat,
            shot_coeff=shot_coeff,
            sigma_floor=sigma_floor,
            seed=int(seed),
            rho_flat=rho_flat,
            global_channel_ids=global_channel_ids,
            line_names=line_names,
            diagnostics_cfg=noise_cfg.get('diagnostics', {}),
        )
        meta['hetero_gaussian'] = {
            'shot_coeff_active_channels': [float(v) for v in shot_coeff.detach().cpu().tolist()],
            'sigma_floor_active_channels': [float(v) for v in sigma_floor.detach().cpu().tolist()],
        }
        return noisy_flat, sigma_flat, meta

    raise NotImplementedError(f"Unknown observation_noise.model='{noise_cfg['model']}'")

def log_observation_noise_diagnostics(meta: dict, logger=None) -> None:
    '''
    given meta from adding noise, logs info
    '''
    if meta is None:
        return

    logger = logger or logging.getLogger('coroNeRF.noise')
    logger.info(
        'Observation noise enabled | model=%s | apply_to=%s | train_pixels=%d | active_channels=%d',
        meta.get('model'),
        meta.get('apply_to'),
        int(meta.get('num_train_pixels', 0)),
        int(meta.get('num_active_channels', 0)),
    )

    for ch in meta.get('channels', []):
        rel_noise = ch.get('mean_abs_rel_noise', None)
        snr_p50 = ch.get('theory_snr_percentiles', {}).get('50', None)

        if meta.get('model') == 'effective_shot':
            counts_p50 = ch.get('expected_counts_percentiles', {}).get('50', None)
            zero_frac = ch.get('frac_zero_observed', None)
            logger.info(
                '  noise ch[g=%d,l=%d] %s | alpha=%.3g | counts_p50=%.3g | theory_snr_p50=%.3g | mean_abs_rel=%.3g | frac_zero=%.3g',
                int(ch.get('global_channel', -1)),
                int(ch.get('local_channel', -1)),
                ch.get('line_name', 'unknown'),
                float(ch.get('alpha', 0.0)),
                float(counts_p50 if counts_p50 is not None else 0.0),
                float(snr_p50 if snr_p50 is not None else 0.0),
                float(rel_noise if rel_noise is not None else 0.0),
                float(zero_frac if zero_frac is not None else 0.0),
            )
        elif meta.get('model') == 'hetero_gaussian':
            sigma_p50 = ch.get('sigma_percentiles', {}).get('50', None)
            frac_neg = ch.get('frac_negative_observed', None)
            logger.info(
                '  noise ch[g=%d,l=%d] %s | a=%.3g | b=%.3g | sigma_p50=%.3g | theory_snr_p50=%.3g | mean_abs_rel=%.3g | frac_neg=%.3g',
                int(ch.get('global_channel', -1)),
                int(ch.get('local_channel', -1)),
                ch.get('line_name', 'unknown'),
                float(ch.get('shot_coeff', 0.0)),
                float(ch.get('sigma_floor', 0.0)),
                float(sigma_p50 if sigma_p50 is not None else 0.0),
                float(snr_p50 if snr_p50 is not None else 0.0),
                float(rel_noise if rel_noise is not None else 0.0),
                float(frac_neg if frac_neg is not None else 0.0),
            )