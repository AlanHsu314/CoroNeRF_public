######################################
## abundance model mismatch code
###################################### 


from __future__ import annotations

import json
import logging
import math
from numbers import Number
from pathlib import Path
from typing import Union

import torch

# Reuse helpers already introduced for observation-noise diagnostics.
from .noise import (
    _as_builtin,
    _resolve_global_channel_ids,
    _channel_name,
    _build_flat_projected_radius_from_mask,
    _quantiles,
)

logger = logging.getLogger("coroNeRF.mismatch")


def save_mismatch_json(meta: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w") as f:
        json.dump(_as_builtin(meta), f, indent=2)


def get_observation_mismatch_cfg(cfg) -> dict:
    """
    Deterministic observation/model mismatch config.

    For abundance mismatch, the default should be apply_to='all',
    because train and holdout views are produced by the same true abundance.
    """
    raw = dict(cfg.get("observation_mismatch", {}) or {})

    out = {
        "enabled": bool(raw.get("enabled", False)),
        "apply_to": raw.get("apply_to", "all"),          # all | train | test
        "model": raw.get("model", "channel_scale"),      # channel_scale
        "diagnostics": dict(raw.get("diagnostics", {}) or {}),
        "channel_scale": dict(raw.get("channel_scale", {}) or {}),
    }

    out["diagnostics"].setdefault("enabled", True)
    out["diagnostics"].setdefault("positive_eps", 1.0e-12)
    out["diagnostics"].setdefault("percentiles", [10.0, 50.0, 90.0])
    out["diagnostics"].setdefault("max_quantile_samples", 200000)
    out["diagnostics"].setdefault("radial_bins", [
        {"name": "low_corona", "r_min": 1.01, "r_max": 1.5},
        {"name": "inner", "r_min": 1.01, "r_max": 2.0},
        {"name": "mid_outer_inner", "r_min": 1.5, "r_max": 2.0},
    ])

    out["channel_scale"].setdefault("scale", None)
    out["channel_scale"].setdefault("scale_by_global_channel", None)

    if not out["enabled"]:
        return out

    if out["apply_to"] not in {"all", "train", "test"}:
        raise NotImplementedError(
            f"observation_mismatch.apply_to='{out['apply_to']}' is not implemented; "
            "use one of: all, train, test"
        )

    if out["model"] != "channel_scale":
        raise NotImplementedError(
            f"Unknown observation_mismatch.model='{out['model']}'. "
            "Currently supported: channel_scale"
        )

    cs = out["channel_scale"]
    if cs.get("scale", None) is None and cs.get("scale_by_global_channel", None) is None:
        raise ValueError(
            "observation_mismatch.enabled=True with model=channel_scale requires "
            "either channel_scale.scale or channel_scale.scale_by_global_channel"
        )

    return out


def resolve_channel_scale_factors(
    mismatch_cfg: dict,
    n_channels: int,
    global_channel_ids=None,
) -> torch.Tensor:
    """
    Resolve multiplicative scale factors for active channels.

    channel_scale.scale:
        scalar or active-channel list

    channel_scale.scale_by_global_channel:
        master-channel list indexed by load_data_kwargs.channel_indices
    """
    cs = dict(mismatch_cfg.get("channel_scale", {}) or {})
    scale = cs.get("scale", None)
    scale_by_global_channel = cs.get("scale_by_global_channel", None)

    # checks, must have one
    if scale is not None and scale_by_global_channel is not None:
        raise ValueError(
            "Provide either channel_scale.scale or "
            "channel_scale.scale_by_global_channel, not both"
        )

    # prefer per-channel scales
    if scale_by_global_channel is not None:
        if not isinstance(scale_by_global_channel, (list, tuple)):
            raise TypeError("channel_scale.scale_by_global_channel must be a list/tuple")

        ids = _resolve_global_channel_ids(n_channels, global_channel_ids)
        max_id = max(ids) if ids else -1
        if max_id >= len(scale_by_global_channel):
            raise IndexError(
                f"channel_scale.scale_by_global_channel has length "
                f"{len(scale_by_global_channel)}, but active global channel id "
                f"{max_id} was requested"
            )

        vals = [float(scale_by_global_channel[i]) for i in ids]
    # else fallback to one scale for all channels
    else:
        if isinstance(scale, Number):
            vals = [float(scale)] * int(n_channels)

        elif isinstance(scale, (list, tuple)):
            vals = [float(v) for v in scale]
            if len(vals) != int(n_channels):
                raise ValueError(
                    f"channel_scale.scale has length {len(vals)}, "
                    f"expected {n_channels} active channels"
                )

        else:
            raise TypeError(
                "channel_scale.scale must be either a positive scalar or a "
                "list/tuple matching the active channel count"
            )

    bad = [float(v) for v in vals if float(v) <= 0.0]
    if bad:
        raise ValueError(f"channel_scale values must be positive, got {bad}")

    return torch.tensor(vals, dtype=torch.float32)


def _pct_dict(percentiles: list[float], values: list[float]) -> dict:
    return {
        str(int(p) if float(p).is_integer() else p): float(v)
        for p, v in zip(percentiles, values)
    }


def _masked_flat_from_image_cube(
    imgs: torch.Tensor,
    mask_2d: Union[torch.Tensor, None],
) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """
    imgs: (V, H, W, C)
    mask_2d: (H, W), True where corona pixels are used
    """
    if imgs.ndim != 4:
        raise ValueError(f"imgs must have shape (V, H, W, C), got {tuple(imgs.shape)}")

    V, H, W, C = imgs.shape

    if mask_2d is None:
        return imgs.reshape(-1, C), None

    if tuple(mask_2d.shape) != (H, W):
        raise ValueError(
            f"mask_2d shape {tuple(mask_2d.shape)} does not match image shape {(H, W)}"
        )

    keep = mask_2d.bool().view(1, H, W, 1).expand(V, H, W, 1).contiguous()
    pix_idx = keep.view(-1).nonzero().squeeze(-1)
    flat = imgs.reshape(-1, C)[pix_idx.to(imgs.device)]

    return flat, pix_idx


def _summarize_channel_scale_subset(
    nominal_c: torch.Tensor,
    observed_c: torch.Tensor,
    scale_c: float,
    percentiles: list[float],
    positive_eps: float,
    max_quantile_samples: int,
) -> dict:
    '''
    summarize mismatch stats helper
    '''
    delta_c = observed_c - nominal_c

    positive = nominal_c > positive_eps
    nominal_pos = nominal_c[positive]
    observed_pos = observed_c[positive]
    delta_pos = delta_c[positive]

    nominal_q = _quantiles(nominal_pos, percentiles, max_samples=max_quantile_samples)
    observed_q = _quantiles(observed_pos, percentiles, max_samples=max_quantile_samples)

    if nominal_c.numel() > 0:
        delta_rmse = float(torch.sqrt(torch.mean(delta_c.pow(2))).item())
        nominal_rms = float(torch.sqrt(torch.mean(nominal_c.pow(2))).item())
        delta_rel_rmse = float(delta_rmse / max(nominal_rms, positive_eps))
    else:
        delta_rmse = 0.0
        delta_rel_rmse = 0.0

    if nominal_pos.numel() > 0:
        mean_abs_rel_delta = float(
            torch.mean(torch.abs(delta_pos) / nominal_pos.clamp_min(positive_eps)).item()
        )
        mean_rel_delta = float(
            torch.mean(delta_pos / nominal_pos.clamp_min(positive_eps)).item()
        )
        nominal_mean_pos = float(torch.mean(nominal_pos).item())
        observed_mean_pos = float(torch.mean(observed_pos).item())
    else:
        mean_abs_rel_delta = None
        mean_rel_delta = None
        nominal_mean_pos = 0.0
        observed_mean_pos = 0.0

    frac_nonpositive_nominal = (
        float(torch.mean((nominal_c <= positive_eps).float()).item())
        if nominal_c.numel() > 0 else 0.0
    )
    frac_nonpositive_observed = (
        float(torch.mean((observed_c <= positive_eps).float()).item())
        if observed_c.numel() > 0 else 0.0
    )

    return {
        "num_pixels": int(nominal_c.numel()),
        "num_positive_pixels": int(nominal_pos.numel()),
        "nominal_mean_positive": nominal_mean_pos,
        "observed_mean_positive": observed_mean_pos,
        "nominal_percentiles": _pct_dict(percentiles, nominal_q),
        "observed_percentiles": _pct_dict(percentiles, observed_q),
        "delta_rmse": delta_rmse,
        "delta_rel_rmse": delta_rel_rmse,
        "mean_abs_rel_delta": mean_abs_rel_delta,
        "mean_rel_delta": mean_rel_delta,
        "frac_nonpositive_nominal": frac_nonpositive_nominal,
        "frac_nonpositive_observed": frac_nonpositive_observed,
        "scale_factor": float(scale_c),
        "dlog_intensity_shift": float(math.log10(float(scale_c))),
    }


def _build_channel_scale_diagnostics(
    nominal_flat: torch.Tensor,
    observed_flat: torch.Tensor,
    scale: torch.Tensor,
    rho_flat: Union[torch.Tensor, None] = None,
    global_channel_ids=None,
    line_names=None,
    diagnostics_cfg=None,
    split: str = "train",
) -> dict:
    diagnostics_cfg = dict(diagnostics_cfg or {})
    percentiles = [float(p) for p in diagnostics_cfg.get("percentiles", [10.0, 50.0, 90.0])]
    positive_eps = float(diagnostics_cfg.get("positive_eps", 1.0e-12))
    max_quantile_samples = int(diagnostics_cfg.get("max_quantile_samples", 200000))
    radial_bins_cfg = list(diagnostics_cfg.get("radial_bins", []) or [])

    n_channels = int(nominal_flat.shape[1])
    global_ids = _resolve_global_channel_ids(n_channels, global_channel_ids)

    if rho_flat is not None:
        rho_flat = rho_flat.to(device=nominal_flat.device)

    channels = []
    for local_idx in range(n_channels):
        global_idx = int(global_ids[local_idx])
        line_name = _channel_name(local_idx, global_idx, line_names=line_names)

        nominal_c = nominal_flat[:, local_idx]
        observed_c = observed_flat[:, local_idx]
        scale_c = float(scale[local_idx].item())

        overall_stats = _summarize_channel_scale_subset(
            nominal_c=nominal_c,
            observed_c=observed_c,
            scale_c=scale_c,
            percentiles=percentiles,
            positive_eps=positive_eps,
            max_quantile_samples=max_quantile_samples,
        )

        radial_bin_stats = []
        if rho_flat is not None and radial_bins_cfg:
            for bin_cfg in radial_bins_cfg:
                name = str(bin_cfg.get("name", "unnamed"))
                r_min = bin_cfg.get("r_min", None)
                r_max = bin_cfg.get("r_max", None)

                select = torch.ones_like(rho_flat, dtype=torch.bool)
                if r_min is not None:
                    select = select & (rho_flat >= float(r_min))
                if r_max is not None:
                    select = select & (rho_flat < float(r_max))

                bin_stats = _summarize_channel_scale_subset(
                    nominal_c=nominal_c[select],
                    observed_c=observed_c[select],
                    scale_c=scale_c,
                    percentiles=percentiles,
                    positive_eps=positive_eps,
                    max_quantile_samples=max_quantile_samples,
                )
                bin_stats["name"] = name
                bin_stats["r_min"] = None if r_min is None else float(r_min)
                bin_stats["r_max"] = None if r_max is None else float(r_max)
                radial_bin_stats.append(bin_stats)

        channels.append({
            "local_channel": int(local_idx),
            "global_channel": int(global_idx),
            "line_name": line_name,
            **overall_stats,
            "radial_bins": radial_bin_stats,
        })

    return {
        "enabled": True,
        "model": "channel_scale",
        "split": split,
        "num_pixels": int(nominal_flat.shape[0]),
        "num_active_channels": int(n_channels),
        "diagnostics_percentiles": percentiles,
        "diagnostics_max_quantile_samples": int(max_quantile_samples),
        "diagnostics_radial_bins": radial_bins_cfg,
        "channels": channels,
    }


def apply_channel_scale_mismatch_flat(
    nominal_flat: torch.Tensor,
    scale: torch.Tensor,
    rho_flat: Union[torch.Tensor, None] = None,
    global_channel_ids=None,
    line_names=None,
    diagnostics_cfg=None,
    split: str = "train",
) -> tuple[torch.Tensor, dict]:
    """
    Apply deterministic multiplicative model mismatch to flattened intensities.
    """
    if nominal_flat.ndim != 2:
        raise ValueError(
            f"nominal_flat must have shape (N, C), got {tuple(nominal_flat.shape)}"
        )

    scale = scale.to(device=nominal_flat.device, dtype=nominal_flat.dtype)
    if scale.ndim != 1 or scale.shape[0] != nominal_flat.shape[1]:
        raise ValueError(
            f"scale must have shape ({nominal_flat.shape[1]},), got {tuple(scale.shape)}"
        )

    observed_flat = nominal_flat * scale.view(1, -1)

    diagnostics_cfg = dict(diagnostics_cfg or {})
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", True))

    if diagnostics_enabled:
        meta = _build_channel_scale_diagnostics(
            nominal_flat=nominal_flat,
            observed_flat=observed_flat,
            scale=scale,
            rho_flat=rho_flat,
            global_channel_ids=global_channel_ids,
            line_names=line_names,
            diagnostics_cfg=diagnostics_cfg,
            split=split,
        )
    else:
        n_channels = int(nominal_flat.shape[1])
        global_ids = _resolve_global_channel_ids(n_channels, global_channel_ids)
        meta = {
            "enabled": True,
            "model": "channel_scale",
            "split": split,
            "num_pixels": int(nominal_flat.shape[0]),
            "num_active_channels": n_channels,
            "diagnostics_enabled": False,
            "channels": [
                {
                    "local_channel": int(local_idx),
                    "global_channel": int(global_ids[local_idx]),
                    "line_name": _channel_name(
                        local_idx,
                        int(global_ids[local_idx]),
                        line_names=line_names,
                    ),
                    "scale_factor": float(scale[local_idx].item()),
                    "dlog_intensity_shift": float(math.log10(float(scale[local_idx].item()))),
                }
                for local_idx in range(n_channels)
            ],
        }

    meta["channel_scale"] = {
        "scale_active_channels": [float(v) for v in scale.detach().cpu().tolist()],
    }

    return observed_flat, meta


def _split_is_enabled(apply_to: str, split: str) -> bool:
    return apply_to == "all" or apply_to == split


def apply_observation_mismatch_images(
    imgs: torch.Tensor,
    cfg,
    split: str,
    global_channel_ids=None,
    line_names=None,
    mask_2d: Union[torch.Tensor, None] = None,
) -> tuple[torch.Tensor, Union[dict, None]]:
    """
    Apply deterministic mismatch to a full image cube.

    imgs: (V, H, W, C)

    For abundance mismatch this should usually be called for both train and test.
    """
    mismatch_cfg = get_observation_mismatch_cfg(cfg)
    if not mismatch_cfg.get("enabled", False):
        return imgs, None

    if not _split_is_enabled(str(mismatch_cfg["apply_to"]), split):
        return imgs, None

    if imgs.ndim != 4:
        raise ValueError(f"imgs must have shape (V, H, W, C), got {tuple(imgs.shape)}")

    V, H, W, C = imgs.shape

    scale = resolve_channel_scale_factors(
        mismatch_cfg=mismatch_cfg,
        n_channels=int(C),
        global_channel_ids=global_channel_ids,
    ).to(device=imgs.device, dtype=imgs.dtype)

    observed_imgs = imgs * scale.view(1, 1, 1, -1)

    nominal_flat, pix_idx = _masked_flat_from_image_cube(imgs, mask_2d)
    rho_flat = None

    diagnostics_cfg = dict(mismatch_cfg.get("diagnostics", {}) or {})
    radial_bins_cfg = list(diagnostics_cfg.get("radial_bins", []) or [])
    if mask_2d is not None and pix_idx is not None and radial_bins_cfg:
        rho_flat = _build_flat_projected_radius_from_mask(
            mask_2d=mask_2d,
            pix_idx=pix_idx,
            H=int(H),
            W=int(W),
        )

    _observed_flat, meta = apply_channel_scale_mismatch_flat(
        nominal_flat=nominal_flat,
        scale=scale,
        rho_flat=rho_flat,
        global_channel_ids=global_channel_ids,
        line_names=line_names,
        diagnostics_cfg=diagnostics_cfg,
        split=split,
    )

    meta["apply_to"] = str(mismatch_cfg["apply_to"])
    meta["num_views"] = int(V)
    meta["image_H"] = int(H)
    meta["image_W"] = int(W)

    return observed_imgs.contiguous(), meta


def combine_observation_mismatch_metadata(
    split_metas: dict[str, Union[dict, None]],
    cfg,
) -> Union[dict, None]:
    split_metas = {str(k): v for k, v in split_metas.items() if v is not None}
    if not split_metas:
        return None

    mismatch_cfg = get_observation_mismatch_cfg(cfg)
    first_meta = next(iter(split_metas.values()))

    channels = []
    for ch in first_meta.get("channels", []):
        channels.append({
            "local_channel": int(ch.get("local_channel", -1)),
            "global_channel": int(ch.get("global_channel", -1)),
            "line_name": ch.get("line_name", "unknown"),
            "scale_factor": float(ch.get("scale_factor", 1.0)),
            "dlog_intensity_shift": float(ch.get("dlog_intensity_shift", 0.0)),
        })

    return {
        "enabled": True,
        "model": str(mismatch_cfg.get("model", "channel_scale")),
        "apply_to": str(mismatch_cfg.get("apply_to", "all")),
        "num_active_channels": int(first_meta.get("num_active_channels", len(channels))),
        "splits_present": sorted(split_metas.keys()),
        "channels": channels,
        "channel_scale": dict(first_meta.get("channel_scale", {}) or {}),
        "splits": split_metas,
    }


def log_observation_mismatch_diagnostics(meta: dict, logger=None) -> None:
    if meta is None:
        return

    logger = logger or logging.getLogger("coroNeRF.mismatch")
    logger.info(
        "Observation mismatch enabled | model=%s | apply_to=%s | splits=%s | active_channels=%d",
        meta.get("model"),
        meta.get("apply_to"),
        ",".join(meta.get("splits_present", [])),
        int(meta.get("num_active_channels", 0)),
    )

    for ch in meta.get("channels", []):
        logger.info(
            "  mismatch ch[g=%d,l=%d] %s | scale=%.4g | dlog=%.4g",
            int(ch.get("global_channel", -1)),
            int(ch.get("local_channel", -1)),
            ch.get("line_name", "unknown"),
            float(ch.get("scale_factor", 1.0)),
            float(ch.get("dlog_intensity_shift", 0.0)),
        )

    for split_name, split_meta in meta.get("splits", {}).items():
        logger.info(
            "  mismatch split=%s | views=%s | masked_pixels=%d",
            split_name,
            split_meta.get("num_views", "unknown"),
            int(split_meta.get("num_pixels", 0)),
        )