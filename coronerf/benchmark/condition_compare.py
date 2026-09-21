##########################################################
### post-training benchmark comparator for run conditions
##########################################################

from __future__ import annotations

from pathlib import Path
from typing import Any, Union
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch

from .aggregate import collect_run_summaries
from .config import load_yaml, save_json, normalize_config
from .naming import short_hash, sanitize_value, shorten_key
from .summary import find_latest_checkpoint, find_final_eval_json

from .. import ExperimentDirs, DotDict
from ..experiments.resolve_paths import resolve_cfg_paths, remap_cfg_paths
from ..pipeline.state import PipelineState
from ..pipeline.builders.data import load_data
from ..pipeline.builders.fields import build_fields
from ..pipeline.builders.model import build_model
from ..pipeline.builders.renderer import build_renderer
from ..eval.run_eval import load_ckpt, render_view
from ..artifacts.scalar_fields import build_scalar_shell_product

logger = logging.getLogger('coroNeRF.benchmark.condition_compare')

def _read_json(path: Path) -> dict:
    with path.open('r') as f:
        return json.load(f)

def _resolve_run_dir(row: dict, benchmark_dir: Path) -> Path | None:
    '''
    if run_dir_raw does not exist, create one that works relative to benchmark dir
    '''
    run_dir_raw = row.get('run_dir')
    run_name = row.get('run_name')

    if run_dir_raw:
        p = Path(run_dir_raw)
        if p.exists():
            return p

    if run_name:
        p = benchmark_dir / 'runs' / run_name
        if p.exists():
            return p

    return None

def _freeze_group_value(value):
    if isinstance(value, list):
        return tuple(_freeze_group_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_group_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_group_value(v)) for k, v in value.items()))
    return value

def _metric_value(row: dict, key: str) -> float:
    val = row.get(key, None)
    if val is None:
        raise KeyError(f'metric {key} missing from row {row.get("run_name")}')
    return float(val)

def _condition_key(row: dict) -> tuple:
    sweep_items = []
    for k in sorted(row.keys()):
        if k.startswith('sweep.'):
            sweep_items.append((k, _freeze_group_value(row.get(k))))
    return (row.get('experiment_name'), tuple(sweep_items))

def _condition_display_name(row: dict) -> str:
    parts = [str(row.get('experiment_name', 'unknown'))]
    sweep_items = [(k, row.get(k)) for k in sorted(row.keys()) if k.startswith('sweep.')]
    for key, value in sweep_items:
        parts.append(f'{shorten_key(key)}={sanitize_value(value)}')
    return ' | '.join(parts)

def _row_folder_name(row: dict) -> str:
    base = str(row.get('experiment_name', 'compare'))
    digest = short_hash({
        'run_name': row.get('run_name'),
        'condition': _condition_key(row),
    }, n=8)
    return f'{base}__{digest}'

def _select_best_run_per_condition(
    rows: list[dict],
    metric_key: str,
    mode: str = 'min',
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}

    for row in rows:
        if row.get('status') != 'completed':
            continue
        if row.get(metric_key) is None:
            continue
        grouped.setdefault(_condition_key(row), []).append(row)

    best_rows = []
    for _, group in grouped.items():
        if mode == 'min':
            best_row = min(group, key=lambda r: _metric_value(r, metric_key))
        else:
            best_row = max(group, key=lambda r: _metric_value(r, metric_key))
        best_rows.append(best_row)

    best_rows.sort(key=lambda r: (_metric_value(r, metric_key), str(r.get('run_name'))))
    return best_rows

def _resolve_reference_row(
    completed_rows: list[dict],
    best_condition_rows: list[dict],
    cfg: dict,
    metric_key: str,
    mode: str,
) -> dict:
    reference_run_name = cfg.get('reference_run_name', None)
    if reference_run_name:
        matches = [r for r in completed_rows if r.get('run_name') == reference_run_name and r.get('status') == 'completed']
        if not matches:
            raise ValueError(f'Could not find completed reference_run_name={reference_run_name}')
        return matches[0]

    reference_experiment = cfg.get('reference_experiment', None)
    if reference_experiment is None:
        raise ValueError('post_condition_compare requires reference_experiment or reference_run_name')

    matches = [r for r in best_condition_rows if r.get('experiment_name') == reference_experiment]
    if not matches:
        raise ValueError(f'No completed conditions found for reference_experiment={reference_experiment}')

    if mode == 'min':
        return min(matches, key=lambda r: _metric_value(r, metric_key))
    return max(matches, key=lambda r: _metric_value(r, metric_key))

def _resolve_comparison_rows(
    completed_rows: list[dict],
    best_condition_rows: list[dict],
    reference_row: dict,
    cfg: dict,
    metric_key: str,
    mode: str,
) -> list[dict]:
    comparison_run_names = cfg.get('comparison_run_names', None)
    if comparison_run_names:
        out = []
        for run_name in comparison_run_names:
            matches = [r for r in completed_rows if r.get('run_name') == run_name and r.get('status') == 'completed']
            if not matches:
                raise ValueError(f'Could not find completed comparison_run_name={run_name}')
            out.append(matches[0])
        return out

    reference_experiment = reference_row.get('experiment_name')
    comparison_experiments = cfg.get('comparison_experiments', None)
    if comparison_experiments is None:
        comparison_experiments = sorted({
            r.get('experiment_name') for r in best_condition_rows
            if r.get('experiment_name') != reference_experiment
        })

    max_per_exp = cfg.get('max_conditions_per_experiment', 1)
    if max_per_exp is not None:
        max_per_exp = int(max_per_exp)

    out = []
    for exp_name in comparison_experiments:
        matches = [r for r in best_condition_rows if r.get('experiment_name') == exp_name]
        if not matches:
            continue

        matches = sorted(matches, key=lambda r: _metric_value(r, metric_key), reverse=(mode == 'max'))
        if max_per_exp is not None:
            matches = matches[:max_per_exp]
        out.extend(matches)

    # drop exact duplicate of reference if user asked for same experiment
    out = [r for r in out if r.get('run_name') != reference_row.get('run_name')]
    return out

def _choose_view_indices(num_views_total: int, num_views_plot: int, strategy: str = 'evenly_spaced') -> list[int]:
    if num_views_total <= 0:
        return []
    if num_views_plot >= num_views_total:
        return list(range(num_views_total))
    if strategy != 'evenly_spaced':
        raise NotImplementedError(f'Unknown view selection strategy: {strategy}')

    xs = np.linspace(0, num_views_total, num=num_views_plot, endpoint=False)
    idx = np.floor(xs).astype(int)
    idx = np.unique(idx)
    while len(idx) < num_views_plot:
        missing = [i for i in range(num_views_total) if i not in set(idx.tolist())]
        idx = np.concatenate([idx, np.asarray(missing[:(num_views_plot - len(idx))], dtype=int)])
    return idx.astype(int).tolist()

def _apply_cfg_overrides(raw_cfg: dict, overrides: dict | None) -> dict:
    """Apply dotted-key overrides (e.g. {'load_data_kwargs.channel_indices': [0,1,2,3]}) onto a raw cfg dict, in place."""
    if not overrides:
        return raw_cfg
    for dotted, val in overrides.items():
        node = raw_cfg
        parts = str(dotted).split('.')
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = val
    return raw_cfg

def _build_state_for_analysis(run_dir: Path, 
                              device_override: str | None = None,
                              path_remap: dict | None = None,
                              cfg_overrides: dict | None = None,
                              ) -> PipelineState:
    raw_cfg = load_yaml(run_dir / 'config.yaml')
    raw_cfg = normalize_config(raw_cfg)

    # remap cfg, if doing things locally
    raw_cfg = remap_cfg_paths(raw_cfg, path_remap = path_remap, root=run_dir)

    # analysis-time overrides (e.g. global-eval channel set / dataset) applied before data + renderer build
    raw_cfg = _apply_cfg_overrides(raw_cfg, cfg_overrides)

    dataset_dir = raw_cfg.get('dataset_dir', None)
    if isinstance(dataset_dir, str) and not Path(dataset_dir).is_absolute():
        raw_cfg = resolve_cfg_paths(raw_cfg, root=run_dir)

    raw_cfg.setdefault('io', {})
    raw_cfg['io']['no_save'] = True
    if device_override is not None:
        raw_cfg['device'] = str(device_override)

    logger.info(
        'analysis paths | run=%s | dataset_dir=%s | LUT_dir=%s',
        run_dir.name,
        raw_cfg.get('dataset_dir', None),
        raw_cfg.get('LUT_dir', None),
    )

    cfg = DotDict(raw_cfg)
    if cfg.mode not in ['train', 'resume']:
        cfg.mode = 'train'

    dirs = ExperimentDirs(
        run_dir=run_dir,
        checkpoints=run_dir / 'checkpoints',
        logs=run_dir / 'logs',
        figs=run_dir / 'figs',
        artifacts=run_dir / 'artifacts',
    )

    state = PipelineState(cfg, dirs=dirs, wb_run=None)
    load_data(state, **state.cfg.load_data_kwargs)
    build_fields(state)
    build_model(state)
    build_renderer(state)

    ckpt_path = find_latest_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f'No checkpoint found for run_dir={run_dir}')
    load_ckpt(state, ckpt_path)
    return state

def _release_state(state: PipelineState | None) -> None:
    if state is None:
        return
    try:
        if hasattr(state, 'model') and state.model is not None:
            del state.model
        if hasattr(state, 'renderer') and state.renderer is not None:
            del state.renderer
        if hasattr(state, 'optimizer') and state.optimizer is not None:
            del state.optimizer
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _channel_labels(state: PipelineState) -> list[str]:
    names = getattr(state, 'selected_line_names', None)
    if names:
        return [str(x) for x in names]
    c = int(state.dataset_num_channels)
    return [f'ch_{i}' for i in range(c)]

def _reconstruct_flat_targets_to_imgs(
    imgs_flat: torch.Tensor,
    pix_idx: torch.Tensor,
    V: int,
    H: int,
    W: int,
    C: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    flat = torch.full((V * H * W, C), float(fill_value), dtype=imgs_flat.dtype)
    flat[pix_idx.detach().cpu()] = imgs_flat.detach().cpu()
    return flat.view(V, H, W, C).numpy()

def _get_reference_observation_cube(state: PipelineState, split: str) -> np.ndarray:
    if split == 'train':
        return state.train_data['imgs'].detach().cpu().numpy()
    if split == 'test':
        return state.test_data['imgs'].detach().cpu().numpy()
    raise ValueError(f'Unknown split={split}')

def _get_comparison_observation_cube(state: PipelineState, split: str) -> np.ndarray:
    if split == 'train':
        '''
        if stochastic noise was applied, actual train targets live only in ds_train.imgs_flat
        mismatch already in state.train_data['imgs'] before CoronaDataset is created
        '''
        noise_meta = getattr(state, 'observation_noise_meta', None)
        if noise_meta is not None:
            V, H, W, C = state.train_data['imgs'].shape
            return _reconstruct_flat_targets_to_imgs(
                imgs_flat=state.ds_train.imgs_flat,
                pix_idx=state.ds_train.pix_idx,
                V=int(V),
                H=int(H),
                W=int(W),
                C=int(C),
            )
        return state.train_data['imgs'].detach().cpu().numpy()

    if split == 'test':
        # model mismatch is deterministic and is reflected in holdout targets
        return state.test_data['imgs'].detach().cpu().numpy()

    raise ValueError(f'Unknown split={split}')

@torch.no_grad()
def _render_selected_views(
    state: PipelineState,
    split: str,
    view_indices: list[int],
    chunk_size: int = 8192,
) -> np.ndarray:
    if split == 'train':
        data = state.train_data
    elif split == 'test':
        data = state.test_data
    else:
        raise ValueError(f'Unknown split={split}')

    rays_o = data['rays_o']
    rays_d = data['rays_d']

    preds = []
    for v in view_indices:
        pred = render_view(state, rays_o[v], rays_d[v], chunk_size=chunk_size)
        preds.append(pred)

    preds = np.asarray(preds, dtype=np.float32)

    # apply the same coronal mask used by the dataset images
    if split == 'train':
        mask_2d = state.train_data['mask']
    else:
        mask_2d = state.test_data['mask']

    preds = _mask_image_cube(preds, mask_2d, fill_value=np.nan)
    return preds

def _slice_views(cube: np.ndarray, view_indices: list[int]) -> np.ndarray:
    return np.asarray(cube[view_indices], dtype=np.float32)

def _mask_image_cube(cube: np.ndarray, mask_2d: torch.Tensor | np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """
    Apply a 2D coronal mask to a (V,H,W,C) cube.
    """
    mask_np = mask_2d.detach().cpu().numpy() if isinstance(mask_2d, torch.Tensor) else np.asarray(mask_2d)
    mask_np = mask_np.astype(bool)

    out = np.array(cube, copy=True)
    out[:, ~mask_np, :] = fill_value
    return out

def _panel_limits(
    panel_cubes: dict[str, np.ndarray],
    channel_idx: int,
    diff_panel_names: set[str],
    eps: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    base_vals = []
    diff_vals = []

    for name, cube in panel_cubes.items():
        arr = cube[..., channel_idx]
        finite = np.isfinite(arr)
        if not np.any(finite):
            continue

        vals = arr[finite]
        if name in diff_panel_names:
            diff_vals.append(np.abs(vals))
        else:
            base_vals.append(np.log10(np.maximum(vals, eps)))

    if base_vals:
        base_cat = np.concatenate(base_vals)
        vmin = float(np.nanpercentile(base_cat, 1.0))
        vmax = float(np.nanpercentile(base_cat, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(base_cat)), float(np.nanmax(base_cat))
    else:
        vmin, vmax = 0.0, 1.0

    if diff_vals:
        diff_cat = np.concatenate(diff_vals)
        dlim = float(np.nanpercentile(diff_cat, 99.0))
        if not np.isfinite(dlim) or dlim <= 0:
            dlim = float(np.nanmax(diff_cat)) if diff_cat.size else 1.0
        dlim = max(dlim, 1e-6)
    else:
        dlim = 1.0

    return (vmin, vmax), (-dlim, dlim)

def plot_multiview_channel_panels(
    panel_cubes: dict[str, np.ndarray],
    channel_labels: list[str],
    view_indices: list[int],
    out_dir: Path,
    prefix: str,
    diff_panel_names: set[str] | None = None,
    log_images: bool = True,
    eps: float = 1.0e-12,
) -> None:
    diff_panel_names = diff_panel_names or set()
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_names = list(panel_cubes.keys())
    if not panel_names:
        return

    n_rows = len(view_indices)
    n_cols = len(panel_names)
    if n_rows == 0:
        return

    for c, label in enumerate(channel_labels):
        (vmin, vmax), (dvmin, dvmax) = _panel_limits(panel_cubes, c, diff_panel_names, eps)

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=(4.4 * n_cols + 1.2, 3.0 * n_rows),
            squeeze=False,
        )
        fig.subplots_adjust(left=0.06, right=0.88, bottom=0.04, top=0.95, wspace=0.04, hspace=0.08)

        im_base = None
        im_diff = None

        inferno_cmap = plt.get_cmap('inferno').copy()
        inferno_cmap.set_bad('black')

        diff_cmap = plt.get_cmap('coolwarm').copy()
        diff_cmap.set_bad('black')

        for r in range(n_rows):
            for j, name in enumerate(panel_names):
                ax = axs[r, j]
                img = panel_cubes[name][r, ..., c]

                if name in diff_panel_names:
                    im = ax.imshow(img, cmap=diff_cmap, vmin=dvmin, vmax=dvmax)
                    if im_diff is None:
                        im_diff = im
                else:
                    disp = np.log10(np.maximum(img, eps)) if log_images else img
                    im = ax.imshow(disp, cmap=inferno_cmap, vmin=vmin, vmax=vmax)
                    if im_base is None:
                        im_base = im

                if r == 0:
                    ax.set_title(name)
                if j == 0:
                    ax.set_ylabel(f'view {view_indices[r]}')
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f'{prefix} | {label}', y=0.995)

        # fixed colorbar axes on the far right
        if im_base is not None:
            cax_base = fig.add_axes([0.895, 0.18, 0.015, 0.64])
            cbar_base = fig.colorbar(im_base, cax=cax_base)
            cbar_base.set_label('log10 intensity' if log_images else 'intensity')

        if im_diff is not None:
            cax_diff = fig.add_axes([0.935, 0.18, 0.015, 0.64])
            cbar_diff = fig.colorbar(im_diff, cax=cax_diff)
            cbar_diff.set_label('prediction - clean')

        fig.savefig(out_dir / f'{prefix}_ch{c}.png', dpi=150)
        plt.close(fig)

def _load_final_eval_report(run_dir: Path) -> dict | None:
    path = find_final_eval_json(run_dir)
    if path is None:
        return None
    return _read_json(path)

def _extract_shell_series(report: dict | None, quantity: str, metric_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    if report is None:
        return None
    field = report.get('scalar_field_eval', {}).get(quantity, {})
    rows = field.get('shells', [])
    if not rows:
        return None

    xs, ys = [], []
    for row in rows:
        r = row.get('r', None)
        y = row.get(metric_key, None)
        if r is None or y is None:
            continue
        xs.append(float(r))
        ys.append(float(y))

    if not xs:
        return None
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

def plot_shell_metric_compare(
    ref_report: dict | None,
    cmp_report: dict | None,
    quantity: str,
    metric_key: str,
    out_path: Path,
    label_ref: str = 'clean',
    label_cmp: str = 'compare',
) -> None:
    ref_series = _extract_shell_series(ref_report, quantity, metric_key)
    cmp_series = _extract_shell_series(cmp_report, quantity, metric_key)
    if ref_series is None or cmp_series is None:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(ref_series[0], ref_series[1], label=label_ref)
    plt.plot(cmp_series[0], cmp_series[1], label=label_cmp)
    plt.xlabel('Radius [Rsun]')
    plt.ylabel(metric_key)
    plt.title(f'{quantity} shell {metric_key}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _collect_shell_scatter_samples(
    state: PipelineState,
    quantity: str,
    radii: list[float],
    max_points_per_shell: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    gt_all = []
    pred_all = []

    for r in radii:
        product = build_scalar_shell_product(state, quantity=quantity, r_target=float(r))
        gt = product.gt_log.reshape(-1)
        pred = product.pred_log.reshape(-1)

        ok = np.isfinite(gt) & np.isfinite(pred)
        gt = gt[ok]
        pred = pred[ok]
        if gt.size == 0:
            continue

        if max_points_per_shell is not None and gt.size > int(max_points_per_shell):
            idx = rng.choice(gt.size, size=int(max_points_per_shell), replace=False)
            gt = gt[idx]
            pred = pred[idx]

        gt_all.append(gt)
        pred_all.append(pred)

    if not gt_all:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    return np.concatenate(gt_all), np.concatenate(pred_all)

def plot_scalar_scatter_compare(
    gt_ref: np.ndarray,
    pred_ref: np.ndarray,
    gt_cmp: np.ndarray,
    pred_cmp: np.ndarray,
    quantity: str,
    out_path: Path,
    label_cmp: str = 'compare',
) -> None:
    if gt_ref.size == 0 or gt_cmp.size == 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    lim_min = float(min(np.nanmin(gt_ref), np.nanmin(pred_ref), np.nanmin(gt_cmp), np.nanmin(pred_cmp)))
    lim_max = float(max(np.nanmax(gt_ref), np.nanmax(pred_ref), np.nanmax(gt_cmp), np.nanmax(pred_cmp)))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
    axs = axs[0]

    axs[0].scatter(gt_ref, pred_ref, s=1, alpha=0.08)
    axs[0].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1)
    axs[0].set_title('clean')
    axs[0].set_xlabel(f'GT log {quantity}')
    axs[0].set_ylabel(f'Pred log {quantity}')
    axs[0].grid(True, alpha=0.3)

    axs[1].scatter(gt_cmp, pred_cmp, s=1, alpha=0.08)
    axs[1].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1)
    axs[1].set_title(label_cmp)
    axs[1].set_xlabel(f'GT log {quantity}')
    axs[1].set_ylabel(f'Pred log {quantity}')
    axs[1].grid(True, alpha=0.3)

    fig.suptitle(f'{quantity} shell scatter')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def generate_condition_compare_plots(
    benchmark_dir: Union[Path, str],
    cfg: dict | None = None,
) -> dict:
    benchmark_dir = Path(benchmark_dir)
    cfg = cfg or {}

    # cfg params
    metric_key = cfg.get('select_by', 'final/temp_shell_low_corona/shell_mean_abs_dlog_mean')
    mode = cfg.get('select_mode', 'min')
    out_dir = benchmark_dir / cfg.get('out_subdir', 'condition_compare')
    device_override = cfg.get('device', None)

    # load run summaries in rows
    rows = collect_run_summaries(benchmark_dir)
    completed_rows = [r for r in rows if r.get('status') == 'completed']
    if not completed_rows:
        return {}

    # load reference and comparison runs of interest
    best_condition_rows = _select_best_run_per_condition(completed_rows, metric_key=metric_key, mode=mode)
    reference_row = _resolve_reference_row(completed_rows, best_condition_rows, cfg, metric_key, mode)
    comparison_rows = _resolve_comparison_rows(completed_rows, best_condition_rows, reference_row, cfg, metric_key, mode)
    if not comparison_rows:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)

    # load reference run once
    ref_run_dir = _resolve_run_dir(reference_row, benchmark_dir)
    if ref_run_dir is None:
        raise FileNotFoundError(f'Could not resolve run dir for reference row {reference_row.get("run_name")}')

    ref_state = _build_state_for_analysis(ref_run_dir, device_override=device_override)
    channel_labels = _channel_labels(ref_state)

    train_cfg = dict(cfg.get('train_panels', {}) or {})
    test_cfg = dict(cfg.get('test_panels', {}) or {})
    shell_cfg = dict(cfg.get('shell_compare', {}) or {})
    scatter_cfg = dict(cfg.get('scatter', {}) or {})

    train_view_indices = _choose_view_indices(
        num_views_total=int(ref_state.train_data['imgs'].shape[0]),
        num_views_plot=int(train_cfg.get('num_views', 5)),
        strategy=train_cfg.get('strategy', 'evenly_spaced'),
    ) if train_cfg.get('enabled', True) else []

    test_view_indices = _choose_view_indices(
        num_views_total=int(ref_state.test_data['imgs'].shape[0]),
        num_views_plot=int(test_cfg.get('num_views', 5)),
        strategy=test_cfg.get('strategy', 'evenly_spaced'),
    ) if test_cfg.get('enabled', True) else []

    ref_artifacts = {
        'channel_labels': channel_labels,
        'train_obs': _slice_views(_get_reference_observation_cube(ref_state, 'train'), train_view_indices) if train_view_indices else None,
        'test_obs': _slice_views(_get_reference_observation_cube(ref_state, 'test'), test_view_indices) if test_view_indices else None,
        'test_pred': _render_selected_views(
            ref_state, 'test', test_view_indices, chunk_size=int(test_cfg.get('chunk_size', 8192))
        ) if (test_cfg.get('enabled', True) and test_cfg.get('use_reference_prediction', True) and test_view_indices) else None,
        'final_eval': _load_final_eval_report(ref_run_dir),
    }

    if scatter_cfg.get('enabled', True):
        radii = scatter_cfg.get('radii', ref_state.cfg.get('eval', {}).get('shell_radii', []))
        ref_artifacts['scatter'] = {}
        for quantity in scatter_cfg.get('quantities', ['ne', 'temp']):
            try:
                gt_log, pred_log = _collect_shell_scatter_samples(
                    ref_state,
                    quantity=quantity,
                    radii=radii,
                    max_points_per_shell=int(scatter_cfg.get('max_points_per_shell', 5000)),
                    seed=int(scatter_cfg.get('seed', 0)),
                )
            except Exception:
                gt_log, pred_log = np.asarray([]), np.asarray([])
            ref_artifacts['scatter'][quantity] = (gt_log, pred_log)

    _release_state(ref_state)
    ref_state = None

    manifest = {
        'benchmark_dir': str(benchmark_dir),
        'reference_run_name': reference_row.get('run_name'),
        'reference_experiment': reference_row.get('experiment_name'),
        'reference_condition': _condition_display_name(reference_row),
        'comparisons': [],
    }

    # main loop to compare to reference run
    for cmp_row in comparison_rows:
        cmp_run_dir = _resolve_run_dir(cmp_row, benchmark_dir)
        if cmp_run_dir is None:
            continue

        cmp_state = _build_state_for_analysis(cmp_run_dir, device_override=device_override)
        cmp_dir = out_dir / _row_folder_name(cmp_row)
        cmp_dir.mkdir(parents=True, exist_ok=True)

        cmp_manifest_row = {
            'run_name': cmp_row.get('run_name'),
            'experiment_name': cmp_row.get('experiment_name'),
            'condition_display': _condition_display_name(cmp_row),
            'run_dir': str(cmp_run_dir),
            'metric_value': cmp_row.get(metric_key),
        }

        # train panels
        if train_cfg.get('enabled', True) and train_view_indices:
            cmp_train_obs = _slice_views(_get_comparison_observation_cube(cmp_state, 'train'), train_view_indices)
            cmp_train_pred = _render_selected_views(
                cmp_state, 'train', train_view_indices, chunk_size=int(train_cfg.get('chunk_size', 8192))
            )

            train_panels = {
                'clean_obs': ref_artifacts['train_obs'],
                'compare_obs': cmp_train_obs,
                'compare_pred': cmp_train_pred,
                'pred_minus_clean': cmp_train_pred - ref_artifacts['train_obs'],
            }

            plot_multiview_channel_panels(
                panel_cubes=train_panels,
                channel_labels=channel_labels,
                view_indices=train_view_indices,
                out_dir=cmp_dir / 'train_panels',
                prefix='train_compare',
                diff_panel_names={'pred_minus_clean'},
                log_images=True,
                eps=float(train_cfg.get('eps', 1.0e-12)),
            )

        # test panels
        if test_cfg.get('enabled', True) and test_view_indices:
            cmp_test_obs = _slice_views(_get_comparison_observation_cube(cmp_state, 'test'), test_view_indices)
            cmp_test_pred = _render_selected_views(
                cmp_state, 'test', test_view_indices, chunk_size=int(test_cfg.get('chunk_size', 8192))
            )

            test_panels = {
                'clean_gt': ref_artifacts['test_obs'],
                'clean_pred': ref_artifacts['test_pred'] if ref_artifacts['test_pred'] is not None else ref_artifacts['test_obs'],
                'compare_obs': cmp_test_obs,
                'compare_pred': cmp_test_pred,
                'pred_minus_compare': cmp_test_pred - cmp_test_obs,
                'pred_minus_clean': cmp_test_pred - ref_artifacts['test_obs'],
            }

            plot_multiview_channel_panels(
                panel_cubes=test_panels,
                channel_labels=channel_labels,
                view_indices=test_view_indices,
                out_dir=cmp_dir / 'test_panels',
                prefix='test_compare',
                diff_panel_names={'pred_minus_compare', 'pred_minus_clean'},
                log_images=True,
                eps=float(test_cfg.get('eps', 1.0e-12)),
            )

        # shell compare
        cmp_report = _load_final_eval_report(cmp_run_dir)
        if shell_cfg.get('enabled', True):
            for quantity in shell_cfg.get('quantities', ['ne', 'temp']):
                plot_shell_metric_compare(
                    ref_report=ref_artifacts['final_eval'],
                    cmp_report=cmp_report,
                    quantity=quantity,
                    metric_key=shell_cfg.get('metric_key', 'mean_abs_dlog'),
                    out_path=cmp_dir / 'shell_compare' / f'{quantity}_{shell_cfg.get("metric_key", "mean_abs_dlog")}.png',
                    label_ref='clean',
                    label_cmp=str(cmp_row.get('experiment_name')),
                )

        # scatter compare
        if scatter_cfg.get('enabled', True):
            radii = scatter_cfg.get('radii', cmp_state.cfg.get('eval', {}).get('shell_radii', []))
            for quantity in scatter_cfg.get('quantities', ['ne', 'temp']):
                try:
                    gt_cmp, pred_cmp = _collect_shell_scatter_samples(
                        cmp_state,
                        quantity=quantity,
                        radii=radii,
                        max_points_per_shell=int(scatter_cfg.get('max_points_per_shell', 5000)),
                        seed=int(scatter_cfg.get('seed', 0)),
                    )
                except Exception:
                    gt_cmp, pred_cmp = np.asarray([]), np.asarray([])

                gt_ref, pred_ref = ref_artifacts.get('scatter', {}).get(quantity, (np.asarray([]), np.asarray([])))
                plot_scalar_scatter_compare(
                    gt_ref=gt_ref,
                    pred_ref=pred_ref,
                    gt_cmp=gt_cmp,
                    pred_cmp=pred_cmp,
                    quantity=quantity,
                    out_path=cmp_dir / 'scatter' / f'{quantity}.png',
                    label_cmp=str(cmp_row.get('experiment_name')),
                )

        _release_state(cmp_state)
        cmp_state = None

        manifest['comparisons'].append(cmp_manifest_row)

    save_json(manifest, out_dir / 'manifest.json')
    return manifest