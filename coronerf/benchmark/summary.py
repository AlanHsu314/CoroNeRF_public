#################################################################
# read training eval JSONS and summarize them
#################################################################

from __future__ import annotations

from pathlib import Path
from typing import Union, Any
import json

from ..pipeline.state import PipelineState


def _safe_read_json(path: Path) -> Union[dict, None]:
    if not path.exists():
        return None
    with path.open('r') as f:
        return json.load(f)
    
def find_latest_inline_eval_json(run_dir: Union[Path, str]) -> Union[Path, None]:
    '''
    gets latest stats
    '''
    # load eval dir
    run_dir = Path(run_dir)
    eval_dir = run_dir / 'artifacts' / 'eval' / 'json'
    if not eval_dir.exists():
        return None
    
    # find latest
    step_files = sorted(eval_dir.glob('eval_step*.json'))
    if not step_files:
        return None
    return step_files[-1]

def find_final_eval_json(run_dir: Union[Path, str]) -> Union[Path, None]:
    '''
    final eval json post-training
    '''
    run_dir = Path(run_dir)
    path = run_dir / 'artifacts' / 'eval' / 'json' / 'eval_final_report.json'
    return path if path.exists() else None

def find_latest_checkpoint(run_dir: Union[Path, str]) -> Union[Path, None]:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / 'checkpoints'
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob('ckpt_step*.pt'))
    if not ckpts:
        return None
    return ckpts[-1]

def _normalize_channel_ids(channel_ids, n_channels: int) -> list[int]:
    '''
    normalize channel ids for eval and plots
    
    (1) if channel_ids is normal: use it
    (2) otherwise just rename the channels
    '''
    if channel_ids is None:
        return list(range(n_channels))
    out = [int(i) for i in channel_ids]
    if len(out) != n_channels:
        return list(range(n_channels))
    return out

def _add_image_metric_summary(out: dict, metric_name: str, vals, channel_ids = None):
    '''
    helper to add metrics to image-space summaries
    vals: array of values for each channel

    adds:
    metric for each channel
    average across all channels
    '''
    # check metric values
    if vals is None:
        return
    vals = [float(v) for v in vals]
    if not vals:
        return
    
    # normalize channel name, add to dict
    channel_ids = _normalize_channel_ids(channel_ids, len(vals))
    for local_idx, global_idx in enumerate(channel_ids):
        out[f'final/{metric_name}_mean_ch_{global_idx}'] = vals[local_idx]

    # add mean over all channels
    out[f'final/{metric_name}_mean_over_channels'] = float(sum(vals) / len(vals))

def _extract_eval_metrics(report: Union[dict, None], channel_ids = None) -> dict:
    '''
    load in all stats from a specific report
    '''
    if report is None:
        return {}
    
    out = {}

    image_eval = report.get('image_eval', {})
    scalar_field_eval = report.get('scalar_field_eval', {})

    #####################################
    # image metrics
    #####################################

    # means are over test viewpoints
    _add_image_metric_summary(out, 'psnr', image_eval.get('psnr_mean_per_channel', None), channel_ids = channel_ids)
    _add_image_metric_summary(out, 'mse', image_eval.get('mse_mean_per_channel', None), channel_ids = channel_ids)
    _add_image_metric_summary(out, 'mae', image_eval.get('mae_mean_per_channel', None), channel_ids = channel_ids)
    _add_image_metric_summary(out, 'logmse', image_eval.get('logmse_mean_per_channel', None), channel_ids = channel_ids)
    _add_image_metric_summary(out, 'logmae', image_eval.get('logmae_mean_per_channel', None), channel_ids = channel_ids)

    ##################################
    # shell metrics
    ##################################

    for quantity, field_report in scalar_field_eval.items():
        if not field_report.get('available', False):
            continue

        shells = field_report.get('shells', [])
        aggs = field_report.get('aggregates', {})

        frac_key = 'frac_ne' if quantity == 'ne' else 'frac_temp'

        # per-shell summaries
        mean_dlog = [row['mean_dlog'] for row in shells if row.get('mean_dlog') is not None]               # signed error
        median_dlog = [row['median_dlog'] for row in shells if row.get('median_dlog') is not None]
        mean_abs_dlog = [row['mean_abs_dlog'] for row in shells if row.get('mean_abs_dlog') is not None]   # abs error
        rmse = [row['rmse_dlog'] for row in shells if row.get('rmse_dlog') is not None]
        frac_mean = [row[f'mean_abs_{frac_key}'] for row in shells if row.get(f'mean_abs_{frac_key}') is not None]
        frac_med = [row[f'median_abs_{frac_key}'] for row in shells if row.get(f'median_abs_{frac_key}') is not None]

        # avg shells
        if mean_dlog:
            out[f'final/{quantity}_shell_mean_dlog_mean'] = float(sum(mean_dlog) / len(mean_dlog))
        if median_dlog:
            out[f'final/{quantity}_shell_median_dlog_mean'] = float(sum(median_dlog) / len(median_dlog))
        if mean_abs_dlog:
            out[f'final/{quantity}_shell_mean_abs_dlog_mean'] = float(sum(mean_abs_dlog) / len(mean_abs_dlog))
        if rmse:
            out[f'final/{quantity}_shell_rmse_dlog_mean'] = float(sum(rmse) / len(rmse))
        if frac_mean:
            out[f'final/{quantity}_shell_mean_abs_{frac_key}_mean'] = float(sum(frac_mean) / len(frac_mean))
        if frac_med:
            out[f'final/{quantity}_shell_median_abs_{frac_key}_mean'] = float(sum(frac_med) / len(frac_med))

        for band_name, band_stats in aggs.items():
            for key, val in band_stats.items():
                if val is None:
                    continue
                if key in ['r_min', 'r_max', 'num_shells']:
                    out[f'final/{quantity}_shell_{band_name}/{key}'] = val
                else:
                    out[f'final/{quantity}_shell_{band_name}/{key}'] = float(val)

    return out

def _extract_observation_noise_metrics(noise_meta: Union[dict, None]) -> dict:
    '''
    loads noise meta from adding observation noise, adds to summary
    '''
    if noise_meta is None:
        return {}

    out = {
        'observation_noise_enabled': bool(noise_meta.get('enabled', False)),
        'observation_noise_model': noise_meta.get('model'),
        'observation_noise_apply_to': noise_meta.get('apply_to'),
        'observation_noise_num_train_pixels': noise_meta.get('num_train_pixels'),
        'observation_noise_num_active_channels': noise_meta.get('num_active_channels'),
    }

    channels = noise_meta.get('channels', [])
    for ch in channels:
        global_idx = int(ch.get('global_channel', ch.get('local_channel', -1)))
        prefix = f'noise/ch_{global_idx}'

        # effective shot noise extras
        out[f'{prefix}/alpha'] = float(ch.get('alpha', 0.0))
        out[f'{prefix}/noise_rmse'] = float(ch.get('noise_rmse', 0.0))

        rel_rmse = ch.get('noise_rel_rmse', None)
        if rel_rmse is not None:
            out[f'{prefix}/noise_rel_rmse'] = float(rel_rmse)

        mean_abs_rel = ch.get('mean_abs_rel_noise', None)
        if mean_abs_rel is not None:
            out[f'{prefix}/mean_abs_rel_noise'] = float(mean_abs_rel)

        frac_zero = ch.get('frac_zero_observed', None)
        if frac_zero is not None:
            out[f'{prefix}/frac_zero_observed'] = float(frac_zero)

        counts_p50 = ch.get('expected_counts_percentiles', {}).get('50', None)
        if counts_p50 is not None:
            out[f'{prefix}/expected_counts_p50'] = float(counts_p50)

        snr_p50 = ch.get('theory_snr_percentiles', {}).get('50', None)
        if snr_p50 is not None:
            out[f'{prefix}/theory_snr_p50'] = float(snr_p50)

        # heterogauss noise extras
        shot_coeff = ch.get('shot_coeff', None)
        if shot_coeff is not None:
            out[f'{prefix}/shot_coeff'] = float(shot_coeff)

        sigma_floor = ch.get('sigma_floor', None)
        if sigma_floor is not None:
            out[f'{prefix}/sigma_floor'] = float(sigma_floor)

        sigma_p50 = ch.get('sigma_percentiles', {}).get('50', None)
        if sigma_p50 is not None:
            out[f'{prefix}/sigma_p50'] = float(sigma_p50)

        frac_neg = ch.get('frac_negative_observed', None)
        if frac_neg is not None:
            out[f'{prefix}/frac_negative_observed'] = float(frac_neg)

    return out

def _extract_observation_mismatch_metrics(mismatch_meta: Union[dict, None]) -> dict:
    """
    loads deterministic observation/model mismatch metadata, adds to run summary
    """
    if mismatch_meta is None:
        return {}

    out = {
        'observation_mismatch_enabled': bool(mismatch_meta.get('enabled', False)),
        'observation_mismatch_model': mismatch_meta.get('model'),
        'observation_mismatch_apply_to': mismatch_meta.get('apply_to'),
        'observation_mismatch_num_active_channels': mismatch_meta.get('num_active_channels'),
        'observation_mismatch_splits_present': ','.join(mismatch_meta.get('splits_present', [])),
    }

    for ch in mismatch_meta.get('channels', []):
        global_idx = int(ch.get('global_channel', ch.get('local_channel', -1)))
        prefix = f'mismatch/ch_{global_idx}'
        out[f'{prefix}/scale_factor'] = float(ch.get('scale_factor', 1.0))
        out[f'{prefix}/dlog_intensity_shift'] = float(ch.get('dlog_intensity_shift', 0.0))

    for split_name, split_meta in (mismatch_meta.get('splits', {}) or {}).items():
        out[f'observation_mismatch_{split_name}_num_pixels'] = split_meta.get('num_pixels')
        out[f'observation_mismatch_{split_name}_num_views'] = split_meta.get('num_views')

        for ch in split_meta.get('channels', []):
            global_idx = int(ch.get('global_channel', ch.get('local_channel', -1)))
            prefix = f'mismatch/{split_name}/ch_{global_idx}'

            out[f'{prefix}/scale_factor'] = float(ch.get('scale_factor', 1.0))
            out[f'{prefix}/dlog_intensity_shift'] = float(ch.get('dlog_intensity_shift', 0.0))

            for key in ['delta_rmse', 'delta_rel_rmse', 'mean_abs_rel_delta', 'mean_rel_delta']:
                val = ch.get(key, None)
                if val is not None:
                    out[f'{prefix}/{key}'] = float(val)

            p50 = ch.get('nominal_percentiles', {}).get('50', None)
            if p50 is not None:
                out[f'{prefix}/nominal_p50'] = float(p50)

            p50 = ch.get('observed_percentiles', {}).get('50', None)
            if p50 is not None:
                out[f'{prefix}/observed_p50'] = float(p50)

    return out

def _extract_loss_history_metrics(run_dir: Union[Path, str]) -> dict:
    """
    Extract final train/heldout loss values from artifacts/train/loss_history.json.
    """
    path = Path(run_dir) / 'artifacts' / 'train' / 'loss_history.json'
    data = _safe_read_json(path)
    if not data:
        return {}

    out = {}

    if data.get('loss'):
        out['final/train_loss'] = float(data['loss'][-1])

    if data.get('loss_main'):
        out['final/train_loss_main'] = float(data['loss_main'][-1])

    if data.get('val_loss'):
        out['final/val_loss'] = float(data['val_loss'][-1])

    if data.get('val_step'):
        out['final/val_loss_step'] = int(data['val_step'][-1])

    return out

def _extract_global_val_metrics(run_dir: Union[Path, str]) -> dict:
    """Cross-condition global validation losses from artifacts/train/global_val.json (if present).
    Emits final/val_<name> keys so a summary rebuild preserves any global eval already computed."""
    path = Path(run_dir) / 'artifacts' / 'train' / 'global_val.json'
    data = _safe_read_json(path)
    if not data:
        return {}
    out = {}
    for k, v in data.items():
        if isinstance(k, str) and k.startswith('final/val_'):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out

def build_summary(
        state: PipelineState,
        run_dir: Union[Path, str],
        status: str,
        error_message: Union[str, None] = None,
) -> dict:
    '''
    get a summary report from the evals produced by a run
    '''
    cfg = dict(state.cfg)
    run_dir = Path(run_dir)

    # get relevant dirs
    final_eval_path = find_final_eval_json(run_dir)
    latest_eval_path = find_latest_inline_eval_json(run_dir)
    ckpt_path = find_latest_checkpoint(run_dir)

    # read in eval
    final_eval = _safe_read_json(final_eval_path) if final_eval_path else None
    latest_eval = _safe_read_json(latest_eval_path) if latest_eval_path else None

    # compute metrics, prefer final eval, fallback to inline eval
    metrics_source = final_eval if final_eval is not None else latest_eval

    # add some benchmark meta
    benchmark_meta = cfg.get('benchmark_meta', {})
    sweep_values = benchmark_meta.get('sweep_values', {})

    # active channels used in training
    active_channel_indices = getattr(state, 'selected_channel_indices', None)
    active_wavelengths = getattr(state, 'selected_wavelengths', None)
    active_line_names = getattr(state, 'selected_line_names', None)

    # observation noise meta
    observation_noise_meta = getattr(state, 'observation_noise_meta', None)

    # observation mismatch meta
    observation_mismatch_meta = getattr(state, 'observation_mismatch_meta', None)

    # create summary
    summary = {
        'status': status,
        'run_dir': str(run_dir),
        'run_name': cfg.get('benchmark_meta', {}).get('run_name'),
        'benchmark_name': cfg.get('benchmark_meta', {}).get('benchmark_name'),
        'experiment_name': cfg.get('benchmark_meta', {}).get('experiment_name'),
        'reconstruction_target': cfg.get('shared', {}).get('reconstruction_target'),
        'forward_backend': cfg.get('forward_backend'),
        'seed': cfg.get('seed'),
        'model_name': cfg.get('model', {}).get('name'),
        'renderer_model': cfg.get('renderer', {}).get('renderer_model'),
        #'dataset_dir': cfg.get('dataset_dir'),                              # could be stale
        'active_dataset_dir': str(getattr(state, 'dataset_dir', '')) if getattr(state, 'dataset_dir', None) is not None else None,
        'dataset_compare_ready': bool(getattr(state, 'dataset_dir', None) is not None),
        'vpgen_n_lon': cfg.get('vpgen', {}).get('n_lon'),
        'train_max_steps': cfg.get('train', {}).get('max_steps'),
        'latest_checkpoint': str(ckpt_path) if ckpt_path else None,
        'latest_inline_eval_json': str(latest_eval_path) if latest_eval_path else None,
        'final_eval_json': str(final_eval_path) if final_eval_path else None,
        'error_message': error_message,
        'available_scalar_fields': ','.join(
            sorted(
                q for q, v in (metrics_source.get('scalar_field_eval', {}) if metrics_source else {}).items()
                if v.get('available', False)
            )
        ) if metrics_source is not None else None,
        'selected_channel_indices': (
            ','.join(str(i) for i in active_channel_indices)
            if active_channel_indices is not None else None
        ),
        'selected_wavelengths': (
            ','.join(str(w) for w in active_wavelengths)
            if active_wavelengths else None
        ),
        'selected_line_names': (
            ' | '.join(str(name) for name in active_line_names)
            if active_line_names else None
        ),
        # observation noise
        'observation_noise_enabled_cfg': bool(cfg.get('observation_noise', {}).get('enabled', False)),
        'observation_noise_model_cfg': cfg.get('observation_noise', {}).get('model', None),
        'observation_noise_apply_to_cfg': cfg.get('observation_noise', {}).get('apply_to', None),
        'observation_noise_meta_present': bool(observation_noise_meta is not None),
        # observation mismatch
        'observation_mismatch_enabled_cfg': bool(cfg.get('observation_mismatch', {}).get('enabled', False)),
        'observation_mismatch_model_cfg': cfg.get('observation_mismatch', {}).get('model', None),
        'observation_mismatch_apply_to_cfg': cfg.get('observation_mismatch', {}).get('apply_to', None),
        'observation_mismatch_meta_present': bool(observation_mismatch_meta is not None),
    }

    # add sweep values to summary
    for k, v in sweep_values.items():
        summary[f'sweep.{k}'] = v

    # update summary with eval metrics and obs noise/model mismatch metrics
    summary.update(_extract_eval_metrics(metrics_source, channel_ids = active_channel_indices))
    summary.update(_extract_loss_history_metrics(run_dir))
    summary.update(_extract_global_val_metrics(run_dir))
    summary.update(_extract_observation_mismatch_metrics(observation_mismatch_meta))
    summary.update(_extract_observation_noise_metrics(observation_noise_meta))


    # add vpgen split metadata, only interpret for vpgen runs
    load_data_kwargs = cfg.get('load_data_kwargs', {})

    summary.update({
        'mode': cfg.get('mode'),
        'dataset_dir': cfg.get('dataset_dir'),
        'train_type': load_data_kwargs.get('train_type'),
        'train_num_views': load_data_kwargs.get('train_num_views'),
        'train_subset_strategy': load_data_kwargs.get('train_subset_strategy'),
        'holdout_type': load_data_kwargs.get('holdout_type'),
        'holdout_num_views': load_data_kwargs.get('holdout_num_views'),
        'vpgen_spatial_H': cfg.get('vpgen', {}).get('spatial_H'),
        'vpgen_spatial_W': cfg.get('vpgen', {}).get('spatial_W'),
    })

    # add dataset metadat (which may change during training)
    summary.update({
        'dataset_num_viewpoints': getattr(state, 'dataset_num_viewpoints', None),
        'train_num_viewpoints_loaded': getattr(state, 'train_num_viewpoints_loaded', None),
        'test_num_viewpoints_loaded': getattr(state, 'test_num_viewpoints_loaded', None),
        'dataset_image_H': getattr(state, 'dataset_image_H', None),
        'dataset_image_W': getattr(state, 'dataset_image_W', None),
        'dataset_num_channels': getattr(state, 'dataset_num_channels', None),
    })

    return summary






