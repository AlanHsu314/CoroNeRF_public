##########################################################
### aggregate history plots from best runs in benchmark
##########################################################

# most of this chatgpt

from __future__ import annotations
from pathlib import Path
from typing import Union
import json

import numpy as np
import matplotlib.pyplot as plt
import torch

from .aggregate import collect_run_summaries
from .config import save_json
from .summary import find_latest_checkpoint
from ..plot.plot_eval_history import (
    load_eval_jsons,
    get_image_mse_mean,
    get_image_logmse_mean,
    get_image_psnr_mean,
    get_scalar_shell_band_metric,
)

def _load_loss_history(run_dir: Path):
    """
    Prefer a saved JSON history if present.
    Fallback to latest checkpoint so this works on old benchmarks too.
    """

    # prefer loss_history json
    json_path = run_dir / 'artifacts' / 'train' / 'loss_history.json'
    if json_path.exists():
        with json_path.open('r') as f:
            data = json.load(f)
        steps = np.asarray(data.get('step', []), dtype=int)
        vals = np.asarray(data.get('loss', []), dtype=float)
        if steps.size and vals.size:
            return steps, vals

    # otherwise load the actual checkpoint and get data from there
    ckpt_path = find_latest_checkpoint(run_dir)
    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location='cpu')
    steps = np.asarray(ckpt.get('step_hist') or [], dtype=int)
    vals = np.asarray(ckpt.get('loss_hist') or [], dtype=float)
    if not steps.size or not vals.size:
        return None
    return steps, vals

def _load_validation_loss_history(run_dir: Path):
    """
    Load heldout validation-loss history from artifacts/train/loss_history.json.
    """
    json_path = run_dir / 'artifacts' / 'train' / 'loss_history.json'
    if json_path.exists():
        with json_path.open('r') as f:
            data = json.load(f)
        steps = np.asarray(data.get('val_step', []), dtype=int)
        vals = np.asarray(data.get('val_loss', []), dtype=float)
        if steps.size and vals.size:
            return steps, vals

    ckpt_path = find_latest_checkpoint(run_dir)
    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location='cpu')
    steps = np.asarray(ckpt.get('val_step_hist') or [], dtype=int)
    vals = np.asarray(ckpt.get('val_loss_hist') or [], dtype=float)
    if not steps.size or not vals.size:
        return None

    return steps, vals

def _load_eval_scalar_series(run_dir: Path, extractor):
    """
    Build a scalar-vs-step curve from eval_step*.json.
    The extractor can return a scalar or per-channel vector.
    If vector, we take the mean over channels.
    """
    eval_dir = run_dir / 'artifacts' / 'eval' / 'json'
    rows = load_eval_jsons(eval_dir)
    if not rows:
        return None

    steps = []
    vals = []
    for step, report in rows:
        value = extractor(report)
        if value is None:
            continue

        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            scalar = float(arr)
        else:
            scalar = float(np.nanmean(arr))

        steps.append(int(step))
        vals.append(scalar)

    if not steps:
        return None

    return np.asarray(steps, dtype=int), np.asarray(vals, dtype=float)

def _select_best_runs(rows: list[dict], metric_key: str, mode: str = 'min') -> dict[str, dict]:
    """
    Select exactly one best completed run per experiment_name, based on some given metric
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get('status') != 'completed':
            continue

        exp_name = row.get('experiment_name')
        if exp_name is None:
            continue

        metric_val = row.get(metric_key)
        if metric_val is None:
            continue

        grouped.setdefault(exp_name, []).append(row)

    best = {}
    for exp_name, group in grouped.items():
        key_fn = lambda r: float(r[metric_key])
        best_row = min(group, key=key_fn) if mode == 'min' else max(group, key=key_fn)
        best[exp_name] = best_row

    return best

def _plot_overlay(
    series_by_label: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
    ylabel: str,
    yscale: Union[str, None] = None,
    hline_y: Union[float, None] = None,
):
    '''
    generic group plotter
    '''
    if not series_by_label:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    for label, (steps, values) in series_by_label.items():
        plt.plot(steps, values, label=label)

    plt.xlabel('Training step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if hline_y is not None:
        plt.axhline(float(hline_y), linestyle = '--', linewidth = 1.0, alpha = 0.6)

    if yscale is not None:
        plt.yscale(yscale)

    if len(series_by_label) > 1:
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _resolve_run_dir(row: dict, benchmark_dir: Path) -> Path | None:
    """
    Resolve a run_dir even if the benchmark directory was copied from another machine
    and the stored absolute run_dir is stale.
    """
    run_dir_raw = row.get('run_dir')
    run_name = row.get('run_name')

    #print(benchmark_dir)
    #print(run_dir_raw)
    #print(run_name)

    # 1) try stored path first
    if run_dir_raw:
        p = Path(run_dir_raw)
        if p.exists():
            return p

    # 2) fallback: benchmark_dir / run_name
    if run_name:
        p = benchmark_dir / 'runs' / run_name
        if p.exists():
            return p

    return None

def _plot_best_run_scalar_band_metric(
    best_runs: dict[str, dict],
    benchmark_dir: Path,
    out_dir: Path,
    quantity: str,
    band_name: str,
    key: str,
    out_file: str,
    title: str,
    ylabel: str,
    hline_y: Union[float, None] = None,
):
    series = {}
    for exp_name, row in best_runs.items():
        run_dir = _resolve_run_dir(row, benchmark_dir)
        if run_dir is None:
            continue

        loaded = _load_eval_scalar_series(
            run_dir,
            lambda report: get_scalar_shell_band_metric(
                report,
                quantity,
                band_name,
                key=key,
            ),
        )
        if loaded is not None:
            series[exp_name] = loaded

    _plot_overlay(
        series,
        out_dir / out_file,
        title,
        ylabel,
        hline_y=hline_y,
    )

def generate_best_run_history_plots(
    benchmark_dir: Union[Path, str],
    cfg: dict | None = None,
) -> dict:
    """
    For each experiment family, choose the best run by one final summary metric,
    then overlay those runs on shared history plots.
    """
    benchmark_dir = Path(benchmark_dir)
    cfg = cfg or {}

    metric_key = cfg.get('select_by', 'final/ne_shell_low_corona/shell_mean_abs_dlog_mean')
    mode = cfg.get('select_mode', 'min')
    band_name = cfg.get('band_name', 'low_corona')
    include = cfg.get(
        'include',
        ['train_loss', 
         'val_loss',
         'image_mse', 
         'image_logmse', 
         'image_psnr',
         'ne_mean_dlog',
         'ne_median_dlog',
         'ne_mean_abs_dlog',
         'ne_mean_abs_frac',
         'temp_mean_dlog',
         'temp_median_dlog', 
         'temp_mean_abs_dlog',
         'temp_mean_abs_frac'],
    )
    #print(include)
    out_dir = benchmark_dir / cfg.get('out_subdir', 'history_plots')

    print(out_dir)

    rows = collect_run_summaries(benchmark_dir)
    best_runs = _select_best_runs(rows, metric_key=metric_key, mode=mode)
    if not best_runs:
        return {}

    manifest = {}
    for exp_name, row in best_runs.items():
        run_dir = _resolve_run_dir(row, benchmark_dir)
        manifest[exp_name] = {
            'run_name': row.get('run_name'),
            'run_dir': str(run_dir) if run_dir is not None else row.get('run_dir'),
            'select_metric': metric_key,
            'select_value': row.get(metric_key),
        }

    save_json(manifest, out_dir / 'best_runs.json')

    if 'train_loss' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue
            loaded = _load_loss_history(run_dir)
            if loaded is not None:
                series[exp_name] = loaded
        _plot_overlay(
            series,
            out_dir / 'train_loss_best_runs.png',
            'Training loss of best run in each experiment family',
            'Train loss',
            yscale='log',
        )

    if 'val_loss' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue

            loaded = _load_validation_loss_history(run_dir)
            if loaded is not None:
                series[exp_name] = loaded

        _plot_overlay(
            series,
            out_dir / 'val_loss_best_runs.png',
            'Heldout image loss of best run in each experiment family',
            'Heldout image loss',
            yscale='log',
        )

    if 'image_mse' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue
            loaded = _load_eval_scalar_series(run_dir, get_image_mse_mean)
            if loaded is not None:
                series[exp_name] = loaded
        _plot_overlay(
            series,
            out_dir / 'image_mse_best_runs.png',
            'Image MSE of best run in each experiment family',
            'Mean image MSE',
        )

    if 'image_logmse' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue
            loaded = _load_eval_scalar_series(run_dir, get_image_logmse_mean)
            if loaded is not None:
                series[exp_name] = loaded
        _plot_overlay(
            series,
            out_dir / 'image_logmse_best_runs.png',
            'Image log-MSE of best run in each experiment family',
            'Mean image log-MSE',
        )

    if 'image_psnr' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue
            loaded = _load_eval_scalar_series(run_dir, get_image_psnr_mean)
            if loaded is not None:
                series[exp_name] = loaded
        _plot_overlay(
            series,
            out_dir / 'image_psnr_best_runs.png',
            'Image PSNR of best run in each experiment family',
            'Mean image PSNR',
        )

    if 'ne_mean_abs_dlog' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue
            loaded = _load_eval_scalar_series(
                run_dir,
                lambda report: get_scalar_shell_band_metric(
                    report,
                    'ne',
                    band_name,
                    key='shell_mean_abs_dlog_mean',
                ),
            )
            if loaded is not None:
                series[exp_name] = loaded
        _plot_overlay(
            series,
            out_dir / f'ne_{band_name}_mean_abs_dlog_best_runs.png',
            f'Density mean abs dlog in {band_name} band for best run in each experiment family',
            r'Mean $|\Delta \log_{10} n_e|$',
        )

    if 'ne_mean_abs_frac' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue
            loaded = _load_eval_scalar_series(
                run_dir,
                lambda report: get_scalar_shell_band_metric(
                    report,
                    'ne',
                    band_name,
                    key='shell_mean_abs_frac_ne_mean',
                ),
            )
            if loaded is not None:
                series[exp_name] = loaded
        _plot_overlay(
            series,
            out_dir / f'ne_{band_name}_mean_abs_frac_best_runs.png',
            f'Density relative MAE in {band_name} band for best run in each experiment family',
            r'Mean $|n_e^{pred} - n_e^{gt}| / |n_e^{gt}|$',
        )

    if 'temp_mean_abs_frac' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue
            loaded = _load_eval_scalar_series(
                run_dir,
                lambda report: get_scalar_shell_band_metric(
                    report,
                    'temp',
                    band_name,
                    key='shell_mean_abs_frac_temp_mean',
                ),
            )
            if loaded is not None:
                series[exp_name] = loaded
        _plot_overlay(
            series,
            out_dir / f'temp_{band_name}_mean_abs_frac_best_runs.png',
            f'Temperature relative MAE in {band_name} band for best run in each experiment family',
            r'Mean $|T^{pred} - T^{gt}| / |T^{gt}|$',
        )

    if 'temp_mean_abs_dlog' in include:
        series = {}
        for exp_name, row in best_runs.items():
            run_dir = _resolve_run_dir(row, benchmark_dir)
            if run_dir is None:
                continue
            loaded = _load_eval_scalar_series(
                run_dir,
                lambda report: get_scalar_shell_band_metric(
                    report,
                    'temp',
                    band_name,
                    key='shell_mean_abs_dlog_mean',
                ),
            )
            if loaded is not None:
                series[exp_name] = loaded
        _plot_overlay(
            series,
            out_dir / f'temp_{band_name}_mean_abs_dlog_best_runs.png',
            f'Temperature mean abs dlog in {band_name} band for best run in each experiment family',
            r'Mean $|\Delta \log_{10} T|$',
        )
    
    # signed metrics
    if 'ne_mean_dlog' in include:
        _plot_best_run_scalar_band_metric(
            best_runs=best_runs,
            benchmark_dir=benchmark_dir,
            out_dir=out_dir,
            quantity='ne',
            band_name=band_name,
            key='shell_mean_dlog_mean',
            out_file=f'ne_{band_name}_mean_dlog_best_runs.png',
            title=f'Density signed mean dlog in {band_name} band for best run in each experiment family',
            ylabel=r'Mean $\Delta \log_{10} n_e$',
            hline_y=0.0,
        )

    if 'ne_median_dlog' in include:
        _plot_best_run_scalar_band_metric(
            best_runs=best_runs,
            benchmark_dir=benchmark_dir,
            out_dir=out_dir,
            quantity='ne',
            band_name=band_name,
            key='shell_median_dlog_mean',
            out_file=f'ne_{band_name}_median_dlog_best_runs.png',
            title=f'Density signed median dlog in {band_name} band for best run in each experiment family',
            ylabel=r'Median $\Delta \log_{10} n_e$',
            hline_y=0.0,
        )

    if 'temp_mean_dlog' in include:
        _plot_best_run_scalar_band_metric(
            best_runs=best_runs,
            benchmark_dir=benchmark_dir,
            out_dir=out_dir,
            quantity='temp',
            band_name=band_name,
            key='shell_mean_dlog_mean',
            out_file=f'temp_{band_name}_mean_dlog_best_runs.png',
            title=f'Temperature signed mean dlog in {band_name} band for best run in each experiment family',
            ylabel=r'Mean $\Delta \log_{10} T$',
            hline_y=0.0,
        )

    if 'temp_median_dlog' in include:
        _plot_best_run_scalar_band_metric(
            best_runs=best_runs,
            benchmark_dir=benchmark_dir,
            out_dir=out_dir,
            quantity='temp',
            band_name=band_name,
            key='shell_median_dlog_mean',
            out_file=f'temp_{band_name}_median_dlog_best_runs.png',
            title=f'Temperature signed median dlog in {band_name} band for best run in each experiment family',
            ylabel=r'Median $\Delta \log_{10} T$',
            hline_y=0.0,
        )


    return manifest











