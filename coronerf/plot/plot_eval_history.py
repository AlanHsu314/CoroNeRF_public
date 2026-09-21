###########################################################
### PLOTTING EVALUATION METRICS POST TRAINING
###########################################################

'''
Reads jsons from runs and makes plots of them

already hooked into end of training pipeline in train/loops.py

but, can run manually run using command:

python -m coronerf.plot.plot_eval_history --run_dir runs/20260313-175926_test_artifacts
'''

from __future__ import annotations
from typing import Union

import json
from pathlib import Path
import yaml

import numpy as np
import matplotlib.pyplot as plt

SCALAR_META = {
    'ne': {
        'name': 'density',
        'ylabel_dlog': r'Mean $|\Delta \log_{10} n_e|$',
        'title_shell': 'Shell density error vs Training Step',
        'title_band': 'Trusted shell-band density error vs step',
        'shell_file': 'density_shell_mean_abs_dlog_vs_step.png',
        'band_file': 'density_shell_band_mean_abs_dlog_vs_step.png',
    },
    'temp': {
        'name': 'temperature',
        'ylabel_dlog': r'Mean $|\Delta \log_{10} T|$',
        'title_shell': 'Shell temperature error vs Training Step',
        'title_band': 'Trusted shell-band temperature error vs step',
        'shell_file': 'temperature_shell_mean_abs_dlog_vs_step.png',
        'band_file': 'temperature_shell_band_mean_abs_dlog_vs_step.png',
    },
}

def load_eval_jsons(eval_dir: Path) -> list[dict]:
    '''
    loads in list of json evals, sorted by step
    '''
    files = sorted(eval_dir.glob('eval_step*.json'))
    rows = []
    for path in files:
        with open(path, 'r') as f:
            data = json.load(f)
        step = int(data['step']) # integer step
        rows.append((step, data))
    rows.sort(key = lambda x: x[0])
    return rows

def get_image_logmse_mean(report: dict) -> Union[float, None]:
    '''
    for specific step, grabs logmse mean data (over viewpoints) per channel
    '''
    image_eval = report.get('image_eval', {})
    vals = image_eval.get('logmse_mean_per_channel', None)
    if vals is None:
        return None
    return np.asarray(vals, dtype= float)

def get_image_mse_mean(report: dict) -> Union[float, None]:
    '''
    for specific step, grabs mse mean data (over viewpoints) per channel
    '''
    image_eval = report.get('image_eval', {})
    vals = image_eval.get('mse_mean_per_channel', None)
    if vals is None:
        return None
    return np.asarray(vals, dtype= float)

def get_image_psnr_mean(report: dict):
    image_eval = report.get('image_eval', {})
    vals = image_eval.get('psnr_mean_per_channel', None)
    if vals is None:
        return None
    return np.asarray(vals, dtype=float)

def get_scalar_eval(report: dict, quantity: str) -> dict:
    '''
    scalar_field_eval if present
    '''
    scalar_eval = report.get('scalar_field_eval', {})
    return scalar_eval.get(quantity, {})

def get_scalar_shell_mean_abs_dlog(report: dict, quantity: str) -> Union[tuple[np.ndarray, list[str]], None]:
    '''
    for step, get mean abs dlog (per shell)
    '''
    shell_eval = get_scalar_eval(report, quantity)
    if not shell_eval.get('available', False):
        return None
    
    shells = shell_eval.get('shells', [])
    vals = [row['mean_abs_dlog'] for row in shells if row['mean_abs_dlog'] is not None]
    labels = [f"r = {row['r']:.02f}" for row in shells if row['mean_abs_dlog'] is not None]
    if not vals:
        return None
    return np.asarray(vals, dtype = float), labels

def get_scalar_shell_band_metric(report: dict, 
                                 quantity: str,
                                 band_name: str, 
                                 key: str = 'shell_mean_abs_dlog_mean') -> Union[float, None]:
    shell_eval = get_scalar_eval(report, quantity)
    if not shell_eval.get('available', False):
        return None

    aggs = shell_eval.get('aggregates', {})
    band = aggs.get(band_name, {})
    val = band.get(key, None)
    return None if val is None else float(val)

def get_available_scalar_quantities(rows: list[tuple[int, dict]]) -> list[str]:
    quants = []
    for q in ['ne', 'temp']:
        for _, report in rows:
            shell_eval = get_scalar_eval(report, q)
            if shell_eval.get('available', False):
                quants.append(q)
                break
    return quants

def get_available_band_names(rows: list[tuple[int, dict]], quantity: str) -> list[str]:
    '''
    get valid band names
    '''
    for _, report in rows:
        shell_eval = get_scalar_eval(report, quantity)
        if not shell_eval.get('available', False):
            continue
        aggs = shell_eval.get('aggregates', {})
        if aggs:
            return list(aggs.keys())
    return []

def _resolve_image_channel_labels(raw_cfg: dict, num_channels: int) -> list[str]:
    '''
    normalize channel names for plotting

    they should match from preprocessing checks, but double check here
    '''
    labels = list(raw_cfg.get('ion_kwarags', {}).get('line_names', []) or [])
    channel_indices = raw_cfg.get('load_data_kwargs', {}).get('channel_indices', None)

    # grab labels
    if channel_indices is not None and labels:
        labels = [labels[i] for i in channel_indices]

    # add extras if not enough until num_channels
    if len(labels) < num_channels:
        labels = labels + [f'ch {i}' for i in range(len(labels), num_channels)]

    # just in case len(labels) > num_channels
    return labels[:num_channels]

def plot_metric_vs_step(
    steps: list[int],
    values: list[Union[float, None]],
    out_path: Path,
    ylabel: str = '',
    title: str = '',
    labels: list[str] = None,
    plot_indiv_channels: bool = True,
    plot_mean_channels: bool = False):
    '''
    generic history plotter

    values can be scalar or vector per step
    '''
    
    # plot non-None values
    xs, ys = [], []
    for s, v in zip(steps, values):
        if v is None:
            continue
        xs.append(s)
        ys.append(v)

    if not xs:
        print(f'No valid data for {out_path.name}')
        return
    
    xs = np.array(xs, dtype = float)
    ys = np.array(ys, dtype = float)

    if ys.ndim == 1:
        ys = ys[:, None]

    num_channels = ys.shape[-1]
    # create labels for each channel, similar to resolving image channels
    if labels is None:
        channel_labels = [f'ch {i}' for i in range(num_channels)]
    else:
        channel_labels = list(labels)
        if len(channel_labels) < num_channels:
            channel_labels = channel_labels + [f'ch {i}' for i in range(len(channel_labels), num_channels)]
        else:
            channel_labels = channel_labels[:num_channels]

    plt.figure(figsize = (6, 4))

    # plot mean if asked
    if plot_mean_channels:
        y_mean = np.nanmean(ys, axis = -1)
        plt.plot(xs, y_mean, linestyle = '--', alpha = 0.7, label = 'mean')
    
    # loop through channels
    if plot_indiv_channels:
        for ci in range(num_channels):
            plt.plot(xs, ys[:,ci], marker = 'o', label = channel_labels[ci])
    
    plt.xlabel('Training step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha = 0.3)

    if num_channels > 1 or plot_mean_channels:
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi = 150)
    plt.close()

def plot_scalar_band_history_metric(
    rows: list[tuple[int, dict]],
    steps: np.ndarray,
    quantity: str,
    plot_dir: Path,
    key: str,
    ylabel: str,
    title: str,
    out_file: str,
):
    band_names = get_available_band_names(rows, quantity)
    if not band_names:
        return

    band_vals = []
    for _, report in rows:
        vals = [
            get_scalar_shell_band_metric(report, quantity, band_name, key=key)
            for band_name in band_names
        ]
        band_vals.append(np.asarray(vals, dtype=float))

    plot_metric_vs_step(
        steps=steps,
        values=band_vals,
        ylabel=ylabel,
        title=title,
        out_path=plot_dir / out_file,
        labels=band_names,
        plot_indiv_channels=True,
        plot_mean_channels=False,
    )

def plot_scalar_history(
    rows: list[tuple[int, dict]],
    steps: np.ndarray,
    quantity: str,
    plot_dir: Path,
):
    '''
    generalized scalar field plotter
    '''
    meta = SCALAR_META[quantity]

    shell_reports = [get_scalar_shell_mean_abs_dlog(report, quantity) for _, report in rows]

    first_valid = next((item for item in shell_reports if item is not None), None)
    if first_valid is None:
        return

    shell_labels = first_valid[1]
    shell_vals = [item[0] if item is not None else None for item in shell_reports]

    plot_metric_vs_step(
        steps=steps,
        values=shell_vals,
        ylabel=meta['ylabel_dlog'],
        title=meta['title_shell'],
        out_path=plot_dir / meta['shell_file'],
        labels=shell_labels,
        plot_indiv_channels=False,
        plot_mean_channels=True,
    )

    band_names = get_available_band_names(rows, quantity)
    if not band_names:
        return

    band_vals = []
    for _, report in rows:
        vals = [
            get_scalar_shell_band_metric(report, quantity, band_name)
            for band_name in band_names
        ]
        band_vals.append(np.asarray(vals, dtype=float))

    plot_metric_vs_step(
        steps=steps,
        values=band_vals,
        ylabel=meta['ylabel_dlog'],
        title=meta['title_band'],
        out_path=plot_dir / meta['band_file'],
        labels=band_names,
        plot_indiv_channels=True,
        plot_mean_channels=False,
    )

    if quantity == 'ne':
        plot_scalar_band_history_metric(
            rows=rows,
            steps=steps,
            quantity='ne',
            plot_dir=plot_dir,
            key='shell_mean_abs_frac_ne_mean',
            ylabel=r'Mean $|n_e^{pred} - n_e^{gt}| / |n_e^{gt}|$',
            title='Trusted shell-band density relative MAE vs step',
            out_file='density_shell_band_mean_abs_frac_vs_step.png',
        )

        plot_scalar_band_history_metric(
            rows=rows,
            steps=steps,
            quantity='ne',
            plot_dir=plot_dir,
            key='shell_median_abs_frac_ne_mean',
            ylabel=r'Median $|n_e^{pred} - n_e^{gt}| / |n_e^{gt}|$',
            title='Trusted shell-band density median relative error vs step',
            out_file='density_shell_band_median_abs_frac_vs_step.png',
        )

    elif quantity == 'temp':
        plot_scalar_band_history_metric(
            rows=rows,
            steps=steps,
            quantity='temp',
            plot_dir=plot_dir,
            key='shell_mean_abs_frac_temp_mean',
            ylabel=r'Mean $|T^{pred} - T^{gt}| / |T^{gt}|$',
            title='Trusted shell-band temperature relative MAE vs step',
            out_file='temperature_shell_band_mean_abs_frac_vs_step.png',
        )

        plot_scalar_band_history_metric(
            rows=rows,
            steps=steps,
            quantity='temp',
            plot_dir=plot_dir,
            key='shell_median_abs_frac_temp_mean',
            ylabel=r'Median $|T^{pred} - T^{gt}| / |T^{gt}|$',
            title='Trusted shell-band temperature median relative error vs step',
            out_file='temperature_shell_band_median_abs_frac_vs_step.png',
        )

def plot_eval_history(run_dir):
    '''
    general eval history plotter
    [1] plots image-space directly here
    [2] field-space eval is done in helpers
    
    '''
    # make dirs
    run_dir = Path(run_dir)
    eval_dir = run_dir / 'artifacts' / 'eval' / 'json'
    plot_dir = run_dir / 'artifacts' / 'eval' / 'plots'
    plot_dir.mkdir(exist_ok = True, parents = True)

    # load jsons
    rows = load_eval_jsons(eval_dir)
    if not rows:
        print(f'No eval jsons found in {eval_dir}')
        return

    steps = np.array([step for step, _ in rows], dtype = int)

    ############################
    ## image-space
    ############################

    # image vals (per step, per channel)
    logmse_vals = np.array([get_image_logmse_mean(report) for _, report in rows], dtype = float)
    mse_vals = np.array([get_image_mse_mean(report) for _, report in rows], dtype = float)
    psnr_vals = np.array([get_image_psnr_mean(report) for _, report in rows], dtype=float)

    # grab intensity channel names
    with open(run_dir / 'config.yaml', 'r') as f:
        raw_cfg = yaml.safe_load(f)
    num_image_channels = int(logmse_vals.shape[-1] if logmse_vals.ndim > 1 else 1)
    image_channel_labels = _resolve_image_channel_labels(raw_cfg, num_channels = num_image_channels)

    # plots
    plot_metric_vs_step(
        steps = steps,
        values = logmse_vals,
        ylabel = 'Mean image log-MSE',
        title = 'Image log-MSE vs Training Step',
        out_path = plot_dir / 'image_logmse_vs_step.png',
        labels = image_channel_labels,
        plot_indiv_channels = True,
        plot_mean_channels = False,
        )
    
    plot_metric_vs_step(
        steps = steps,
        values = mse_vals,
        ylabel = 'Mean image MSE',
        title = 'Image MSE vs Training Step',
        out_path = plot_dir / 'image_mse_vs_step.png',
        labels = image_channel_labels,
        plot_indiv_channels = True,
        plot_mean_channels = False,
        )
    
    plot_metric_vs_step(
        steps = steps,
        values = psnr_vals,
        ylabel = 'Mean image PSNR',
        title = 'Image PSNR vs Training Step',
        out_path = plot_dir / 'image_psnr_vs_step.png',
        labels = image_channel_labels,
        plot_indiv_channels = True,
        plot_mean_channels = False,
    )
    

    

    ############################
    ## field-space
    ############################

    for quantity in get_available_scalar_quantities(rows):
        plot_scalar_history(rows, steps, quantity, plot_dir)

    # # shell vals (per step, per radii)
    # shell_reports = [get_shell_mean_abs_dlog(report) for _, report in rows]
    # shell_vals = np.array([v for v, _ in shell_reports], dtype = float)
    # shell_radii_labels = shell_reports[0][1] if len(shell_reports) > 0 else None # assume all the same for all rows

    # # more shell data, this time per metric band
    # inner_vals = np.array([get_shell_band_metric(report, 'inner') for _, report in rows], dtype=float)
    # mid_vals = np.array([get_shell_band_metric(report, 'mid') for _, report in rows], dtype=float)
    # full_vals = np.array([get_shell_band_metric(report, 'full') for _, report in rows], dtype=float)

    # plot_metric_vs_step(
    #     steps = steps,
    #     values = shell_vals,
    #     ylabel = r'Mean $|\Delta \log_{10}n_e|$',
    #     title = 'Shell density error vs Training Step',
    #     out_path = plot_dir / 'shell_mean_abs_dlog_vs_step.png',
    #     labels = shell_radii_labels,
    #     plot_indiv_channels = False,
    #     plot_mean_channels = True,
    #     )
    

    # # quick function to plot shell bands
    # plt.figure(figsize=(6, 4))
    # plt.plot(steps, inner_vals, marker='o', label='inner')
    # plt.plot(steps, mid_vals, marker='s', label='mid')
    # plt.plot(steps, full_vals, marker='^', label='full')
    # plt.xlabel('Training step')
    # plt.ylabel(r'Mean $|\Delta \log_{10} n_e|$')
    # plt.title('Trusted shell-band density error vs step')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(plot_dir / 'shell_band_mean_abs_dlog_vs_step.png', dpi=150)
    # plt.close()

def main(run_dir):
    plot_eval_history(run_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.run_dir)

