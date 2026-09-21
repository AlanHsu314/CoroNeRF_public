#######################################################
### PLOTTING EVALUATION METRICS DURING TRAINING
#######################################################

'''
Reads jsons from runs and makes plots of them

already hooked into training pipeline in train/loops.py

but, can run manually run using command:

python -m coronerf.plot.plot_eval_history --run_dir runs

python -m coronerf.plot.plot_eval_snapshot --json_path runs/20260313-175926_test_artifacts/artifacts/eval/json/eval_step020000.json
'''

from __future__ import annotations
from typing import Union

import json
from pathlib import Path
import yaml

import numpy as np
import matplotlib.pyplot as plt

SCALAR_PLOT_META = {
    'ne': {
        'title': 'Density shell error vs radius',
        'ylabel': 'Error',
        'mean_label': r"Mean $|\Delta \log_{10} n_e|$",
        'rmse_label': r"RMSE $\Delta \log_{10} n_e$",
        'filename': 'density_shell_error_vs_radius_step{step:06d}.png',
    },
    'temp': {
        'title': 'Temperature shell error vs radius',
        'ylabel': 'Error',
        'mean_label': r"Mean $|\Delta \log_{10} T|$",
        'rmse_label': r"RMSE $\Delta \log_{10} T$",
        'filename': 'temperature_shell_error_vs_radius_step{step:06d}.png',
    },
}

def get_available_scalar_quantities(report: dict) -> list[str]:
    scalar_eval = report.get('scalar_field_eval', {})
    out = []
    for quantity, field_report in scalar_eval.items():
        if field_report.get('available', False):
            out.append(quantity)
    return out

def plot_scalar_shell_error_vs_radius(report: dict, quantity: str, out_dir: Path, step: int):
    scalar_eval = report.get('scalar_field_eval', {})
    field_report = scalar_eval.get(quantity, {})

    if not field_report.get('available', False):
        return

    shells = field_report.get('shells', [])
    rs = []
    mean_abs = []
    rmse = []

    for row in shells:
        if row.get('mean_abs_dlog') is None:
            continue
        rs.append(row['r'])
        mean_abs.append(row['mean_abs_dlog'])
        rmse.append(row['rmse_dlog'])

    if not rs:
        return

    meta = SCALAR_PLOT_META[quantity]
    out_path = out_dir / meta['filename'].format(step=step)

    plt.figure(figsize=(6, 4))
    plt.plot(rs, mean_abs, marker='o', label=meta['mean_label'])
    plt.plot(rs, rmse, marker='s', label=meta['rmse_label'])
    plt.xlabel(r'Radius $R_\odot$')
    plt.ylabel(meta['ylabel'])
    plt.title(meta['title'])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_eval_snapshot(json_path: str | Path):
    json_path = Path(json_path)
    with open(json_path, 'r') as f:
        report = json.load(f)

    step = int(report['step'])
    out_dir = json_path.parent.parent / 'plots'
    out_dir.mkdir(exist_ok=True, parents=True)

    for quantity in get_available_scalar_quantities(report):
        plot_scalar_shell_error_vs_radius(report, quantity=quantity, out_dir=out_dir, step=step)

def main(json_path):
    plot_eval_snapshot(json_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()

    main(args.json_path)












