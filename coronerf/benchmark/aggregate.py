#########################################################
### aggregate results from a benchmark run
#########################################################

from __future__ import annotations
from typing import Union

from pathlib import Path
import csv
import json

def _freeze_group_value(value):
    '''
    Convert nested containers into a hashable form for grouping keys.
    '''
    if isinstance(value, list):
        return tuple(_freeze_group_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_group_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_group_value(v)) for k, v in value.items()))
    return value

def _csv_safe_value(value):
    if isinstance(value, list):
        if all(isinstance(v, (int, float)) for v in value):
            return '[' + ', '.join(f'{v:g}' for v in value) + ']'
        return json.dumps(value, sort_keys=True)
    if isinstance(value, tuple):
        return json.dumps(list(value), sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value

def _read_json(path: Path) -> dict:
    with path.open('r') as f:
        return json.load(f)
    
def collect_run_summaries(benchmark_dir: Union[Path, dir]) -> list[dict]:
    '''
    load in the summary for each valid run
    '''
    
    benchmark_dir = Path(benchmark_dir)
    runs_dir = benchmark_dir / 'runs'
    if not runs_dir.exists():
        return []
    
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / 'summary.json'
        if not summary_path.exists():
            continue
        rows.append(_read_json(summary_path))
    return rows

def _all_keys(rows: list[dict]) -> list[str]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return sorted(keys)

def write_csv(rows: list[dict], path: Union[Path, str]) -> None:
    '''
    write the rows to csv
    '''
    path = Path(path)
    path.parent.mkdir(exist_ok = True, parents = True)

    if not rows:
        with path.open('w', newline='') as f:
            f.write('')
        return
    
    keys = _all_keys(rows)
    with path.open('w', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# some helpers for grouping
def _is_metric_key(key: str) -> bool:
    """
    Metric keys that should be grouped across seeds in aggregate_grouped.csv.

    We intentionally do not aggregate every final/* key because some are
    metadata, counters, or non-numeric diagnostics.
    """
    return (
        (key.startswith('final/val_') and not key.endswith('_step'))
        or key == 'final/train_loss'
        or key == 'final/train_loss_main'
        or key.startswith('final/psnr_')
        or key.startswith('final/mse_')
        or key.startswith('final/mae_')
        or key.startswith('final/logmse_')
        or key.startswith('final/logmae_')
        or key.startswith('final/temp_shell_')
        or key.startswith('final/ne_shell_')
    )

def _is_excluded_group_key(key:str) -> bool:
    return key in {
        'stats',
        'run_dir',
        'run_name',
        'seed',
        'latest_checkpoint',
        'latest_inline_eval_json',
        'final_eval_json',
        'error_message',
    }

def _group_keys(rows: list[dict]) -> list[str]:
    '''
    helper to get keys to group the data
    '''
    preferred = [
        'benchmark_name',
        'experiment_name',
        'reconstruction_target',
        'model_name',
        'renderer_model',
        'forward_backend',
        'active_dataset_dir',
        'train_num_views',
        'train_max_steps',
    ]

    keys = set()
    for row in rows:
        keys.update(row.keys())

    sweep_keys = sorted(k for k in keys if k.startswith('sweep.'))
    return [k for k in preferred if k in keys] + sweep_keys
    # if not rows:
    #     return []
    
    # keys = set()
    # for row in rows:
    #     keys.update(row.keys())
    
    # group_keys = []
    # for key in sorted(keys):
    #     if _is_metric_key(key):
    #         continue
    #     if _is_excluded_group_key(key):
    #         continue
    #     group_keys.append(key)

    # return group_keys

def aggregate_benchmark_dir(benchmark_dir: Union[Path, str]) -> list[dict]:
    '''
    writes csv summaries of all runs
    '''

    benchmark_dir = Path(benchmark_dir)
    rows = collect_run_summaries(benchmark_dir)
    
    # raw rows
    write_csv(rows, benchmark_dir / 'aggregate.csv')

    # per-run grouped summary over seeds
    group_keys = _group_keys(rows)

    grouped = {}
    for row in rows:
        key = tuple(_freeze_group_value(row.get(k)) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    # construct grouped version of rows
    grouped_rows = []
    for key_tuple, group in grouped.items():
        first_row = group[0]
        out = {k: _csv_safe_value(first_row.get(k)) for k in group_keys}
        out['num_seeds'] = len(group)

        # grab all relevant metric keys from summary
        metric_keys = sorted({
            k
            for row in group
            for k in row.keys()
            if _is_metric_key(k)
        })

        # compute grouped mean and std for the metrics
        for mk in metric_keys:
            # pick out vals, accounts for nonnumerical vals like strings
            vals = []
            for row in group:
                val = row.get(mk, None)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                vals.append(val)
            # then only do stats on numerical ones
            if vals:
                out[f'{mk}/mean'] = sum(vals) / len(vals)
                if len(vals) > 1:
                    mean = out[f'{mk}/mean']
                    var = sum((v-mean)**2 for v in vals) / len(vals)
                    out[f'{mk}/std'] = var**0.5
                else:
                    out[f'{mk}/std'] = 0.0

        grouped_rows.append(out)

    write_csv(grouped_rows, benchmark_dir / 'aggregate_grouped.csv')
    return rows



