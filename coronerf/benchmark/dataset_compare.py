##################################################
### metrics to compare datasets in vpgen
##################################################

# most chatGPT

from __future__ import annotations
import re

from pathlib import Path
from typing import Union
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

# IO

def _read_json(path: Path) -> dict:
    with path.open('r') as f:
        return json.load(f)
    
def _write_json(data: dict | list, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=2)

def _write_csv(rows: list[dict], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open('w', newline='') as f:
            f.write('')
        return

    keys = sorted({k for row in rows for k in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _find_vp_dir(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / 'viewpoints_gt',
        dataset_dir / 'viewpoints',
    ]
    # will prefer viewpoints_gt, fallback to viewpoints
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f'Could not find viewpoints_gt/ or viewpoints/ under {dataset_dir}')

def _sorted_vp_files(vp_dir: Path) -> list[Path]:
    pat = re.compile(r'^vp_(\d+)\.npz$')

    files = []
    for p in vp_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m is not None:
            files.append((int(m.group(1)), p))

    if not files:
        raise FileNotFoundError(f'No vp_<number>.npz files found in {vp_dir}')

    files.sort(key=lambda t: t[0])
    return [p for _, p in files]

def _safe_log_mse(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    da = np.log10(np.maximum(a, eps))
    db = np.log10(np.maximum(b, eps))
    return float(np.mean((da - db) ** 2))

### main functions

def compare_two_datasets(
        ref_dataset_dir: Union[str, Path],
        test_dataset_dir: Union[str, Path],
        out_dir: Union[str, Path],
        save_num_worst_views: int = 3,
) -> dict:
    '''
    compare generated datasets per-view
    assumes matching vp_*.npz filenames and intensity shapes
    '''
    
    # load datasets and info
    ref_dataset_dir = Path(ref_dataset_dir)
    test_dataset_dir = Path(test_dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_vp_dir = _find_vp_dir(ref_dataset_dir)
    test_vp_dir = _find_vp_dir(test_dataset_dir)

    ref_files = _sorted_vp_files(ref_vp_dir)
    test_files = _sorted_vp_files(test_vp_dir)

    ref_names = [p.name for p in ref_files]
    test_names = [p.name for p in test_files]
    if ref_names != test_names:
        raise ValueError('Reference and test dataset do not have matching vp_*.npz file lists')

    # list to save summary of each vp compare
    per_view_rows = []

    # main loop
    for ref_path, test_path in zip(ref_files, test_files):
        # load images
        ref_npz = np.load(ref_path, allow_pickle=False)
        test_npz = np.load(test_path, allow_pickle=False)

        ref_img = ref_npz['intensities'].astype(np.float64)
        test_img = test_npz['intensities'].astype(np.float64)

        if ref_img.shape != test_img.shape:
            raise ValueError(
                f'Intensity shape mismatch for {ref_path.name}: '
                f'{ref_img.shape} vs {test_img.shape}'
            )

        # metrics
        diff = test_img - ref_img
        abs_diff = np.abs(diff)
        sq_diff = diff ** 2

        # avg over img
        C = ref_img.shape[-1]
        mse_per_channel = [float(np.mean(sq_diff[..., c])) for c in range(C)]
        mae_per_channel = [float(np.mean(abs_diff[..., c])) for c in range(C)]
        max_abs_per_channel = [float(np.max(abs_diff[..., c])) for c in range(C)]
        logmse_per_channel = [_safe_log_mse(ref_img[..., c], test_img[..., c]) for c in range(C)]

        # append to running list
        view_idx = int(ref_path.stem.split('_')[-1])

        per_view_rows.append({
            'view': view_idx,
            'filename': ref_path.name,
            'H': int(ref_img.shape[0]),
            'W': int(ref_img.shape[1]),
            'C': int(ref_img.shape[2]),
            'mse_mean_over_channels': float(np.mean(mse_per_channel)),
            'mae_mean_over_channels': float(np.mean(mae_per_channel)),
            'max_abs_mean_over_channels': float(np.mean(max_abs_per_channel)),
            'logmse_mean_over_channels': float(np.mean(logmse_per_channel)),
            **{f'mse_ch{c}': mse_per_channel[c] for c in range(C)},
            **{f'mae_ch{c}': mae_per_channel[c] for c in range(C)},
            **{f'max_abs_ch{c}': max_abs_per_channel[c] for c in range(C)},
            **{f'logmse_ch{c}': logmse_per_channel[c] for c in range(C)},
        })

    # aggregate over all rows
    mse_vals = [r['mse_mean_over_channels'] for r in per_view_rows]
    mae_vals = [r['mae_mean_over_channels'] for r in per_view_rows]
    max_abs_vals = [r['max_abs_mean_over_channels'] for r in per_view_rows]
    logmse_vals = [r['logmse_mean_over_channels'] for r in per_view_rows]

    summary = {
        'reference_dataset_dir': str(ref_dataset_dir),
        'test_dataset_dir': str(test_dataset_dir),
        'num_views': len(per_view_rows),
        'mse_mean_over_views': float(np.mean(mse_vals)),
        'mse_std_over_views': float(np.std(mse_vals)),
        'mae_mean_over_views': float(np.mean(mae_vals)),
        'mae_std_over_views': float(np.std(mae_vals)),
        'max_abs_mean_over_views': float(np.mean(max_abs_vals)),
        'max_abs_std_over_views': float(np.std(max_abs_vals)),
        'logmse_mean_over_views': float(np.mean(logmse_vals)),
        'logmse_std_over_views': float(np.std(logmse_vals)),
    }

    # write per-view
    _write_csv(per_view_rows, out_dir / 'per_view_metrics.csv')
    _write_json(per_view_rows, out_dir / 'per_view_metrics.json')

    # write avg over views
    _write_json(summary, out_dir / 'summary.json')

    # save worst-case diff plots
    if save_num_worst_views > 0 and per_view_rows:
        # grab top X worst views
        worst = sorted(
            per_view_rows,
            key=lambda r: r['logmse_mean_over_channels'],
            reverse=True,
        )[:save_num_worst_views]

        # plot for each worst
        for row in worst:
            ref_path = ref_vp_dir / row['filename']
            test_path = test_vp_dir / row['filename']

            ref_npz = np.load(ref_path, allow_pickle=False)
            test_npz = np.load(test_path, allow_pickle=False)

            # compute
            ref_img = ref_npz['intensities'].astype(np.float64)
            test_img = test_npz['intensities'].astype(np.float64)
            diff = test_img - ref_img

            # plot
            C = ref_img.shape[-1]
            for c in range(C):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(ref_img[..., c])
                axs[0].set_title(f'ref ch{c}')
                axs[1].imshow(test_img[..., c])
                axs[1].set_title(f'test ch{c}')
                im = axs[2].imshow(diff[..., c])
                axs[2].set_title(f'diff ch{c}')
                fig.colorbar(im, ax=axs[2])
                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.suptitle(f'view {row["view"]} | {row["filename"]}')
                fig.tight_layout()
                fig.savefig(out_dir / f'worst_view_{row["view"]:03d}_ch{c}.png', dpi=150)
                plt.close(fig)
        
    return {
        'summary': summary,
        'per_view_rows': per_view_rows,
    }

def compare_vpgen_benchmark(
    benchmark_dir: Union[str, Path],
    reference_experiment: str,
    out_subdir: str = 'dataset_compare',
    save_num_worst_views: int = 3,
) -> dict:
    '''
    compare all vpgen runs in a benchmark against one reference experiment
    '''

    # define benchmark
    benchmark_dir = Path(benchmark_dir)
    runs_dir = benchmark_dir / 'runs'
    if not runs_dir.exists():
        raise FileNotFoundError(f'No runs dir at {runs_dir}')
    
    # grab summaries from benchmark
    summaries = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / 'summary.json'
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        if summary.get('status') != 'completed':
            continue
        summaries.append(summary)

    if not summaries:
        raise ValueError('No completed run summaries found for dataset comparison')
    
    # there should only be 1 experiment in benchmark with name reference_experiment
    ref_candidates = [s for s in summaries if s.get('experiment_name') == reference_experiment]
    if len(ref_candidates) != 1:
        raise ValueError(
            f'Expected exactly one completed reference run for experiment_name={reference_experiment}, '
            f'found {len(ref_candidates)}'
        )
    
    # grab reference experiment, and also get the dataset used in that experiment
    ref = ref_candidates[0]
    ref_dataset_dir = ref.get('active_dataset_dir')
    if not ref_dataset_dir:
        raise ValueError('Reference run summary does not contain active_dataset_dir')

    # out dir
    compare_root = benchmark_dir / out_subdir
    compare_root.mkdir(parents=True, exist_ok=True)

    compare_rows = []
    compare_json = {
        'benchmark_dir': str(benchmark_dir),
        'reference_experiment': reference_experiment,
        'reference_run_name': ref.get('run_name'),
        'reference_dataset_dir': ref_dataset_dir,
        'comparisons': [],
    }

    # compare against each other run
    for s in summaries:
        if s.get('run_name') == ref.get('run_name'):
            continue

        test_dataset_dir = s.get('active_dataset_dir')
        if not test_dataset_dir:
            continue

        out_dir = compare_root / s['run_name']

        # compute call
        result = compare_two_datasets(
            ref_dataset_dir=ref_dataset_dir,
            test_dataset_dir=test_dataset_dir,
            out_dir=out_dir,
            save_num_worst_views=save_num_worst_views,
        )

        # save
        row = {
            'reference_experiment': reference_experiment,
            'reference_run_name': ref.get('run_name'),
            'reference_dataset_dir': ref_dataset_dir,
            'test_experiment': s.get('experiment_name'),
            'test_run_name': s.get('run_name'),
            'test_dataset_dir': test_dataset_dir,
            **result['summary'],
        }
        compare_rows.append(row)
        compare_json['comparisons'].append(row)

    # final save
    _write_csv(compare_rows, compare_root / 'dataset_compare_summary.csv')
    _write_json(compare_json, compare_root / 'dataset_compare_summary.json')

    return compare_json











