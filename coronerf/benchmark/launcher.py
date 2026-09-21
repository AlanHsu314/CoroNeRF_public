##########################################################
## launches all runs sequentially for a benchmark
##########################################################

from __future__ import annotations
from typing import Union

from pathlib import Path

from .config import (
    load_yaml,
    save_yaml,
    save_json,
    apply_overrides,
)
from .spec import load_benchmark_spec
from .expand import expand_benchmark_spec
from .run_single import run_single_experiment
from .aggregate import aggregate_benchmark_dir

from ..experiments.resolve_paths import resolve_cfg_paths

from .dataset_compare import compare_vpgen_benchmark

from .history_plots import generate_best_run_history_plots

from .condition_compare import generate_condition_compare_plots

def _run_dir_for(output_dir: Path, run_name: str) -> Path:
    return output_dir / 'runs' / run_name

def _is_completed(run_dir: Path) -> bool:
    summary_path = run_dir / 'summary.json'
    if not summary_path.exists():
        return False
    
    import json
    with summary_path.open('r') as f:
        summary = json.load(f)
    return summary.get('status') == 'completed'

def run_benchmark(spec_path: Union[Path, str]) -> list[dict]:
    # load benchmark spec
    # this should make all paths in spec absolute as well
    spec = load_benchmark_spec(spec_path)
    bench = spec['benchmark']

    # process benchmark specs
    output_dir = Path(bench['output_dir'])
    output_dir.mkdir(exist_ok = True, parents = True)
    (output_dir / 'runs').mkdir(parents = True, exist_ok = True)

    save_yaml(spec, output_dir / 'spec_snapshot.yaml')

    # resolve base cfg to base cfg FIRST
    base_cfg = load_yaml(bench['base_config']) 
    base_cfg = resolve_cfg_paths(base_cfg, root = Path(bench['base_config']).parent)
    
    expanded = expand_benchmark_spec(spec)
    spec_root = Path(bench['_spec_dir'])

    # summarize benchmark
    manifest = {
        'benchmark_name': bench['name'],
        'spec_path': str(spec_path),
        'base_config': str(bench['base_config']),
        'num_runs': len(expanded),
        'runs': [],
    }   

    results = []

    # send out runs
    for run_spec in expanded:
        run_name = run_spec['run_name']
        run_dir = _run_dir_for(output_dir, run_name)

        row = {
            'run_name': run_name,
            'experiment_name': run_spec['experiment_name'],
            'seed': run_spec['seed'],
            'run_dir': str(run_dir),
            'status': 'pending',
        }

        # skip if completed (when rerunning)
        if _is_completed(run_dir):
            row['status'] = 'skipped_completed'
            manifest['runs'].append(row)
            continue

        # override config
        resolved = apply_overrides(base_cfg, run_spec['overrides'])
        # IMPORTANT: resolve config now to spec root
        resolved = resolve_cfg_paths(resolved, root = spec_root)

        # add necessary additional overrides / keys
        resolved['base_dir'] = str(output_dir / 'runs')
        resolved['run_dir_name'] = run_name
        resolved['use_dummy_dir'] = False

        # add benchmark meta to run config
        resolved.setdefault('benchmark_meta', {})
        resolved['benchmark_meta']['benchmark_name'] = run_spec['benchmark_name']
        resolved['benchmark_meta']['experiment_name'] = run_spec['experiment_name']
        resolved['benchmark_meta']['run_name'] = run_name
        resolved['benchmark_meta']['sweep_keys'] = run_spec.get('sweep_keys', [])
        resolved['benchmark_meta']['sweep_values'] = run_spec.get('sweep_values', {})


        # save the run spec
        run_dir.mkdir(exist_ok = True, parents = True)
        save_json(run_spec, run_dir / 'run_spec.json')

        # dispatch
        summary = run_single_experiment(resolved, cfg_path = bench['base_config'])
        results.append(summary)

        # save status
        row['status'] = summary.get('status', 'unknown')
        manifest['runs'].append(row)

        # save running update of manifest
        save_json(manifest, output_dir / 'manifest.json')

    # final manifest write
    save_json(manifest, output_dir / 'manifest.json')

    # aggregate at the end
    aggregate_benchmark_dir(output_dir)

    # optional best-run family history plots
    history_plots_cfg = bench.get('history_plots', {})
    if history_plots_cfg.get('enabled', False):
        generate_best_run_history_plots(output_dir, history_plots_cfg)

    # optional post-training condition compare plots
    post_condition_compare = bench.get('post_condition_compare', {})
    if post_condition_compare.get('enabled', False):
        generate_condition_compare_plots(output_dir, post_condition_compare)

    # optional post-vpgen dataset compare
    post_vpgen_compare = bench.get('post_vpgen_compare', {})
    mode_fixed = bench.get('fixed', {}).get('mode', None)

    if mode_fixed == 'vpgen' and post_vpgen_compare.get('enabled', False):
        reference_experiment = post_vpgen_compare.get('reference_experiment', None)
        if reference_experiment is None:
            raise ValueError(
                "benchmark.post_vpgen_compare.enabled=True requires "
                "benchmark.post_vpgen_compare.reference_experiment"
            )

        compare_vpgen_benchmark(
            benchmark_dir=output_dir,
            reference_experiment=reference_experiment,
            out_subdir=post_vpgen_compare.get('out_subdir', 'dataset_compare'),
            save_num_worst_views=int(post_vpgen_compare.get('save_num_worst_views', 3)),
        )

    return results
