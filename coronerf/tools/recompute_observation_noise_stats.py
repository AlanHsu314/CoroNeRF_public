# manual recompute training dataset noise observation statistics

from __future__ import annotations

import argparse
from pathlib import Path

from .. import ExperimentDirs, DotDict
from ..benchmark.aggregate import collect_run_summaries
from ..benchmark.config import load_yaml, normalize_config
from ..benchmark.summary import find_latest_checkpoint
from ..data.noise import save_noise_json
from ..experiments.resolve_paths import resolve_cfg_paths
from ..pipeline.state import PipelineState
from ..pipeline.builders.data import load_data

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_BENCHMARK_DIR = REPO_ROOT / "runs_benchmarks"

def _resolve_run_dir(row: dict, benchmark_dir: Path) -> Path | None:
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


def recompute_noise_stats_for_run(run_dir: Path, device: str = 'cpu', output_name: str = 'observation_noise.json') -> bool:
    cfg = load_yaml(run_dir / 'config.yaml')
    cfg = normalize_config(cfg)

    dataset_dir = cfg.get('dataset_dir', None)
    if isinstance(dataset_dir, str) and not Path(dataset_dir).is_absolute():
        cfg = resolve_cfg_paths(cfg, root=run_dir)

    cfg.setdefault('io', {})
    cfg['io']['no_save'] = True
    cfg['device'] = str(device)

    state = PipelineState(
        DotDict(cfg),
        dirs=ExperimentDirs(
            run_dir=run_dir,
            checkpoints=run_dir / 'checkpoints',
            logs=run_dir / 'logs',
            figs=run_dir / 'figs',
            artifacts=run_dir / 'artifacts',
        ),
        wb_run=None,
    )

    # load_data will apply observation mismatch/noise hooks and compute noise meta
    load_data(state, **state.cfg.load_data_kwargs)

    noise_meta = getattr(state, 'observation_noise_meta', None)
    if noise_meta is None:
        return False

    out_path = run_dir / 'artifacts' / 'train' / output_name
    save_noise_json(noise_meta, out_path)
    print(f'[noise-stats] wrote {out_path}')
    return True


def recompute_noise_stats_for_benchmark(
    benchmark_dir: Path,
    device: str = 'cpu',
    output_name: str = 'observation_noise.json',
    only_experiments: list[str] | None = None,
) -> None:
    rows = collect_run_summaries(benchmark_dir)
    completed = [r for r in rows if r.get('status') == 'completed']

    for row in completed:
        exp_name = row.get('experiment_name')
        if only_experiments and exp_name not in only_experiments:
            continue

        run_dir = _resolve_run_dir(row, benchmark_dir)
        if run_dir is None:
            print(f'[noise-stats] skipping unresolved run_dir for {row.get("run_name")}')
            continue

        try:
            recompute_noise_stats_for_run(run_dir, device=device, output_name=output_name)
        except Exception as e:
            print(f'[noise-stats] FAILED {run_dir}: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_dir', type=str, default=None)
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_name', type=str, default='observation_noise.json')
    parser.add_argument('--only_experiments', nargs='*', default=None)
    args = parser.parse_args()

    if (args.run_dir is None) and (args.benchmark_dir is None):
        raise ValueError('Provide either --run_dir or --benchmark_dir')

    if args.run_dir is not None:
        recompute_noise_stats_for_run(
            RUNS_BENCHMARK_DIR / args.benchmark_dir / "runs" / args.run_dir,
            device=args.device,
            output_name=args.output_name,
        )
    else:
        recompute_noise_stats_for_benchmark(
            RUNS_BENCHMARK_DIR / args.benchmark_dir,
            device=args.device,
            output_name=args.output_name,
            only_experiments=args.only_experiments,
        )


if __name__ == '__main__':
    main()