from __future__ import annotations

import argparse
from pathlib import Path

from ..benchmark.config import load_yaml
from ..benchmark.condition_compare import generate_condition_compare_plots

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "configs" / "benchmarks"
RUNS_BENCHMARK_DIR = REPO_ROOT / "runs_benchmarks"

#  python -m coronerf.tools.benchmark_condition_compare --runs_benchmarks_dir noise --spec noise.yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_benchmarks_dir', type=str, required=True)
    parser.add_argument(
        '--spec',
        type=str,
        default=None,
        help='optional yaml containing either benchmark.post_condition_compare or the compare block directly',
    )
    args = parser.parse_args()

    cfg = {}
    if args.spec is not None:
        raw = load_yaml(SPEC_DIR / args.spec)
        if 'benchmark' in raw and 'post_condition_compare' in raw['benchmark']:
            cfg = raw['benchmark']['post_condition_compare']
        else:
            cfg = raw

    generate_condition_compare_plots(RUNS_BENCHMARK_DIR / args.runs_benchmarks_dir, cfg)


if __name__ == '__main__':
    main()