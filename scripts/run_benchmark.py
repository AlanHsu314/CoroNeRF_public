######################################################
### BENCHMARK CLI WRAPPER
######################################################

import argparse

from coronerf.benchmark.launcher import run_benchmark

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1] # parents[1] is the repo root
SPEC_DIR = REPO_ROOT / "configs" / "benchmarks"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, required=True)
    args = parser.parse_args()

    spec_path = SPEC_DIR / args.spec
    print(str(spec_path))
    results = run_benchmark(spec_path)
    print(f"Finished benchmark. Num completed/attempted runs this invocation: {len(results)}")


if __name__ == "__main__":
    main()



















