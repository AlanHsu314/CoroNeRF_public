from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..identifiability import (
    build_operator_from_spec, diagnose_from_spec,
    generate_identifiability_run_figure, generate_identifiability_sweeps_figure,
    run_conditions_sweep, aggregate_conditions_sweep,
    generate_floor_sweep,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_DIR = _REPO_ROOT / "configs" / "identifiability"

'''
# (once, expensive) build the operator if you haven't on the new path:
python -m coronerf.tools.identifiability --kind build    --spec idops_noise_x9_operator.yaml --device cuda
# (cheap, re-runnable) all diagnostics from the saved operator:
python -m coronerf.tools.identifiability --kind diagnose --spec idops_noise_x9_diagnostics.yaml

python -m coronerf.tools.identifiability --kind sweep     --spec sweep_conditions_paper_F.yaml --device cuda
python -m coronerf.tools.identifiability --kind aggregate --spec sweep_conditions_paper_F.yaml

python -m coronerf.tools.identifiability --kind sweep     --spec sweep_operating_points_paper_F.yaml --device cuda
python -m coronerf.tools.identifiability --kind aggregate --spec sweep_operating_points_paper_F.yaml

python -m coronerf.tools.identifiability --kind floor_sweep --spec floor_sweep_paper_F.yaml --device cuda

'''

_KIND = {
    "build":     build_operator_from_spec,      # expensive, once
    "diagnose":  diagnose_from_spec,            # cheap, re-runnable; reads the saved operator
    "sweep":     run_conditions_sweep,          # NEW: build/derive per condition -> sweep_manifest.json
    "aggregate": aggregate_conditions_sweep,    # NEW: manifest -> cross-condition figures
    "run":       generate_identifiability_run_figure,    # legacy
    "sweeps":    generate_identifiability_sweeps_figure, # legacy eff-rank FIGURE (note: != "sweep")
    "floor_sweep": generate_floor_sweep,                 # sweeps noise floor, internal check on floor-sensitive quantities
}


def _resolve_spec(spec: str) -> Path:
    p = Path(spec)
    cands = [p] if p.is_absolute() else [Path.cwd() / p, _SPEC_DIR / p]
    for c in cands:
        if c.exists():
            return c.resolve()
    return (_SPEC_DIR / p).resolve()


def _setup_logging(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.INFO if int(verbosity) >= 1 else logging.WARNING,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
    )


def main():
    p = argparse.ArgumentParser(description="Linearized identifiability analysis (operator build + diagnostics).")
    p.add_argument("--kind", required=True, choices=list(_KIND))
    p.add_argument("--spec", required=True, help="YAML in configs/identifiability/ (bare name ok) or a path.")
    p.add_argument("--device", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--verbosity", type=int, default=1)
    a = p.parse_args()
    _setup_logging(a.verbosity)
    _KIND[a.kind](spec_path=_resolve_spec(a.spec), device_override=a.device, output_dir_override=a.output_dir)


if __name__ == "__main__":
    main()