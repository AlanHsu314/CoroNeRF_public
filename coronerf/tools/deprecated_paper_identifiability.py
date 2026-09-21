from __future__ import annotations
import argparse
from ._paper_cli import resolve_paper_spec_path, setup_paper_logging
from ..identifiability import generate_identifiability_run_figure, generate_identifiability_sweeps_figure, build_operator_from_spec

'''
python -m coronerf.tools.paper_identifiability --kind run --spec configs/paper_figures/diag_identifiability_single.yaml --device cuda
python -m coronerf.tools.paper_identifiability --kind sweeps --spec configs/paper_figures/diag_identifiability_sweeps.yaml --device cuda
'''

_KIND = {"run": generate_identifiability_run_figure,
         "sweeps": generate_identifiability_sweeps_figure,
         "build": build_operator_from_spec,}


def main():
    p = argparse.ArgumentParser(description="Linearized identifiability analysis (Jacobian/Fisher/SVD).")
    p.add_argument("--kind", required=True, choices=list(_KIND))
    p.add_argument("--spec", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--verbosity", type=int, default=1)
    a = p.parse_args()
    setup_paper_logging(a.verbosity)
    _KIND[a.kind](spec_path=resolve_paper_spec_path(a.spec),
                  device_override=a.device, output_dir_override=a.output_dir)


if __name__ == "__main__":
    main()