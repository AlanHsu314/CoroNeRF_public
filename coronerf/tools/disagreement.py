from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..identifiability import (
    generate_seed_disagreement_figure,
    generate_seed_disagreement_batch,
    generate_disagreement_radial_figure,
    generate_disagreement_decomp_figure,
    generate_disagreement_conditions_figure,
    generate_radial_image_figure,
    generate_mode_projection,
    run_baselines,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_DIR = _REPO_ROOT / "configs" / "identifiability"

# Unified seed-disagreement / ensemble diagnostics CLI
# (replaces tools/seed_disagreement.py and tools/paper_disagreement.py).
#   python -m coronerf.tools.disagreement --kind hero         --spec diag_seed_disagreement.yaml --device cuda
#   python -m coronerf.tools.disagreement --kind radial       --spec <spec>.yaml --device cuda
#   python -m coronerf.tools.disagreement --kind decomp       --spec <spec>.yaml --device cuda
#   python -m coronerf.tools.disagreement --kind conditions   --spec ../paper_figures/paper_F_conditions.yaml --device cuda
#   python -m coronerf.tools.disagreement --kind radial_image --spec <spec>.yaml --device cuda
#   python -m coronerf.tools.disagreement --kind batch        --spec seed_disagreement_batch_paper_F.yaml --device cuda
#   python -m coronerf.tools.disagreement --kind mode_projection --spec mode_projection_paper_F.yaml
#   python -m coronerf.tools.disagreement --kind baselines --spec baselines_paper_F.yaml --device cuda

_KIND = {
    "hero":         generate_seed_disagreement_figure,
    "batch":        generate_seed_disagreement_batch,   # NEW: all conditions -> per-id folders
    "radial":       generate_disagreement_radial_figure,
    "decomp":       generate_disagreement_decomp_figure,
    "conditions":   generate_disagreement_conditions_figure,
    "radial_image": generate_radial_image_figure,
    "mode_projection": generate_mode_projection,
    "baselines":       run_baselines,          # sigma_ens vs physical proxies (emissivity/density/Fisher) + AUSE
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
    ap = argparse.ArgumentParser(description="Seed-disagreement / ensemble diagnostics (hero + a/b/c).")
    ap.add_argument("--kind", required=True, choices=list(_KIND))
    ap.add_argument("--spec", required=True, help="YAML in configs/identifiability/ (bare name ok) or a path.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--verbosity", type=int, default=1)
    a = ap.parse_args()
    _setup_logging(a.verbosity)
    _KIND[a.kind](spec_path=_resolve_spec(a.spec), device_override=a.device, output_dir_override=a.output_dir)


if __name__ == "__main__":
    main()
