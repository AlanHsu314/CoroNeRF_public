from __future__ import annotations
import argparse, logging
from pathlib import Path
from ..identifiability.oppoints import compare_operating_points
from ..identifiability.oppoint_spectrum import generate_oppoint_spectrum


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_DIRS = [_REPO_ROOT / "configs" / "identifiability", _REPO_ROOT / "configs" / "paper_figures", _REPO_ROOT / "configs"]
_KIND = {"compare": compare_operating_points, "spectrum": generate_oppoint_spectrum}

'''
python -m coronerf.tools.operating_points --kind compare --spec oppoints_mstar_mhat.yaml --device cuda
python -m coronerf.tools.operating_points --kind spectrum --spec oppoint_spectrum_paper_F.yaml --device cuda
'''


def _resolve_spec(spec: str) -> Path:
    p = Path(spec)
    if p.is_absolute(): return p.resolve()
    for c in [Path.cwd() / p, *(d / p for d in _SPEC_DIRS)]:
        if c.exists(): return c.resolve()
    return (_SPEC_DIRS[0] / p).resolve()


def main():
    ap = argparse.ArgumentParser(description="Operating-point (m* vs m̂) identifiability comparison.")
    ap.add_argument("--kind", default="compare", choices=list(_KIND))
    ap.add_argument("--spec", required=True, help="YAML (bare name searches configs/identifiability, paper_figures, configs).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--verbosity", type=int, default=1)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if int(a.verbosity) >= 1 else logging.WARNING,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    _KIND[a.kind](spec_path=_resolve_spec(a.spec), device_override=a.device, output_dir_override=a.output_dir)


if __name__ == "__main__":
    main()