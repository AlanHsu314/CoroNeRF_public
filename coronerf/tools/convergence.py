from __future__ import annotations
import argparse, logging
from pathlib import Path
from ..paper.dataset_convergence import generate_stepsize_convergence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_DIRS = [_REPO_ROOT / "configs" / "paper_figures", _REPO_ROOT / "configs" / "diagnostics", _REPO_ROOT / "configs"]
_KIND = {"stepsize": generate_stepsize_convergence}   # grow this for future dataset diagnostics

'''
python -m coronerf.tools.convergence --kind stepsize --spec convergence_stepsize_paperF.yaml --device cuda
'''


def _resolve_spec(spec: str) -> Path:
    p = Path(spec)
    if p.is_absolute(): return p.resolve()
    for c in [Path.cwd() / p, *(d / p for d in _SPEC_DIRS)]:
        if c.exists(): return c.resolve()
    return (_SPEC_DIRS[0] / p).resolve()


def main():
    ap = argparse.ArgumentParser(description="Dataset / forward-model convergence diagnostics.")
    ap.add_argument("--kind", default="stepsize", choices=list(_KIND))
    ap.add_argument("--spec", required=True, help="YAML (bare name searches configs/paper_figures, diagnostics, configs).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--verbosity", type=int, default=1)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if int(a.verbosity) >= 1 else logging.WARNING,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    _KIND[a.kind](spec_path=_resolve_spec(a.spec), device_override=a.device, output_dir_override=a.output_dir)


if __name__ == "__main__":
    main()