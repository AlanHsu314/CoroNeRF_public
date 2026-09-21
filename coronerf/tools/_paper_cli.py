from __future__ import annotations

import logging
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "configs" / "paper_figures"


def resolve_paper_spec_path(spec: str | Path) -> Path:
    """
    Resolve a paper-figure YAML spec.

    Allows:
      python -m ... --spec paper_A_density_shell_r1p5.yaml
      python -m ... --spec configs/paper_figures/paper_A_density_shell_r1p5.yaml
      python -m ... --spec /absolute/path/to/spec.yaml
    """
    p = Path(spec)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path.cwd() / p,
            SPEC_DIR / p,
        ])

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    # Return the most likely intended path so the eventual FileNotFoundError is clear.
    if p.is_absolute():
        return p
    return (SPEC_DIR / p).resolve()


def setup_paper_logging(verbosity: int = 1) -> None:
    level = logging.INFO if int(verbosity) >= 1 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def figure_out_dir(raw: dict, override=None):
    """Paper-figure output dir derived from the config's top-level `figure_id: {group, slug}`.
    Returns None if figure_id is absent -> caller lets the builder use its own cfg.output_dir
    (so un-migrated configs keep working). CLI --output_dir always wins."""
    if override:
        p = Path(override)
        return (p if p.is_absolute() else (Path.cwd() / p)).resolve()
    fid = (raw.get("figure_id") or {})
    group, slug = fid.get("group"), fid.get("slug")
    # require figure_id to have group and slug for correct storage of generated figure
    if not group or not slug:
        raise ValueError("paper-figure config must declare figure_id: {group, slug} "
                         "(or pass --output_dir); refusing to fall back into diagnostics.")
    d = REPO_ROOT / "paper_outputs" / "figures" / str(group) / str(slug)
    d.mkdir(parents=True, exist_ok=True)
    return d