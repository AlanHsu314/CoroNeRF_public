from __future__ import annotations
from pathlib import Path

def _coerce_for_compare(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return str(x)

def _values_match(a, b, atol: float = 1.0e-9) -> bool:
    aa, bb = _coerce_for_compare(a), _coerce_for_compare(b)
    if aa is None or bb is None:
        return aa is bb
    if isinstance(aa, float) and isinstance(bb, float):
        return abs(aa - bb) <= atol
    return str(aa) == str(bb)

def _row_matches_where(row: dict, where) -> bool:
    if not where:
        return True
    for key, expected in dict(where).items():
        key = str(key)
        if key in row:
            actual = row.get(key)
        elif f"sweep.{key}" in row:
            actual = row.get(f"sweep.{key}")
        else:
            return False
        if not _values_match(actual, expected):
            return False
    return True

def _collect_condition_seed_rows(rows: list[dict], cond_cfg: dict) -> list[dict]:
    """All completed seed-runs that form one condition/ensemble."""
    completed = [r for r in rows if r.get("status") == "completed"]
    run_dirs, run_names = cond_cfg.get("run_dirs"), cond_cfg.get("run_names")
    if run_dirs:
        out = [{"status": "completed", "run_dir": str(Path(rd)), "run_name": Path(rd).name,
                "experiment_name": cond_cfg.get("experiment", Path(rd).name), "seed": None} for rd in run_dirs]
    elif run_names:
        wanted = {str(x) for x in run_names}
        out = [r for r in completed if str(r.get("run_name")) in wanted]
    else:
        exp = cond_cfg.get("experiment")
        if exp is None:
            raise ValueError("condition requires one of: experiment, run_names, run_dirs")
        where = cond_cfg.get("where") or cond_cfg.get("filters")
        out = [r for r in completed if str(r.get("experiment_name")) == str(exp) and _row_matches_where(r, where)]
    out = sorted(out, key=lambda r: str(r.get("seed")))
    max_seeds = cond_cfg.get("max_seeds")
    if max_seeds is not None:
        out = out[: int(max_seeds)]
    min_seeds = int(cond_cfg.get("min_seeds", 2))
    if len(out) < min_seeds:
        raise ValueError(f"condition matched {len(out)} completed seed(s) < min_seeds={min_seeds}: {cond_cfg}")
    return out