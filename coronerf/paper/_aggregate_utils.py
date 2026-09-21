from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_grouped_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grouped aggregate CSV does not exist: {path}")
    return pd.read_csv(path)


def resolve_metric_column(df: pd.DataFrame, metric: str, stat: str = "mean") -> str:
    """
    Resolve either:
      final/foo/bar/mean
    or:
      final/foo/bar with stat='mean'
    """
    metric = str(metric)

    if metric in df.columns:
        return metric

    candidate = f"{metric}/{stat}"
    if candidate in df.columns:
        return candidate

    raise KeyError(
        f"Could not find metric column {metric!r} or {candidate!r}. "
        f"Available columns containing the last token: "
        f"{[c for c in df.columns if metric.split('/')[-1] in c][:20]}"
    )


def _coerce_compare_value(x: Any):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        return float(x)
    except Exception:
        return str(x)


def values_match(a: Any, b: Any, atol: float = 1.0e-9) -> bool:
    aa = _coerce_compare_value(a)
    bb = _coerce_compare_value(b)

    if aa is None or bb is None:
        return aa is bb

    if isinstance(aa, float) and isinstance(bb, float):
        return abs(aa - bb) <= atol

    return str(aa) == str(bb)


def filter_df(df: pd.DataFrame, where: dict | None) -> pd.DataFrame:
    if not where:
        return df

    mask = np.ones(len(df), dtype=bool)
    for key, expected in dict(where).items():
        key = str(key)
        if key not in df.columns:
            alt = f"sweep.{key}"
            if alt in df.columns:
                key = alt
            else:
                raise KeyError(f"Filter key {key!r} not found in grouped aggregate columns")

        mask &= np.asarray([values_match(v, expected) for v in df[key].to_numpy()], dtype=bool)

    return df.loc[mask].copy()


def lookup_one(df: pd.DataFrame, where: dict) -> pd.Series:
    sub = filter_df(df, where)
    if len(sub) != 1:
        raise ValueError(f"Expected exactly one row for filter {where}, found {len(sub)}")
    return sub.iloc[0]


def metric_mean_std(row: pd.Series, metric: str) -> tuple[float, float]:
    mean_col = resolve_metric_column(row.to_frame().T, metric, stat="mean")
    if mean_col.endswith("/mean"):
        std_col = mean_col[:-5] + "/std"
    else:
        std_col = f"{metric}/std"

    mean = float(row[mean_col])
    std = float(row[std_col]) if std_col in row.index and not pd.isna(row[std_col]) else 0.0
    return mean, std


def numeric_or_nan(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")