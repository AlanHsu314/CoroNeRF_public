from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_QTY_TEX = {
    "ne":   {"field": r"$\log_{10} n_e$", "err": r"$|\Delta \log_{10} n_e|$", "std": r"$\sigma(\log_{10} n_e)$"},
    "temp": {"field": r"$\log_{10} T$",   "err": r"$|\Delta \log_{10} T|$",   "std": r"$\sigma(\log_{10} T)$"},
}

def _qty_tex(q: str, kind: str = "field") -> str:
    return _QTY_TEX.get(str(q), {}).get(kind, str(q))

def _resolve_path(path, root) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (Path(root) / p).resolve()

def _slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    return text.strip("_") or "panel"

def _savefig(fig, out_base, formats, dpi) -> list[str]:
    paths = []
    for ext in formats:
        ext = str(ext).lstrip(".")
        p = Path(out_base).with_suffix(f".{ext}")
        fig.savefig(p, dpi=int(dpi), bbox_inches="tight")
        paths.append(str(p))
    return paths

def _cmap_with_bad(name, bad_color):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(bad_color)
    return cmap

def _maybe_suptitle(fig, text):
    if text:
        fig.suptitle(str(text))

def _reorder_cols(arr, order):
    return np.asarray(arr)[:, order]

def _longitude_order(lon_deg, mode):
    lon_deg = np.asarray(lon_deg, dtype=float)
    if mode in {"minus180_180", "-180_180", "180"}:
        lon_plot = ((lon_deg + 180.0) % 360.0) - 180.0
        order = np.argsort(lon_plot)
        return lon_plot[order], order
    if mode in {"zero_360", "0_360", "native"}:
        return lon_deg, np.arange(lon_deg.size)
    raise ValueError(f"Unknown longitude_mode={mode}")

def _finite_percentile(vals, pct, fallback=(0.0, 1.0)):
    vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(fallback[0]), float(fallback[1])
    lo = float(np.nanpercentile(vals, float(pct[0]))); hi = float(np.nanpercentile(vals, float(pct[1])))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi

def _correlations(x, y) -> dict:
    x = np.asarray(x); y = np.asarray(y)
    out = {"spearman": None, "pearson": None, "n": int(x.size)}
    if x.size < 3:
        return out
    try:
        from scipy.stats import spearmanr, pearsonr
        out["spearman"] = float(spearmanr(x, y).correlation); out["pearson"] = float(pearsonr(x, y)[0])
    except Exception:
        try:
            out["pearson"] = float(np.corrcoef(x, y)[0, 1])
            xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
            out["spearman"] = float(np.corrcoef(xr, yr)[0, 1])
        except Exception:
            pass
    return out

def diagnostics_dir(benchmark_dir, kind: str, name: str | None = None) -> Path:
    """Canonical home for benchmark/ensemble-derived diagnostics:
       <benchmark_dir>/diagnostics/<kind>[/<name>].  One rule, every diagnostic."""
    d = Path(benchmark_dir) / "diagnostics" / kind
    return (d / name) if name else d
