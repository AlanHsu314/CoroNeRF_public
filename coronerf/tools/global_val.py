from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from ..benchmark.aggregate import aggregate_benchmark_dir
from ..benchmark.condition_compare import _build_state_for_analysis, _release_state
from ..benchmark.config import load_yaml
from ..train.loops import compute_validation_loss

"python -m coronerf.tools.global_val --spec configs/benchmarks/2026_08_31_global_val_paper_B2.yaml"

logger = logging.getLogger("coroNeRF.tools.global_val")


def _global_loss_cfg(gcfg: dict) -> dict:
    """Fixed image-loss cfg used for ALL runs (cross-condition comparability)."""
    il = dict(gcfg.get("image_loss", {}) or {})
    il.setdefault("name", "hetero_gaussian_asinh")
    il.setdefault("lambda_asinh", 1.0)
    il.setdefault("lambda_gauss", 0.0)
    il.setdefault("asinh_scale_mode", "fixed")
    if "asinh_scale_by_channel" not in il:
        raise ValueError("global_validation.image_loss.asinh_scale_by_channel is required "
                         "(one fixed scale per global channel)")
    return il


def _state_overrides(gcfg: dict) -> dict:
    """Dotted cfg overrides that turn any run's state into the global-eval state."""
    ov: dict = {}
    ch = gcfg.get("channel_indices")
    if ch is not None:
        ov["load_data_kwargs.channel_indices"] = list(ch)     # <- the whole trick: score on the global channel set
    ds = gcfg.get("dataset_dir")
    if ds:
        ov["dataset_dir"] = str(ds)                            # optional: a bigger/denser reference set (absolute path)
    for k, v in dict(gcfg.get("load_data_overrides", {}) or {}).items():
        ov[f"load_data_kwargs.{k}"] = v                        # optional: e.g. holdout_num_views, arc.*
    return ov


def compute_global_val_for_run(run_dir, gcfg: dict, device=None, path_remap=None) -> tuple[float, dict]:
    """Render a trained run on the global channel set and return (loss, meta). No retraining, no artifacts written."""
    loss_cfg = _global_loss_cfg(gcfg)
    overrides = _state_overrides(gcfg)
    state = _build_state_for_analysis(Path(run_dir), device_override=device,
                                      path_remap=path_remap, cfg_overrides=overrides)
    try:
        nch = int(getattr(state, "dataset_num_channels", -1))
        exp = len(loss_cfg["asinh_scale_by_channel"])
        if nch != exp:
            raise ValueError(f"global-eval channel mismatch for {Path(run_dir).name}: "
                             f"dataset loaded {nch} channels but image_loss has {exp} asinh scales")
        loss, meta = compute_validation_loss(
            state,
            batch_rays=int(gcfg.get("batch_rays", 4096)),
            max_batches=gcfg.get("max_batches"),
            num_workers=int(gcfg.get("num_workers", 0)),
            loss_cfg=loss_cfg,
        )
    finally:
        _release_state(state)
    return float(loss), dict(meta)


def _merge_into_summary(run_dir: Path, name: str, loss: float, meta: dict) -> None:
    """Write a durable global_val.json AND non-clobberingly merge final/val_<name> into summary.json."""
    key = f"final/val_{name}"
    nrays = int(meta.get("num_val_rays", 0))

    # (a) durable per-run record (survives a summary rebuild via the summary.py extractor in Part 4)
    gv_path = run_dir / "artifacts" / "train" / "global_val.json"
    gv_path.parent.mkdir(parents=True, exist_ok=True)
    gv_path.write_text(json.dumps(
        {"name": name, key: float(loss), f"{key}_nrays": nrays, "loss_name": meta.get("loss_name")}, indent=2))

    # (b) immediate, non-clobbering merge into the run's summary.json (only ADDS keys)
    spath = run_dir / "summary.json"
    if spath.exists():
        summ = json.loads(spath.read_text())
        summ[key] = float(loss)
        summ[f"{key}_nrays"] = nrays
        spath.write_text(json.dumps(summ, indent=2))
    else:
        logger.warning("no summary.json at %s; wrote global_val.json only", run_dir.name)


def run_global_validation(spec_path, device_override=None) -> dict:
    spec_path = Path(spec_path).resolve()
    gcfg = load_yaml(spec_path).get("global_validation", {})
    name = str(gcfg.get("name", "global"))
    benchmark_dir = Path(gcfg["benchmark_dir"])
    if not benchmark_dir.is_absolute():
        benchmark_dir = (spec_path.parent / benchmark_dir).resolve()
    device = device_override if device_override is not None else gcfg.get("device")
    path_remap = gcfg.get("path_remap")

    runs_dir = benchmark_dir / "runs"
    results = {"benchmark_dir": str(benchmark_dir), "name": name, "key": f"final/val_{name}", "runs": []}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "summary.json").exists():
            continue
        try:
            loss, meta = compute_global_val_for_run(run_dir, gcfg, device=device, path_remap=path_remap)
            _merge_into_summary(run_dir, name, loss, meta)
            results["runs"].append({"run": run_dir.name, "status": "ok",
                                    f"final/val_{name}": loss, "nrays": int(meta.get("num_val_rays", 0))})
            logger.info("global-val[%s] %s -> final/val_%s = %.6e", name, run_dir.name, name, loss)
        except Exception as e:
            logger.warning("global-val[%s] FAILED for %s: %s", name, run_dir.name, e)
            results["runs"].append({"run": run_dir.name, "status": "failed", "error": str(e)})

    # non-clobbering: re-reads EVERY summary.json (now incl. final/val_<name>) and rewrites aggregate*.csv
    aggregate_benchmark_dir(benchmark_dir)
    (benchmark_dir / f"global_val_{name}_manifest.json").write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results["runs"] if r["status"] == "ok")
    logger.info("global-val '%s' done: %d/%d runs -> re-aggregated %s",
                name, n_ok, len(results["runs"]), benchmark_dir)
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Cross-condition global (all-channel) held-out validation loss; writes final/val_<name>.")
    ap.add_argument("--spec", required=True, help="YAML with a `global_validation:` block")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_global_validation(args.spec, device_override=args.device)


if __name__ == "__main__":
    main()