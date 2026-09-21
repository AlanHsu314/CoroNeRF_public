from __future__ import annotations
import argparse, inspect, datetime
from pathlib import Path
from ._paper_cli import resolve_paper_spec_path, setup_paper_logging, figure_out_dir, REPO_ROOT
from ..benchmark.config import load_yaml

# specific paper plots
from ..paper.metric_heatmaps import generate_metric_heatmaps
from ..paper.metric_curves import generate_metric_curves
from ..paper.radial_metric_curves import generate_radial_metric_curves
from ..paper.metric_bars import generate_metric_bar_figure
from ..paper.image_field_scatter import generate_image_field_scatter, generate_image_field_grid
from ..paper.composite_recon_scatter import generate_composite_recon_scatter
from ..paper.joint_shell_panels import generate_joint_shell_figure
from ..paper.multi_radius_shell_grid import generate_multi_radius_shell_grid
from ..paper.observation_denoising import generate_observation_denoising_figure
from ..paper.robustness_shell_grid import generate_robustness_shell_grid
from ..paper.shell_panels import generate_shell_panel_figure
from ..paper.empirical_analytic import generate_empirical_vs_analytic
from ..paper.uncertainty_quantification import fig_ladder, fig_identifiability, fig_operating_heatmaps, fig_ensemble_panels, fig_oppoint, fig_calibration, fig_fnull_curves, fig_ensemble_slice_composite
from ..paper.los_filmstrip import generate_los_filmstrip
from ..paper.channel_strip import fig_channel_strip
from ..paper.uq_baselines import fig_baseline_ladder, fig_sparsification, fig_baseline_summary, fig_errorcap_ladder

# identifiability diagnostics 
from ..identifiability.disagreement_diagnostics import (
    generate_radial_image_figure,
)

_KIND = {
    "heatmap":            generate_metric_heatmaps,                    # generic 2D heatmaps of metric over 2 condition axes (paperC)
    "curves":             generate_metric_curves,                      # generic metric v condition curves (paperC, D, E)
    "radial_curves":      generate_radial_metric_curves,               # generic metric v radius plots (paperB)
    "bars":               generate_metric_bar_figure,                  # generic bar plot (paperB)
    "image_field":        generate_image_field_scatter,                # generic imaage-space vs field-space scatter (papers BCDE)
    "image_field_grid":   generate_image_field_grid,                   # image-field scatter grid, rows are field metrics, cols are image metrics
    "composite_scatter":  generate_composite_recon_scatter,            # shell + scatter, specific for proposals that need concise plots (paperB)
    "joint_shell":        generate_joint_shell_figure,                 # generic joint shell panels (paperB)
    "multi_radius":       generate_multi_radius_shell_grid,            # generic multi radius shell panels (paperB)
    "denoising":          generate_observation_denoising_figure,       # generic image-space denoise panels (paperC)
    "robustness_shell":   generate_robustness_shell_grid,              # generic multi condition shell panels (paperC, D, E) 
    "shell_panels":       generate_shell_panel_figure,                 # generic shell panels (paperA)
    "slice_panels":       generate_shell_panel_figure,                 # same builder, slice mode via yaml `slice:`
    "empirical_analytic": generate_empirical_vs_analytic,              # specific empirical v analytic heatmaps (paperC + uq)
    "uq_ladder":          fig_ladder,                                  # uq ladder of rhos
    "uq_identifiability": fig_identifiability,                         # uq spectrum, CRB, posterior, prior dominance
    "uq_opheat":          fig_operating_heatmaps,                      # uq eff rank, PR, and post med heatmaps
    "uq_oppoint":         fig_oppoint,                                 # uq operating point analysis (m^* vs cross-seed \hat{m})
    "uq_calibration":     fig_calibration,                             # calibration curve, err vs sigma_ens scatter (paperF + uq)
    "uq_baseline_ladder": fig_baseline_ladder,                         # baseline correlation multicondition ladders (uq)
    "uq_sparsification":  fig_sparsification,                          # sparsification + nAUSE multicondition plots (uq)
    "uq_baseline_summary":fig_baseline_summary,                        # correlation + sparsification multipanel (uq)
    "uq_errorcap_ladder": fig_errorcap_ladder,                         # error capture summary (uq)
    "uq_fnull":           fig_fnull_curves,                            # nullspace fraction multicondition curves (uq)
    "radial_composite":   generate_radial_image_figure,                # specific radial sweep + image composite (paperF)
    "los_filmstrip":      generate_los_filmstrip,                      # rendered ne and Te filmstrip
    "ensemble_panels":    fig_ensemble_panels,                         # ensemble mean/err/disagreement (+ analytic post/crb/pdom, paperF) 
    "uq_ensemble_slice":  fig_ensemble_slice_composite,                # composite ensemble shell and slice for the paper (paperF)
    "channel_strip":      fig_channel_strip,                           # channel strip, displays dataset, requires bench (paperB)                    
}

def _call(builder, spec_path, out_dir, device):
    params = inspect.signature(builder).parameters
    kw = {}
    if "output_dir_override" in params and out_dir is not None:
        kw["output_dir_override"] = str(out_dir)
    if "device_override" in params:
        kw["device_override"] = device
    if "spec_path" in params:
        return builder(spec_path=spec_path, **kw)
    return builder(spec_path, **kw)

def _index():
    root = REPO_ROOT / "paper_outputs" / "figures"
    rows = ["# Paper figure registry", "",
            "| group | slug | folder | files | updated |", "|---|---|---|---|---|"]
    for man in sorted(root.rglob("*_manifest.json")):
        rel = man.parent.relative_to(root).parts
        group = rel[0] if rel else "?"
        slug = rel[1] if len(rel) > 1 else man.stem.replace("_manifest", "")
        files = ", ".join(sorted(p.name for p in man.parent.glob("*") if p.suffix in (".png", ".pdf", ".csv")))
        upd = datetime.date.fromtimestamp(man.stat().st_mtime).isoformat()
        rows.append(f"| {group} | {slug} | {man.parent.relative_to(REPO_ROOT)} | {files} | {upd} |")
    (root / "REGISTRY.md").write_text("\n".join(rows) + "\n")
    print(f"wrote {root/'REGISTRY.md'}  ({len(rows)-4} figures)")

def main():
    ap = argparse.ArgumentParser(description="Unified paper-figure builder + index.")
    ap.add_argument("--kind", choices=list(_KIND))
    ap.add_argument("--spec")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--index", action="store_true", help="regenerate REGISTRY.md and exit")
    ap.add_argument("--verbosity", type=int, default=1)
    a = ap.parse_args()
    setup_paper_logging(a.verbosity)
    if a.index:
        _index(); return
    if not (a.kind and a.spec):
        ap.error("--kind and --spec are required (unless --index)")
    spec = resolve_paper_spec_path(a.spec)
    out_dir = figure_out_dir(load_yaml(spec), a.output_dir)
    _call(_KIND[a.kind], spec, out_dir, a.device)

if __name__ == "__main__":
    main()