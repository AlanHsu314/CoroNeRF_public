# Reproducing CoroNeRF

Pipeline: **build LUT** (`data/README.md`) → **build dataset** (`datasets/README.md`) →
**train benchmarks** → **run ensemble/error-localization diagnostics** → **render figures/tables**.
Set `dataset_dir` / `LUT_dir` in the specs to your regenerated paths first.

## 1. Train the paper benchmarks
Each writes `runs_benchmarks/<name>/` with per-run `summary.json` + an `aggregate*.csv`.
```bash
python scripts/run_benchmark.py --spec 2026_08_20__paper_A2_density_representation_fe_pair.yaml   # Table 12 / Fig 2
python scripts/run_benchmark.py --spec 2026_08_16__paper_B2_joint_target_channel_ablation.yaml   # Table 15
python -m coronerf.tools.global_val --spec configs/benchmarks/2026_08_31__global_val_paper_B2.yaml --device cuda # generate global common-probe metric
python scripts/run_benchmark.py --spec 2026_08_31__paper_C2_noise_view_stress_test.yaml          # Fig 5
python scripts/run_benchmark.py --spec 2026_09_14__paper_D2_abundance_mismatch.yaml              # Table 17
python scripts/run_benchmark.py --spec 2026_09_17__paper_E2_loss_ablation.yaml                   # Table 13
python scripts/run_benchmark.py --spec 2026_06_19__paper_F_epistemic_uncertainty.yaml            # σ_ens ensemble (Tables 2–10)
```

## 2. Ensemble/error-localization diagnostics (needs the paper-F ensemble runs)
```bash
python -m coronerf.tools.disagreement --kind batch --spec seed_disagreement_batch_paper_F.yaml --device cuda # specifically for ensemble panels (Figs 6 and 9)

python -m coronerf.tools.disagreement --kind baselines --spec baselines_paper_F.yaml --device cuda   # correlations, sparsification/nAUSE, E_cap, LOO
```
Writes `runs_identifiability/`. Toggles inside the spec select LOO, bootstrap resamples (2000),
and the exact-random reference.

## 3. Figures & tables

Invocation: `python -m coronerf.tools.paper_figures --kind <KIND> --spec <spec>.yaml [--device cuda]`
(`--kind` is one of `_KIND` in `coronerf/tools/paper_figures.py`; `--index` regenerates `REGISTRY.md`.)

| Fig | Sec | `--kind` | `--spec` (in `configs/paper_figures/`) |
|----:|:---|:---|:---|
| 1 | main | — | schematic (not one spec): channel strip = `channel_strip` + `setup_channel_strip.yaml`; corona/field via `coronerf.tools.diagram` + `configs/diagram/*` |
| 2 | main | `shell_panels` | `paper_A2_density_shell_r1p5.yaml` |
| 3 | main | `los_filmstrip` | `paper_B2_los_filmstrip.yaml` |
| 4 | main | `image_field_grid` | `paper_B2_image_field_grid.yaml` |
| 5 | main | `heatmap` | `paper_C2_noise_view_heatmaps_x1.yaml` |
| 6 | main | `ensemble_panels` | `uq_ensemble_panels.yaml` |
| 7 | main | `uq_baseline_summary` | `uq_baseline_summary.yaml` |
| 8 | main | `uq_errorcap_ladder` | `uq_errorcap_ladder.yaml` |
| 9 | app | `ensemble_panels` | `uq_ensemble_panels_appendix.yaml` |
| 10 | app | `uq_sparsification` | `uq_sparsification.yaml` |
| 11 | app | `slice_panels` | `paper_A2_density_slice_r1p5.yaml` |
| 12 | app | `robustness_shell` | `paper_E2_loss_transition_shell_residuals_r1p5.yaml`|
| 13 | app | `multi_radius` | `paper_B2_all4_multi_radius_shells.yaml` |
| 14 | app | `radial_curves` | `paper_B2_spectral_radial_curves.yaml` |
| 15 | app | `image_field_grid` | `paper_B2_image_field_grid_appendix.yaml`|
| 16 | app | `robustness_shell` | `paper_C2_field_robustness_shell_r1p5.yaml` |
| 17 | app | `denoising` | `paper_C2_noisy_obs_denoising.yaml` |
| 18 | app | `robustness_shell` | `paper_D2_abundance_shell_r1p5.yaml` |
| 19 | app | `los_filmstrip` | `paper_F_los_filmstrip_appendix.yaml` |
| 20 | app | `los_filmstrip` | `paper_F_los_filmstrip_appendix_temp.yaml` |
| 21 | app | `los_filmstrip` | `paper_F_los_filmstrip_emissivity.yaml` |

## Compute & determinism
One reconstruction ≈ 3 GPU-h (80 GB A100); a K=10 ensemble ≈ 30 GPU-h; the full cross-condition
study ≈ 330 GPU-h, ten conditions in Tables 2-10 ≈ 300 GPU-h. Hash-grid training is not bitwise-reproducible, so expect small run-to-run
differences; we report mean ± std across seeds.