# CoroNeRF

Differentiable, multi-line neural tomography that recovers 3-D coronal electron density (n_e) and
temperature (T_e) from multiview, multiline spectral-intensity images of the solar corona, hence the name CoroNeRF.

## Contents
- `coronerf/` — the package (forward model, fields, renderer, training, benchmark + identifiability tools)
- `configs/` — every experiment/benchmark/paper-level figure is a YAML (base config + benchmark specs + figure specs)
- `scripts/` — thin CLI wrappers around `coronerf.pipeline`
- `src/` — one-time data builders (CHIANTI LUT, PSI cube loading)
- `REPRODUCE.md` — how to regenerate datasets, runs, tables, and figures

Large artifacts (`data/`, `datasets/`, `runs*/`) are **not** included; `REPRODUCE.md` and
`data`/`datasets` READMEs explain how to regenerate them.

## Install
```bash
#---------- (0) Download CHIANTI 10.1 and add to path
# if linux, export XUVTOP=<CHIANTI_PATH>
# if windows, handled in python API already
# not tested for mac

#---------- (1) create env
conda create -n coronerf python=3.11
conda activate coronerf

#---------- (2) install Pycelp for our CHIANTI API
# specify numpy and numba versions, latest Pycelp version uses this
pip install -c constraints.txt "numpy==1.26.0" "numba==0.60.0"
# cd into modified pycelp repo, (in another dir)
# if linux, you may also choose to git clone from original repo, but windows must use the given modified version
cd third_party
cd pycelp
pip install -e . --config-settings editable_mode=compat

#----------- (3) GPU PyTorch (CUDA 11.8 build used in the paper, other CUDA builds may work):
cd ../..
pip install -c constraints.txt torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

#----------- (4) install other dependencies
pip install -c constraints.txt matplotlib scipy tqdm nerfacc PyYAML ninja wandb pandas pyhdf opencv-python==4.11.0.86
# be careful installing requirements, might override the specific versions above pip install -r requirements.txt

#---------- (5) register the coronerf package
pip install -c constraints.txt -e .
```

## Quickstart
```bash
# download PSI cube: see data/README.md

# generate dataset: see datasets/README.md

# sanity smoke first
python scripts/run_benchmark.py --spec paper_canon_smoke.yaml

# joint density-temperature benchmark
python scripts/run_benchmark.py --spec 2026_08_16__paper_B2_joint_target_channel_ablation.yaml
```

## Reproducing the paper
See `REPRODUCE.md`. Benchmark specs live in `configs/benchmarks/`,
ensemble/error-localization tools in `configs/identifiability/`, figures in `configs/paper_figures/`.

## Notes
- GPU hash-grid training is **not** bitwise-deterministic (atomic scatter in the hash backward and
  LOS accumulation); fixed seeds give statistically equivalent, not identical, runs.
- Config values in `config_coronerf.yaml` are smoke defaults (e.g. `max_steps: 15`); the paper
  settings live in the benchmark specs, which override them.

## Third-party code
`third_party/pycelp/` is a modified copy of **pyCELP** (Thomas A. Schad, National
Solar Observatory), redistributed under the BSD-3-Clause license. See
`third_party/pycelp/LICENSE.txt` and `third_party/pycelp/NOTICE` for the license
and a summary of local modifications. All other code in this repository is under
the MIT License (see `LICENSE`).

## Citing
If you use this code, please cite the CoroNeRF paper (see `CITATION.cff`) and the
third-party components it builds on:
- **pyCELP** - Schad & Dima (2020), *Solar Physics*, 295, 98 (ADS `2020SoPh..295...98S`); software: ASCL `ascl:2112.001`
- **CHIANTI** atomic database - Dere et al. (1997) plus the version paper for the CHIANTI release you use
- **PSI/MAS** coronal MHD model (Carrington Rotation 2283) - Predictive Science Inc. (predsci.com), per their data-usage terms
