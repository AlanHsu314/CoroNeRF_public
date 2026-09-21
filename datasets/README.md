# datasets/ — not included; regenerate with vpgen

A training dataset is a set of rendered multi-line viewpoints plus copies of the LUT and a
photosphere mask. Generate it from the corona cube + LUT (build those first — see `data/README.md`).

```bash
python scripts/run_benchmark.py --spec <vpgen_config>.yaml
```

We recommend testing first with `vpgen_multiline_smoke.yaml`, which generates 256x256 images, 30 views, 4 channels, but you may further modify the dataset config block as follows:

The `vpgen:` block in the config controls image size (`spatial_H`/`spatial_W`), field of view
(`fov_rsun`), viewpoint count (`n_lon`), and observer distance (`obs_r`). Output lands in
`datasets/<name>/` as `viewpoints_gt/vp_*.npz`, copied `ccoef_LUT.npy` / `grid_vals.npz` /
`psi_fields.npz`, `mask.npy`, and `stats_dict.pkl`.

We further note that all benchmark specs will save some stats and manifests in `runs_benchmarks/`, but the dataset will be saved under `datasets/`.

**Paper dataset:** If you want to generate the dataset used in the paper, use `--spec vpgen_multiline.yaml`, which uses the yaml in `configs/benchmarks/vpgen_multiline.yaml`. 256×256 images, 3000 evenly-spaced longitudes, 4 channels, seed 12648430
(`datasets/master_datasets_multiline/res256__seed-12648430__eb4c3af3`). The trailing hash is
content-derived; after regenerating, point each benchmark spec's `dataset_dir` at your new folder.

Note: `scripts/run_vpgen.py` is a deprecated way of generating the dataset: it probably still works, but I would recommend using `run_benchmark.py` instead.