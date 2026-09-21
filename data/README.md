# data/ — not included; regenerate

The training forward model reads one LUT directory: `data/LUT/v2/` containing
`ccoef_LUT.npy`, `grid_vals.npz`, `psi_fields.npz`. We build it below.

## 1. PSI MAS cube → data/corona/CR2283/
Download the CR2283 thermodynamic MHD run from Predictive Science (PSI/MAS):
https://www.predsci.com/data/runs/cr2283-high/hmi_mast_mas_std_0201/  (model `hmi_mast_mas_std_0201`).
Place the cube under `data/corona/CR2283/` (n_e, T, B, and lon/colat/r axes, as loaded by `psi.Model`). Download the `corona.zip` and unzip it in `data/corona/CR2283/`. The files in this directly should be `.hdf` files. Finally, make sure the `omas` file indicates "Thermodynamic MAS Solution," not the polytropic one.

## 2. Cube + CHIANTI → data/LUT/v2/
Requires CHIANTI 10.1 (linux: set `export XUVTOP=/path/to/CHIANTI_10.1`), and pyCELP.
```bash
python src/compute_ccoef_LUT_V2.py     # reads data/corona/CR2283/, writes data/LUT/v2/
```
This tabulates the emission collisional coefficient C(log n_e, log T, log r) per line for the
Fe XIII 1074.7/1079.8 nm and Si IX 2584.6/3929.3 nm channels, and resamples the PSI fields onto
`psi_fields.npz`. Confirm the output path and grid extents near the bottom of the script.