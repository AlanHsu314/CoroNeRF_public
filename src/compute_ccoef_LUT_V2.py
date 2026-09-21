from __future__ import annotations

###########################################################
### compute (and store) multi-line ccoef LUT for training
###########################################################

import copy
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import tqdm

# Keep CHIANTI happy on Windows without clobbering HOME on Unix-like systems.
if "HOME" not in os.environ and "USERPROFILE" in os.environ:
    os.environ["HOME"] = os.environ["USERPROFILE"]

import psi
import pycelp


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SCRIPT_DIR / "../data/LUT/v2").resolve()
PSI_MODEL_DIR = (SCRIPT_DIR / "../data/corona/CR2283").resolve()

# Channel order is the contract with coroNeRF.
# Keep renderer.wavelengths / ion_kwargs.wavelengths / line_names in the same order later.
ION_SPECS: List[Dict] = [
    {
        "ion_name": "fe_13",
        "n_levels": 50,
        "wavelengths": [10747.0, 10798.0],
        "line_names": ["Fe XIII 10747", "Fe XIII 10798"],
    },
    {
        "ion_name": "si_9",
        "n_levels": 50,
        "wavelengths": [25846.4727, 39292.7305],
        "line_names": ["Si IX 25846", "Si IX 39293"],
    },
]

DEFAULT_BOUNDS = {
    "ne": {
        "quantile_lo": 0.01,
        "quantile_hi": 0.99,
        "margin_lo": 0.3,   # dex
        "margin_hi": 0.3,   # dex
        "num_grid": 128,
    },
    "temp": {
        "quantile_lo": 0.01,
        "quantile_hi": 0.99,
        "margin_lo": 0.05,  # dex
        "margin_hi": 0.05,  # dex
        "num_grid": 64,
    },
    "r": {
        "global_bound_hi": np.log10(1 + 9),
        "global_bound_lo": np.log10(1 + 1e-3),
        "num_grid": 32,
    },
}


def get_temp_bounds_from_ion(ion, eta: float = 1e-3, pad_dex: float = 0.2):
    """
    Given an ion object, get reasonable log10(T) bounds from its ion fraction curve.
    """
    logT = np.asarray(ion.ioneq_logtemp, dtype=np.float64)
    frac = np.asarray(ion.ioneq_frac, dtype=np.float64)

    fmax = np.nanmax(frac)
    mask = frac >= eta * fmax
    if not np.any(mask):
        raise ValueError("Ion fraction too small or missing data.")

    logTmin = float(logT[mask].min() - pad_dex)
    logTmax = float(logT[mask].max() + pad_dex)
    return logTmin, logTmax


def compute_ccoefs_for_lines(log_ne: float, log_temp: float, log_r: float, ion, lines):
    """
    Compute C_coeff for multiple lines of the same ion with one calc_rho_sym call.
    """
    thetab = np.degrees(np.arccos(1.0 / np.sqrt(3.0)))

    ne = 10.0 ** log_ne
    temp = 10.0 ** log_temp
    r = 10.0 ** log_r
    height = r - 1.0

    ion.calc_rho_sym(ne, temp, height, thetab)

    coeffs = np.empty(len(lines), dtype=np.float64)
    for i, line in enumerate(lines):
        rho00_u = ion.rho[line.upper_level_index, 0]
        upper_level_pop_frac = np.sqrt(2.0 * line.Jupp + 1.0) * rho00_u
        coeffs[i] = (line.hnu / (4.0 * np.pi)) * line.Einstein_A * upper_level_pop_frac * ion.totn

    return coeffs


def save_psi_data(output_dir: Path, corona) -> None:
    """
    Save PSI fields using the same basic convention as the existing LUT bundle.
    """
    np.savez_compressed(
        output_dir / "psi_fields.npz",
        ne=corona.ne,
        T=corona.temp,
        lons=corona.lons,
        # Keep the key name `colats` to match the current coroNeRF PSI loader.
        colats=corona.lats,
        rs=corona.rs,
    )


def compute_psi_bounds(corona, bounds_dict: Dict) -> Dict:
    """
    Compute ne/temp bounds from the PSI cube statistics.
    """
    psi_vals = {"ne": corona.ne, "temp": corona.temp}

    for key, cube_vals in psi_vals.items():
        cube = np.log10(np.asarray(cube_vals, dtype=np.float64))

        # compute quantile
        quantile_lo = float(np.quantile(cube, bounds_dict[key]["quantile_lo"]))
        quantile_hi = float(np.quantile(cube, bounds_dict[key]["quantile_hi"]))

        # pad with margin (dex)
        margin_lo = quantile_lo - bounds_dict[key]["margin_lo"]
        margin_hi = quantile_hi + bounds_dict[key]["margin_hi"]

        # clip to global max and min
        gmin = float(cube.min())
        gmax = float(cube.max())

        bound_lo = float(np.clip(margin_lo, a_min=gmin, a_max=None))
        bound_hi = float(np.clip(margin_hi, a_min=None, a_max=gmax))

        # store
        bounds_dict[key]["psi_bound_lo"] = bound_lo
        bounds_dict[key]["psi_bound_hi"] = bound_hi

        print(f"PSI {key} bounds: {bound_lo:.6f} .. {bound_hi:.6f}")

    return bounds_dict


def build_ion_models(ion_specs: List[Dict]):
    """
    Instantiate pycelp ions, get emission-line handles, and collect atomic temp bounds.
    """
    ion_models = []
    for spec in ion_specs:
        ion = pycelp.Ion(spec["ion_name"], nlevels=spec["n_levels"])
        lines = [ion.get_emissionLine(wvl) for wvl in spec["wavelengths"]]
        temp_lo, temp_hi = get_temp_bounds_from_ion(ion, eta=1e-3, pad_dex=0.2)

        ion_models.append(
            {
                "spec": spec,
                "ion": ion,
                "lines": lines,
                "temp_lo": temp_lo,
                "temp_hi": temp_hi,
            }
        )

        print(
            f"Atomic temp bounds for {spec['ion_name']}: "
            f"{temp_lo:.6f} .. {temp_hi:.6f}"
        )

    return ion_models


def combine_bounds(bounds_dict: Dict, ion_models) -> Dict:
    """
    Build one shared grid over (log ne, log T, log r) for all channels.

    For temperature, use the union of per-ion atomic ranges and then clip with PSI bounds.
    This keeps one common grid_vals.npz for all channels.
    """
    atomic_temp_lo = min(m["temp_lo"] for m in ion_models)
    atomic_temp_hi = max(m["temp_hi"] for m in ion_models)

    # chianti bounds, reasonabale bounds known apriori
    bounds_dict["ne"]["chianti_bound_lo"] = 6.0
    bounds_dict["ne"]["chianti_bound_hi"] = 11.0
    bounds_dict["temp"]["chianti_bound_lo"] = atomic_temp_lo
    bounds_dict["temp"]["chianti_bound_hi"] = atomic_temp_hi

    bounds_dict["ne"]["global_bound_lo"] = bounds_dict["ne"]["psi_bound_lo"]
    bounds_dict["ne"]["global_bound_hi"] = bounds_dict["ne"]["psi_bound_hi"]

    # combine bounds
    bounds_dict["temp"]["global_bound_lo"] = max(
        bounds_dict["temp"]["psi_bound_lo"],
        bounds_dict["temp"]["chianti_bound_lo"],
    )
    bounds_dict["temp"]["global_bound_hi"] = min(
        bounds_dict["temp"]["psi_bound_hi"],
        bounds_dict["temp"]["chianti_bound_hi"],
    )

    if bounds_dict["temp"]["global_bound_lo"] >= bounds_dict["temp"]["global_bound_hi"]:
        raise ValueError(
            "Temperature bounds are invalid after combining PSI and atomic constraints: "
            f"{bounds_dict['temp']['global_bound_lo']} >= {bounds_dict['temp']['global_bound_hi']}"
        )

    print(
        "Global ne bounds: "
        f"{bounds_dict['ne']['global_bound_lo']:.6f} .. {bounds_dict['ne']['global_bound_hi']:.6f}"
    )
    print(
        "Global temp bounds: "
        f"{bounds_dict['temp']['global_bound_lo']:.6f} .. {bounds_dict['temp']['global_bound_hi']:.6f}"
    )
    print(
        "Global r bounds: "
        f"{bounds_dict['r']['global_bound_lo']:.6f} .. {bounds_dict['r']['global_bound_hi']:.6f}"
    )

    return bounds_dict


def build_grid(bounds_dict: Dict):
    """
    Build the shared regular grid and flattened point list.
    """

    # mesh grid
    ne_arr = np.linspace(
        bounds_dict["ne"]["global_bound_lo"],
        bounds_dict["ne"]["global_bound_hi"],
        bounds_dict["ne"]["num_grid"],
        dtype=np.float64,
    )
    temp_arr = np.linspace(
        bounds_dict["temp"]["global_bound_lo"],
        bounds_dict["temp"]["global_bound_hi"],
        bounds_dict["temp"]["num_grid"],
        dtype=np.float64,
    )
    r_arr = np.linspace(
        bounds_dict["r"]["global_bound_lo"],
        bounds_dict["r"]["global_bound_hi"],
        bounds_dict["r"]["num_grid"],
        dtype=np.float64,
    )

    ne_mg, temp_mg, r_mg = np.meshgrid(ne_arr, temp_arr, r_arr, indexing="ij")
    pts = np.stack([ne_mg.ravel(), temp_mg.ravel(), r_mg.ravel()], axis=-1)

    return ne_arr, temp_arr, r_arr, ne_mg.shape, pts


def compute_ccoef_grid(pts: np.ndarray, grid_shape, ion_models):
    """
    Compute one shared ccoef grid with channel axis first.
    """
    num_channels = sum(len(m["lines"]) for m in ion_models)
    ccoef_arr = np.zeros((num_channels, pts.shape[0]), dtype=np.float64)

    # counter for channels
    channel_offset = 0

    # loop per ion
    for model in ion_models:
        spec = model["spec"]
        ion = model["ion"]
        lines = model["lines"]
        n_lines = len(lines)

        desc = f"{spec['ion_name']} ({n_lines} lines)"
        for i_pt, (log_ne, log_temp, log_r) in enumerate(
            tqdm.tqdm(pts, total=pts.shape[0], desc=desc)
        ):
            ccoef_arr[channel_offset:channel_offset + n_lines, i_pt] = compute_ccoefs_for_lines(
                log_ne=log_ne,
                log_temp=log_temp,
                log_r=log_r,
                ion=ion,
                lines=lines,
            )

        channel_offset += n_lines

    ccoef_grid = ccoef_arr.reshape((num_channels, *grid_shape))
    return ccoef_grid


def print_channel_order(ion_models) -> None:
    print("\nChannel order written to ccoef_LUT.npy:")
    i_channel = 0
    for model in ion_models:
        spec = model["spec"]
        for line_name, wavelength in zip(spec["line_names"], spec["wavelengths"]):
            print(f"  ch{i_channel}: {line_name} ({spec['ion_name']}, {wavelength} A)")
            i_channel += 1
    print()


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PSI model from: {PSI_MODEL_DIR}")
    corona = psi.Model(str(PSI_MODEL_DIR) + '/') # for some reason pyhdf wanted the '/'

    print(f"Saving PSI fields to: {output_dir / 'psi_fields.npz'}")
    save_psi_data(output_dir, corona)

    print("Building ion models...")
    ion_models = build_ion_models(ION_SPECS)
    print_channel_order(ion_models)

    bounds_dict = copy.deepcopy(DEFAULT_BOUNDS)
    bounds_dict = compute_psi_bounds(corona, bounds_dict)
    bounds_dict = combine_bounds(bounds_dict, ion_models)

    print("Building shared LUT grid...")
    ne_arr, temp_arr, r_arr, grid_shape, pts = build_grid(bounds_dict)
    print(f"Grid shape per channel: {grid_shape}")
    print(f"Total grid points: {pts.shape[0]}")

    print(f"Saving grid values to: {output_dir / 'grid_vals.npz'}")
    np.savez_compressed(
        output_dir / "grid_vals.npz",
        ne_arr=ne_arr,
        temp_arr=temp_arr,
        r_arr=r_arr,
    )

    print("Computing ccoef LUT...")
    ccoef_grid = compute_ccoef_grid(pts, grid_shape, ion_models)
    print(f"ccoef grid shape: {ccoef_grid.shape}")

    print(f"Saving ccoef LUT to: {output_dir / 'ccoef_LUT.npy'}")
    np.save(output_dir / "ccoef_LUT.npy", ccoef_grid)

    print("Done.")


if __name__ == "__main__":
    main()
