from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ..benchmark.config import load_yaml
from ..util.figure import _resolve_path, _savefig


def _stats_limits(dataset_dir, channel_indices):
    """Per-channel (log10) color bounds + line names from the dataset stats_dict (global channel keys)."""
    with open(Path(dataset_dir) / "stats_dict.pkl", "rb") as f:
        st = pickle.load(f)
    lims, names = [], []
    for g in channel_indices:
        s = st.get(int(g), {}) or {}
        names.append(str(s.get("name", f"ch {g}")))
        lo, hi = s.get("-3_sigma"), s.get("+3_sigma")
        lims.append((float(np.log10(lo)) if lo and lo > 0 else None,
                     float(np.log10(hi)) if hi and hi > 0 else None))
    return lims, names


def fig_channel_strip(spec_path, device_override=None, output_dir_override=None):
    spec_path = Path(spec_path).resolve(); root = spec_path.parent
    cfg = load_yaml(spec_path).get("channel_strip", {})
    out = Path(output_dir_override) if output_dir_override else _resolve_path(
        cfg.get("output_dir", "../../paper_outputs/figures/setup"), root)
    out.mkdir(parents=True, exist_ok=True)
    plot = dict(cfg.get("plot", {}) or {})

    # --- resolve dataset_dir + channel_indices from the benchmark spec (experiment override wins) ---
    bspec_path = _resolve_path(cfg["benchmark_spec"], root)
    bspec = load_yaml(bspec_path).get("benchmark", {})
    fixed = bspec.get("fixed", {}) or {}
    ov = {}
    exp_name = cfg.get("experiment")
    if exp_name:
        ex = next((e for e in bspec.get("experiments", []) if e.get("name") == exp_name), None)
        if ex is None:
            raise ValueError(f"experiment {exp_name!r} not found in {bspec_path.name}")
        ov = ex.get("overrides", {}) or {}
    def gk(key, default=None):                     # flat dotted-key lookup; benchmark configs are flat
        return ov.get(key, fixed.get(key, default))

    dataset_dir = _resolve_path(gk("dataset_dir"), bspec_path.parent)   # dataset paths are relative to the benchmark cfg
    channels = list(gk("load_data_kwargs.channel_indices", [0, 1, 2, 3]))

    # --- load one view + mask ---
    view = int(cfg.get("view_index", 0))
    z = np.load(Path(dataset_dir) / "viewpoints_gt" / f"vp_{view}.npz")
    I = np.asarray(z["intensities"]).astype(float)                     # (H, W, C_all)
    mask = np.load(Path(dataset_dir) / "mask.npy").astype(bool)        # (H, W)
    lims, names = _stats_limits(dataset_dir, channels)

    # --- grid of channels (plot.ncols configurable; default single-row strip) ---
    C = len(channels)
    ncols = int(plot.get("ncols", cfg.get("ncols", C)))               # e.g. 2 -> 2x2 for four channels
    nrows = int(np.ceil(C / ncols))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=tuple(plot.get("figsize", [2.0 * ncols, 2.7 * nrows])), squeeze=False)
    styles = cfg.get("channel_styles", {}) or {}
    def _style(g):                                   # yaml keys may be int (0:) or str ("0":)
        return styles.get(g, styles.get(str(g), {})) or {}
    for i, g in enumerate(channels):
        st = _style(int(g))
        disp = np.log10(np.clip(I[:, :, int(g)], 1e-12, None))          # clip on raw (>=0) avoids log warnings
        disp[~mask] = np.nan                                           # off-disk -> white
        vmin = st.get("vmin", lims[i][0]); vmax = st.get("vmax", lims[i][1])   # in log10 intensity (colorbar units)
        cmap = plt.get_cmap(st.get("cmap", plot.get("cmap", "inferno"))).copy()
        cmap.set_bad(plot.get("bad_color", "white"))
        r, c = divmod(i, ncols)
        ax = axs[r][c]
        im = ax.imshow(disp, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                aspect=plot.get("aspect", "equal"),
                interpolation=plot.get("interpolation", "nearest"))
        ax.set_title(st.get("label", names[i]), fontsize=plot.get("title_fontsize", 9))
        ax.set_xticks([]); ax.set_yticks([])
        if plot.get("colorbar", True):
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
            cb.ax.tick_params(labelsize=plot.get("cbar_fontsize", 7))
    for j in range(C, nrows * ncols):                                 # hide any unused cells
        rr, cc = divmod(j, ncols); axs[rr][cc].set_axis_off()
    if plot.get("suptitle"):
        fig.suptitle(plot["suptitle"], fontsize=plot.get("suptitle_fontsize", 10))
    if (plot.get("wspace") is not None) or (plot.get("hspace") is not None):
        fig.subplots_adjust(wspace=float(plot.get("wspace", 0.05)), hspace=float(plot.get("hspace", 0.05)))
    else:
        fig.tight_layout()                                            # unchanged default
    return _savefig(fig, out / cfg.get("name", "fig_channel_strip"),
                    cfg.get("formats", ["pdf", "png"]), int(cfg.get("dpi", 300)))