#################################################
### shared generic plotting
#################################################

from __future__ import annotations
from typing import Union

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


from .scalar_fields import ScalarShellProduct, get_scalar_field_spec

def fig_to_numpy(fig):
	# Render a Matplotlib figure to a HxWx3 uint8 NumPy array (RGB).
	fig.canvas.draw()

	# RGBA uint8 view, shape (H, W, 4)
	buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)

	# Drop alpha -> RGB, then convert to BGR for OpenCV
	return buf[:, :, :3][:, :, ::-1].copy()

def save_scalar_shell_triptych(product: ScalarShellProduct, out_path: Union[Path, None] = None, save: bool = True) -> np.ndarray:
    '''
    can be used to 
    [1] save a plot (pass out_path)
    [2] return the plot as a frame, saved or not depending on "save" 
    '''
    
    spec = get_scalar_field_spec(product.quantity)

    fig, axs = plt.subplots(1, 3, figsize = (14, 4), constrained_layout = True)

    im0 = axs[0].imshow(
        product.gt_log,
        origin="lower",
        vmin=product.vmin,
        vmax=product.vmax,
        cmap=spec.cmap,
        aspect="auto",
    )
    axs[0].set_title(f'GT {spec.log_label} @ r={product.r:.2f}')
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(
        product.pred_log,
        origin="lower",
        vmin=product.vmin,
        vmax=product.vmax,
        cmap=spec.cmap,
        aspect="auto",
    )
    axs[1].set_title(f'Pred {spec.log_label} @ r={product.r:.2f}')
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(
        product.diff_log,
        origin="lower",
        cmap="coolwarm",
        aspect="auto",
    )
    axs[2].set_title('Pred - GT')
    fig.colorbar(im2, ax=axs[2])

    if save:
        if out_path is None:
            raise ValueError('asked to save plot, but out_path is None')
        out_path.parent.mkdir(exist_ok = True, parents = True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    frame = fig_to_numpy(fig)
    plt.close(fig)

    return frame









