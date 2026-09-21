#################################################################
## callbacks for loss curves
#################################################################

# mostly chatgpt

import logging
import json
from pathlib import Path

def save_loss_history(pipeline, fname: str = "loss_history.json") -> None:
    if getattr(pipeline, "no_save", False):
        logging.getLogger("coroNeRF.save_loss_history").debug("io.no_save = True -> skipping loss history save")
        return

    if not hasattr(pipeline, "loss_hist") or len(pipeline.loss_hist) == 0:
        return

    steps = getattr(pipeline, "step_hist", list(range(1, len(pipeline.loss_hist) + 1)))

    out_dir = pipeline.dirs.artifacts / 'train'
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        'step': [int(x) for x in steps],
        'loss': [float(x) for x in pipeline.loss_hist],
    }

    if hasattr(pipeline, 'loss_main_hist') and pipeline.loss_main_hist:
        payload['loss_main'] = [float(x) for x in pipeline.loss_main_hist]

    if hasattr(pipeline, 'val_loss_hist') and pipeline.val_loss_hist:
        payload['val_step'] = [int(x) for x in getattr(pipeline, 'val_step_hist', [])]
        payload['val_loss'] = [float(x) for x in pipeline.val_loss_hist]
        payload['val_loss_meta'] = getattr(pipeline, 'val_loss_meta_hist', [])

    out_path = out_dir / fname
    with out_path.open('w') as f:
        json.dump(payload, f, indent=2)

    logging.getLogger("coroNeRF.loss_curve").info(f"Saved loss history to {out_path}")

def save_loss_curve(pipeline, fname: str = "loss_curve.png") -> None:
    """
    Save a simple loss-vs-step curve into the figs directory.
    """
    import matplotlib.pyplot as plt

    if getattr(pipeline, "no_save", False):
        logging.getLogger("coroNeRF.save_loss_curve").debug("io.no_save = True -> skipping loss curve save")
        return

    if not hasattr(pipeline, "loss_hist") or len(pipeline.loss_hist) == 0:
        logging.getLogger("coroNeRF.loss_curve").warning(
            "No losses recorded; skipping loss curve save."
        )
        return

    steps = getattr(pipeline, "step_hist", list(range(1, len(pipeline.loss_hist) + 1)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, pipeline.loss_hist, label='train total')

    if hasattr(pipeline, 'loss_main_hist') and pipeline.loss_main_hist:
        if len(pipeline.loss_main_hist) == len(steps):
            ax.plot(steps, pipeline.loss_main_hist, label='train image')

    if hasattr(pipeline, 'val_loss_hist') and pipeline.val_loss_hist:
        val_steps = getattr(pipeline, 'val_step_hist', [])
        if len(val_steps) == len(pipeline.val_loss_hist):
            ax.plot(val_steps, pipeline.val_loss_hist, marker='o', label='heldout image')

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training and heldout image loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = pipeline.dirs.figs / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logging.getLogger("coroNeRF.loss_curve").info(f"Saved loss curve to {out_path}")

    # save loss history to artifacts
    save_loss_history(pipeline)