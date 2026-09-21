#############################################################
## checkpoints handling
#############################################################

import torch
import logging
from pathlib import Path

from ..experiments.wandb import wandb_log_checkpoint_artifact

# chatGPT function
def save_checkpoint(pipeline, step: int, max_keep: int = 2):
    """
    Save a checkpoint of the model + optimizer.
    Keeps only the most recent `max_keep` checkpoints.
    """

    if getattr(pipeline, 'no_save', False):
        logging.getLogger('coroNeRF.save_checkpoint').debug('io.no_save = True -> skipping checkpoint save')
        return

    ckpt_dir = pipeline.dirs.checkpoints
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    ckpt_path = ckpt_dir / f"ckpt_step{step:08d}.pt"

    # Build checkpoint dict
    ckpt = {
        "step": step,
        "model": pipeline.model.state_dict(),
        "optimizer": pipeline.optimizer.state_dict(),
        "loss_hist": getattr(pipeline, "loss_hist", None),
        "loss_main_hist": getattr(pipeline, "loss_main_hist", None),
        "step_hist": getattr(pipeline, "step_hist", None),
        "val_step_hist": getattr(pipeline, "val_step_hist", None),
        "val_loss_hist": getattr(pipeline, "val_loss_hist", None),
        "val_loss_meta_hist": getattr(pipeline, "val_loss_meta_hist", None),
        "model_name": pipeline.cfg.get("model", {}).get("name", None),
        "reconstruction_target": pipeline.shared.get("reconstruction_target", None),

    }

    torch.save(ckpt, ckpt_path)

    # wandb logging
    if pipeline.wb_run is not None:
        art_name = f'model-{pipeline.dirs.run_dir.name}'
        wandb_log_checkpoint_artifact(pipeline.wb_run, ckpt_path, art_name)

    # ---------------------------
    # prune old checkpoints
    # ---------------------------
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if len(ckpts) > max_keep:
        for old in ckpts[:-max_keep]:
            old.unlink()

    logging.getLogger("coroNeRF.ckpt").info(f"Saved checkpoint: {ckpt_path}")


# chatGPT function
def load_checkpoint(pipeline, path: str | Path):
    ckpt = torch.load(path, map_location=pipeline.device)

    # checks
    ckpt_model_name = ckpt.get("model_name", None)
    ckpt_target = ckpt.get("reconstruction_target", None)
    cfg_model_name = pipeline.cfg.get("model", {}).get("name", None)
    cfg_target = pipeline.shared.get("reconstruction_target", None)

    if ckpt_model_name is not None and cfg_model_name is not None and ckpt_model_name != cfg_model_name:
        logging.getLogger("coroNeRF.ckpt").warning(
            f"Checkpoint model_name ({ckpt_model_name}) != config model_name ({cfg_model_name})"
        )

    if ckpt_target is not None and cfg_target is not None and ckpt_target != cfg_target:
        logging.getLogger("coroNeRF.ckpt").warning(
            f"Checkpoint reconstruction_target ({ckpt_target}) != config reconstruction_target ({cfg_target})"
        )

    # load data and build pipeline
    pipeline.model.load_state_dict(ckpt["model"])
    pipeline.optimizer.load_state_dict(ckpt["optimizer"])
    pipeline.start_step = ckpt["step"]
    pipeline.step_hist = ckpt.get("step_hist") or []
    pipeline.loss_hist = ckpt.get("loss_hist") or []
    pipeline.loss_main_hist = ckpt.get("loss_main_hist") or []
    pipeline.val_step_hist = ckpt.get("val_step_hist") or []
    pipeline.val_loss_hist = ckpt.get("val_loss_hist") or []
    pipeline.val_loss_meta_hist = ckpt.get("val_loss_meta_hist") or []

    logging.getLogger("coroNeRF.ckpt").info(
        f"Loaded checkpoint from {path}, starting at step {pipeline.start_step}"
    )

