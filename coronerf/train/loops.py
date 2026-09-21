############################################################
### training loops
############################################################

'''
eventually ill move regularizers and actual loss computations to another script, but they are here for now
'''

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..util.torch import match_field
from ..geom.coords import llr_to_xyz

from ..models.field_outputs import get_log_ne_from_raw, get_log_temp_from_raw

from ..artifacts.scalar_fields import available_scalar_fields
from ..tools.debug_renders import render_view_debug, render_scalar_shell_debug
from ..tools.monitoring import save_loss_curve
from .checkpoints import save_checkpoint, load_checkpoint
from ..eval.run_eval import run_evaluation

#################################################
### loss helpers
#################################################

def _compute_log_ratio_cosh_loss(pred: torch.Tensor, target: torch.Tensor, eps: float) -> tuple[torch.Tensor, dict]:
    rel = (pred + eps) / (target + eps)
    loss_main = torch.log(torch.cosh(torch.log(rel).abs())).mean()
    return loss_main, {}

def _resolve_asinh_scale_tensor(
    pred: torch.Tensor,
    batch: dict,
    loss_cfg: dict,
    eps: float,
) -> torch.Tensor:
    """
    Resolve per-pixel/per-channel asinh scale tensor.

    Modes:
    - auto: use target_sigma if present, else fixed scale
    - target_sigma: require batch['target_sigma']
    - fixed: always use asinh_scale_by_channel or asinh_scale
    """
    mode = str(loss_cfg.get('asinh_scale_mode', 'auto')).strip().lower()

    def _fixed_scale() -> torch.Tensor:
        # grab per-channel scales, if available
        asinh_scale_by_channel = loss_cfg.get('asinh_scale_by_channel', None)
        if asinh_scale_by_channel is not None:
            vals = torch.tensor(
                [float(v) for v in asinh_scale_by_channel],
                dtype=pred.dtype,
                device=pred.device,
            )
            if vals.ndim != 1 or vals.shape[0] != pred.shape[-1]:
                raise ValueError(
                    f'asinh_scale_by_channel must have length {pred.shape[-1]}, got {tuple(vals.shape)}'
                )
            return vals.view(1, -1).expand_as(pred).clamp_min(eps)

        # else grab the global version
        asinh_scale = float(loss_cfg.get('asinh_scale', 1.0))
        if asinh_scale <= 0.0:
            raise ValueError(f'asinh_scale must be positive, got {asinh_scale}')
        return torch.full_like(pred, fill_value=asinh_scale).clamp_min(eps)

    if mode == 'target_sigma':
        if 'target_sigma' not in batch:
            raise KeyError(
                "asinh_scale_mode='target_sigma' requires batch['target_sigma']"
            )
        return batch['target_sigma'].to(pred.device).clamp_min(eps)

    if mode == 'fixed':
        return _fixed_scale()

    if mode == 'auto':
        if 'target_sigma' in batch:
            return batch['target_sigma'].to(pred.device).clamp_min(eps)
        return _fixed_scale()

    raise ValueError(
        f"Unknown train.image_loss.asinh_scale_mode='{mode}'. "
        f"Expected one of: auto, target_sigma, fixed"
    )

def _compute_hetero_gaussian_asinh_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict,
    loss_cfg: dict,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    lambda_gauss = float(loss_cfg.get('lambda_gauss', 1.0))
    lambda_asinh = float(loss_cfg.get('lambda_asinh', 0.1))

    loss_terms = {}
    loss_main = torch.zeros((), device=pred.device, dtype=pred.dtype)

    # Gaussian term requires target_sigma
    if lambda_gauss > 0.0:
        if 'target_sigma' not in batch:
            raise KeyError(
                "Batch is missing 'target_sigma'. A positive lambda_gauss requires "
                "observation_noise.model=hetero_gaussian (or another source of target_sigma). "
                "For clean asinh-only runs, set lambda_gauss=0."
            )
        sigma = batch['target_sigma'].to(pred.device).clamp_min(eps)
        loss_gauss = ((pred - target) / sigma).pow(2).mean()
        loss_main = loss_main + lambda_gauss * loss_gauss
        loss_terms['loss_gauss'] = float(loss_gauss.item())

    # asinh term can use target_sigma if present, otherwise fallback scales from config
    if lambda_asinh > 0.0:
        asinh_scale = _resolve_asinh_scale_tensor(
            pred=pred,
            batch=batch,
            loss_cfg=loss_cfg,
            eps=eps,
        )

        delta_asinh = torch.asinh(pred / asinh_scale) - torch.asinh(target / asinh_scale)

        if bool(loss_cfg.get('use_huber_asinh', False)):
            delta = float(loss_cfg.get('huber_delta', 0.05))
            loss_asinh = F.huber_loss(
                delta_asinh,
                torch.zeros_like(delta_asinh),
                reduction='mean',
                delta=delta,
            )
        else:
            loss_asinh = delta_asinh.abs().mean()

        loss_main = loss_main + lambda_asinh * loss_asinh
        loss_terms['loss_asinh'] = float(loss_asinh.item())

    if lambda_gauss <= 0.0 and lambda_asinh <= 0.0:
        raise ValueError('At least one of lambda_gauss or lambda_asinh must be > 0')

    return loss_main, loss_terms

def _compute_image_loss_from_cfg(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict,
    loss_cfg: dict,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    """
    Shared image-loss dispatch for training and validation.
    """
    loss_cfg = dict(loss_cfg or {})
    name = loss_cfg.get('name', 'log_ratio_cosh')

    if name == 'log_ratio_cosh':
        return _compute_log_ratio_cosh_loss(pred, target, eps)

    if name == 'hetero_gaussian_asinh':
        return _compute_hetero_gaussian_asinh_loss(
            pred=pred,
            target=target,
            batch=batch,
            loss_cfg=loss_cfg,
            eps=eps,
        )

    raise NotImplementedError(f'Unknown image loss name={name}')


def _compute_main_image_loss(
    pipeline,
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    loss_cfg = dict(pipeline.cfg.train.get('image_loss', {}) or {})
    return _compute_image_loss_from_cfg(
        pred=pred,
        target=target,
        batch=batch,
        loss_cfg=loss_cfg,
        eps=eps,
    )

#####################################
### val loss
#####################################

def _resolve_validation_loss_cfg(pipeline) -> dict:
    """
    Resolve the image loss used for heldout validation.

    By default this mirrors train.image_loss. For loss ablations with a
    Gaussian term, override train.validation_loss.image_loss so validation
    is a stable image-space diagnostic, usually fixed-scale asinh-only.
    """
    train_loss_cfg = dict(pipeline.cfg.train.get('image_loss', {}) or {})
    val_cfg = dict(pipeline.cfg.train.get('validation_loss', {}) or {})
    override = val_cfg.get('image_loss', None)

    if override is None:
        return train_loss_cfg

    out = dict(train_loss_cfg)
    out.update(dict(override or {}))
    return out

@torch.no_grad()
def compute_validation_loss(
    pipeline,
    batch_rays: int = 4096,
    max_batches=None,
    num_workers: int = 0,
    loss_cfg: dict | None = None,
) -> tuple[float, dict]:
    """
    Compute heldout image loss on state.ds_test.

    This is intentionally image-loss only: it does not include density
    oracle regularizers or model smoothness penalties.
    
    loss_cfg: if given, use this image-loss cfg verbatim (e.g. a fixed all-channel
    asinh scale for a cross-condition global eval); otherwise resolve from the run.
    """
    device = pipeline.renderer.device
    loss_eps = pipeline.cfg.train.get('loss_eps', 1e-6)
    loss_cfg = dict(loss_cfg) if loss_cfg is not None else _resolve_validation_loss_cfg(pipeline)

    dl_val = DataLoader(
        pipeline.ds_test,
        batch_size=int(batch_rays),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )

    was_training = pipeline.model.training
    pipeline.model.eval()
    pipeline.renderer.eval()

    total_loss = 0.0
    total_count = 0
    term_sums: dict[str, float] = {}
    num_batches = 0

    for batch in dl_val:
        if max_batches is not None and num_batches >= int(max_batches):
            break

        target = batch['target'].to(device)
        ro = batch['ray_o'].to(device)
        rd = batch['ray_d'].to(device)

        pred = pipeline.renderer(rays_o=ro, rays_d=rd)
        loss, terms = _compute_image_loss_from_cfg(
            pred=pred,
            target=target,
            batch=batch,
            loss_cfg=loss_cfg,
            eps=loss_eps,
        )

        n = int(target.shape[0])
        total_loss += float(loss.item()) * n
        total_count += n
        num_batches += 1

        for key, val in terms.items():
            term_sums[key] = term_sums.get(key, 0.0) + float(val) * n

    if was_training:
        pipeline.model.train()
        pipeline.renderer.train()

    if total_count <= 0:
        return float('nan'), {'num_val_rays': 0, 'num_val_batches': 0}

    meta = {
        'num_val_rays': int(total_count),
        'num_val_batches': int(num_batches),
        'loss_name': loss_cfg.get('name', 'log_ratio_cosh'),
    }
    for key, val in term_sums.items():
        meta[f'val_{key}'] = float(val / total_count)

    return float(total_loss / total_count), meta

################################################
### main loop
################################################

def train_loop(pipeline, resume: bool = False):
    '''
    Trains the nerf over multiple epochs
    '''
    lg = logging.getLogger('coroNeRF.train')

    # extra device checks, although they should all already be on device
    device = pipeline.renderer.device

    # dataloader
    dl = DataLoader(pipeline.ds_train, 
                    batch_size=pipeline.cfg.train['batch_rays'], 
                    shuffle=True, 
                    num_workers=0, 
                    pin_memory=True)
    
    # which parameters to train
    if pipeline.renderer.cfg['renderer_model'] == 'uniform': # no need to learn sigma scale
        parameters = pipeline.renderer.model.parameters()
    elif pipeline.renderer.cfg['renderer_model'] == 'nerfacc': # learn sigma scale
        parameters = pipeline.renderer.parameters()
    else:
        raise ValueError("Undefined cfg renderer_model")
    
    # optimizer
    opt = torch.optim.AdamW(parameters, lr=pipeline.cfg.train['lr'], weight_decay = pipeline.cfg.train.get('weight_decay', 0))
    pipeline.optimizer = opt # save for later logging

    # resume with ckpt if needed
    if resume:
        last_ckpts = sorted(pipeline.dirs.checkpoints.glob("ckpt_step*.pt"))
        if last_ckpts:
            load_checkpoint(pipeline, last_ckpts[-1])
            step = pipeline.start_step
        else:
            step = 0
            # loss history
            pipeline.step_hist = []
            pipeline.loss_hist = []
            pipeline.loss_main_hist = []
            pipeline.val_step_hist = []
            pipeline.val_loss_hist = []
            pipeline.val_loss_meta_hist = []
    else:
        step = 0
        # loss history
        pipeline.step_hist = []             # x-axis step
        pipeline.loss_hist = []             # total loss (incl. regualrizers if set)
        pipeline.loss_main_hist = []        # main images loss, used to compare to val
        pipeline.val_step_hist = []
        pipeline.val_loss_hist = []         # image loss on heldout set
        pipeline.val_loss_meta_hist = []

    # Backward-compatible history fields for older checkpoints, basically before we implemented val hist
    # can take away after a while
    if not hasattr(pipeline, 'loss_main_hist') or pipeline.loss_main_hist is None:
        pipeline.loss_main_hist = []
    if not hasattr(pipeline, 'val_step_hist') or pipeline.val_step_hist is None:
        pipeline.val_step_hist = []
    if not hasattr(pipeline, 'val_loss_hist') or pipeline.val_loss_hist is None:
        pipeline.val_loss_hist = []
    if not hasattr(pipeline, 'val_loss_meta_hist') or pipeline.val_loss_meta_hist is None:
        pipeline.val_loss_meta_hist = []
    
    # train hyperparameters
    loss_eps = pipeline.cfg.train.get('loss_eps', 1e-6)
    log_every = pipeline.cfg.train.get('log_every', 100)
    shell_every = pipeline.cfg.train.get('shell_debug_every', 1000)
    render_every = pipeline.cfg.train.get('render_every', 1000)
    ckpt_every = pipeline.cfg.train.get('checkpoint_every', 2000)
    loss_every = pipeline.cfg.train.get('loss_curve_every', 1000)
    ne_every = pipeline.cfg.train.get('ne_every', 1)
    max_keep = pipeline.cfg.train.get('max_keep', 2)

    # nerfacc
    occ_update_every = pipeline.cfg.train.get('occ_update_every', 16)

    # eval hyperparameters
    eval_cfg = pipeline.cfg.get('eval', dict())
    eval_inline = eval_cfg.get('enable_inline', False)
    eval_every = eval_cfg.get('eval_every', ckpt_every)

    # val loss hyperparameters
    val_cfg = pipeline.cfg.train.get('validation_loss', {}) or {}
    val_enabled = bool(val_cfg.get('enabled', False))
    val_every = int(val_cfg.get('every') or eval_every or loss_every or ckpt_every)
    val_batch_rays = int(val_cfg.get('batch_rays', pipeline.cfg.train.get('batch_rays', 1024)))
    val_max_batches = val_cfg.get('max_batches', None)
    val_num_workers = int(val_cfg.get('num_workers', 0))

    # regularization
    lambda_ne = pipeline.cfg.train.get('lambda_ne', 0.0)
    if lambda_ne > 0.0:
        density_supervision = True
        N_ne = pipeline.cfg.train.get('batch_ne_points', 8192)
        lg.info(f'density regularization on, lambda_ne {lambda_ne:.4f}, N_ne {N_ne}')
    else:
        density_supervision = False

    # spherical-grid smoothness regularization
    lambda_smooth = pipeline.cfg.train.get('lambda_smooth', 0.0)
    smooth_every = pipeline.cfg.train.get('smooth_every', 1)
    smooth_kwargs = pipeline.cfg.train.get('smooth_kwargs', {})
    use_smoothness = (lambda_smooth > 0.0) and hasattr(pipeline.model, 'smoothness_loss')
    if lambda_smooth > 0.0:
        if use_smoothness:
            lg.info(f'smoothness regularization on, lambda_smooth {lambda_smooth:.4e}')
        else:
            lg.warning(f'lambda_smooth > 0 but model has no smoothness_loss(), so ignoring it')

    # main loop
    lg.info('training coronerf...')
    while step < pipeline.cfg.train['max_steps']:
        for batch in dl:
            step += 1

            # batch load to device
            target = batch["target"].to(device)
            ro     = batch["ray_o"].to(device)
            rd     = batch["ray_d"].to(device)

            # adaptive sampling occupancy grid update
            if pipeline.renderer.cfg['renderer_model'] == 'nerfacc' and  \
                step % occ_update_every == 0:
                with torch.no_grad():
                    pipeline.renderer.update_occupancy(step)

            # predict with model
            pred = pipeline.renderer(rays_o = ro, 
                                    rays_d = rd)
            
            # main image loss
            loss_main, loss_main_terms = _compute_main_image_loss(
                pipeline = pipeline,
                pred = pred,
                target = target,
                batch = batch,
                eps = loss_eps,
            )

            # the over gradient update starts with the image term
            loss = loss_main

            # regularizers

            # oracle supervision
            if density_supervision and (step % ne_every == 0):
                loss_ne = _get_density_regularizer(pipeline, N_ne = N_ne)
                loss = loss + lambda_ne * loss_ne
            else:
                loss_ne = None

            # smoothness regularizer
            if use_smoothness and (step % smooth_every == 0):
                loss_smooth = pipeline.model.smoothness_loss(**smooth_kwargs)
                loss = loss + lambda_smooth * loss_smooth
            else:
                loss_smooth = None

            # gradient update
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0) # useful for early training
            opt.step()

            # save
            pipeline.step_hist.append(step)
            pipeline.loss_hist.append(loss.item())
            pipeline.loss_main_hist.append(float(loss_main.item()))

            # save on every
            if step % log_every == 0:
                # main message
                msg = f'step {step:06d} loss={loss.item():.4e} loss_main={loss_main.item(): .4e}'
                
                # hetero-gauss loss decomposition
                if 'loss_gauss' in loss_main_terms:
                    msg += f' loss_gauss={loss_main_terms["loss_gauss"]:.4e}'
                if 'loss_asinh' in loss_main_terms:
                    msg += f' loss_asinh={loss_main_terms["loss_asinh"]:.4e}'
                
                # add regularizer msg
                if loss_ne is not None:
                    msg += f' loss_ne={loss_ne.item():.4e}'
                if loss_smooth is not None:
                    msg += f' loss_smooth={loss_smooth.item():.4e}'
                
                # send final msg to logger
                lg.info(msg)

                # wandb
                if pipeline.wb_run is not None:
                    payload = {'step': step, 
                               'loss': float(loss.item()), 
                               'loss_main': float(loss_main.item())}
                    
                    # extra update with decomposition of main loss, if available
                    payload.update(loss_main_terms)
                    
                    if loss_ne is not None:
                        payload['loss_ne'] = float(loss_ne.item())
                    if loss_smooth is not None:
                        payload['loss_smooth'] = float(loss_smooth.item())

                    pipeline.wb_run.log(payload, step = step)
                
            # memory check
            # if step % 100 == 0:
            # 	mb = torch.cuda.memory_allocated() / 1024**2
            # 	lg.info(f"GPU mem allocated: {mb:.1f} MB")

            if shell_every and (step % shell_every == 0):
                shell_debug_radii = pipeline.cfg.train.get('shell_debug_radii', [1.5])
                for quantity in available_scalar_fields(pipeline):
                    for r_target in shell_debug_radii:
                        render_scalar_shell_debug(
                            pipeline, 
                            quantity=quantity, 
                            step=step, 
                            r_target=float(r_target),
                        )

            if render_every and (step % render_every == 0):
                render_view_debug(pipeline, step = step, vp_idx = 0)

            if val_enabled and val_every and (step % val_every == 0):
                val_loss, val_meta = compute_validation_loss(
                    pipeline,
                    batch_rays=val_batch_rays,
                    max_batches=val_max_batches,
                    num_workers=val_num_workers,
                )

                pipeline.val_step_hist.append(int(step))
                pipeline.val_loss_hist.append(float(val_loss))
                pipeline.val_loss_meta_hist.append(val_meta)

                msg = f'step {step:06d} val_loss={val_loss:.4e}'
                for key, val in sorted(val_meta.items()):
                    if key.startswith('val_loss_') or key in {'val_loss_asinh', 'val_loss_gauss'}:
                        msg += f' {key}={float(val):.4e}'
                lg.info(msg)

                if pipeline.wb_run is not None:
                    payload = {'step': step, 'val_loss': float(val_loss)}
                    payload.update({
                        k: float(v)
                        for k, v in val_meta.items()
                        if isinstance(v, (int, float))
                    })
                    pipeline.wb_run.log(payload, step=step)
            
            if loss_every and (step % loss_every == 0):
                save_loss_curve(pipeline)

            if ckpt_every and (step % ckpt_every == 0):
                save_checkpoint(pipeline, step = step, max_keep = max_keep)
            
            if eval_inline and (step % eval_every == 0):
                run_evaluation(state = pipeline, step = step)

            if step >= pipeline.cfg.train['max_steps']:
                save_loss_curve(pipeline)
                save_checkpoint(pipeline, step = step, max_keep = max_keep)
                if eval_inline:
                    # post-training case for run_eval
                    if getattr(pipeline, "no_save", False):
                        lg.debug("io.no_save = True -> skipping evaluation save")
                    else:
                        run_evaluation(post_training = True, run_dir = pipeline.dirs.run_dir)

                lg.info('done training coronerf!')	
                return

    # final saves, identical to last exit case in loop
    save_loss_curve(pipeline)
    save_checkpoint(pipeline, step = step, max_keep = max_keep)
    if eval_inline: 
        # post-training case for run_eval
        if getattr(pipeline, "no_save", False):
            lg.debug("io.no_save = True -> skipping evaluation save")
        else:
            run_evaluation(post_training = True, run_dir = pipeline.dirs.run_dir)

    lg.info('done training coronerf!')	

def _get_density_regularizer(pipeline, N_ne: int = 8192, chunk_ne: int = 2048):
    '''
    called during training
    [.] basically uses fit_ne train loop to supervise density
    [.] code taken from that function, maybe later on can combine if we keep using this

    inputs
    N_ne: number of density points to sample
    '''
    device = pipeline.renderer.device
    dtype = pipeline.renderer.dtype

    r_shell_min = 1.1
    r_shell_max = 3

    lon_axis = pipeline.ne_field.lon_axis.to(device = device, dtype = dtype)
    lat_axis = pipeline.ne_field.phi_axis.to(device = device, dtype = dtype)
    r_axis = pipeline.ne_field.r_axis.to(device = device, dtype = dtype)

    lon_min, lon_max = lon_axis[0].item(), lon_axis[-1].item()
    lat_min, lat_max = lat_axis[0].item(), lat_axis[-1].item()
    r_min, r_max = r_shell_min, r_shell_max #r_axis[0].item(), r_axis[-1].item()

    lon = torch.empty(N_ne, device = device, dtype = dtype).uniform_(lon_min, lon_max)
    lat = torch.empty(N_ne, device = device, dtype = dtype).uniform_(lat_min, lat_max)
    r = torch.empty(N_ne, device = device, dtype = dtype).uniform_(r_min, r_max)
    x_llr = torch.stack([lon, lat, r], dim = -1) # (Ne, 3)

    x_xyz = llr_to_xyz(x_llr)

    with torch.no_grad():
        log_ne_gt = pipeline.ne_field(match_field(pipeline.ne_field.field, x_xyz)) # (Ne,)

    x_AABB = (x_xyz / pipeline.renderer.aabb_scale).to(device = device, dtype = dtype)
    
    # chunked MSE to reduce peak activations
    total_loss = 0.0
    count = 0
    for start in range(0, N_ne, chunk_ne):
        end = min(start + chunk_ne, N_ne)
        #log_pred_chunk = pipeline.model(x_AABB[start:end]).view(-1)
        raw_chunk = pipeline.model(x_AABB[start:end])
        target = pipeline.shared.get('reconstruction_target', 'ne')
        log_pred_chunk = get_log_ne_from_raw(raw_chunk, target=target).reshape(-1)

        gt_chunk = log_ne_gt[start:end].to(device=device)
        # sum reduction so we can average manually
        total_loss = total_loss + F.mse_loss(log_pred_chunk, gt_chunk, reduction="sum")
        count += (end - start)

    loss_ne = total_loss / count

    return loss_ne

###############################################
### Fit via Ne --> Ne tests
###############################################
def fit_ne_loop(pipeline):
    '''
    directly fit densityMLP to the ground-truth log10(n_e) from self.ne_field

    notes:
    [.] this completely bypasses the LOS renderer: we
        -- sample 3D pts in llr
        -- convert to xyz
        -- query the GT ne_field
        -- train model to reproduce log10 n_e at those points
    '''

    lg = logging.getLogger('coroNeRF.fit_ne')

    # main params
    cfg_fit = pipeline.cfg.get('fit_ne', dict())
    device = pipeline.renderer.device
    dtype = pipeline.renderer.dtype
    model = pipeline.model

    # hyperparams (with defaults)
    lr = cfg_fit.get('lr', 3e-4)
    batch_pts = cfg_fit.get('batch_pts', 65536)
    max_steps = cfg_fit.get('max_steps', 20000)
    r_min = cfg_fit.get('r_min', float(pipeline.ne_field.r_axis[0].item()))
    r_max = cfg_fit.get('r_max', float(pipeline.ne_field.r_axis[-1].item()))
    log_every = cfg_fit.get('log_every', 100)
    shell_every = cfg_fit.get('shell_debug_every', 1000)
    r_shell = cfg_fit.get('r_shell', 1.5)
    ckpt_every = cfg_fit.get('checkpoint_every', 2000)
    max_keep = cfg_fit.get('max_keep', 2)
    wd = cfg_fit.get('weight_decay', 1e-5)

    # optimizer
    pipeline.optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)

    # histories
    pipeline.loss_hist = []
    pipeline.step_hist = []
    pipeline.start_step = 0 # if, for some reason, we want to add resume function later

    # get axes ranges for lon, lat (r determined by hyperparameters)
    lon_axis = pipeline.ne_field.lon_axis.to(device = device, dtype=dtype)
    phi_axis = pipeline.ne_field.phi_axis.to(device = device, dtype=dtype)

    lon_min, lon_max = lon_axis[0].item(), lon_axis[-1].item()
    phi_min, phi_max = phi_axis[0].item(), phi_axis[-1].item()

    lg.info(
        f'starting density-field fit: '
        f'lr = {lr}, batch_points = {batch_pts}, steps = {max_steps}, '
        f'r in [{r_min}, {r_max}]'
    )

    # main loop (build dataset directly as we train)
    step = pipeline.start_step 
    while step < max_steps:
        step += 1

        # sample random points (not uniformly) in llr
        lon = torch.empty(batch_pts, device = device, dtype = dtype).uniform_(lon_min, lon_max)
        lat = torch.empty(batch_pts, device = device, dtype = dtype).uniform_(phi_min, phi_max)
        r = torch.empty(batch_pts, device = device, dtype = dtype).uniform_(r_min, r_max)

        x_llr = torch.stack([lon, lat, r], dim = -1) # (N, 3)

        # llr -> xyz
        x_xyz = llr_to_xyz(x_llr)

        # get ground truth values to match (make sure to match field first)
        with torch.no_grad():
            log_ne_gt = pipeline.ne_field(match_field(pipeline.ne_field.field, x_xyz)) # (N,)

        # get predicted values
        # technically dont need to scale, but lets do it for consistency
        x_AABB = (x_xyz / pipeline.renderer.aabb_scale).to(device = device, dtype = dtype) 
        log_ne_pred = model(x_AABB).view(-1)

        # compute loss (mse for now)
        loss = F.mse_loss(log_ne_pred, log_ne_gt.to(device = device))

        # backprop
        pipeline.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        pipeline.optimizer.step()

        # update history
        pipeline.loss_hist.append(loss.item())
        pipeline.step_hist.append(step)

        # save on every
        if step % log_every == 0:
            lg.info(f'step {step:06d} loss={loss.item():.4e}')

            # wandb logging
            if pipeline.wb_run is not None:
                pipeline.wb_run.log({'step': step, 'loss': float(loss.item())}, step = step)

        if shell_every and (step % shell_every == 0):
            render_scalar_shell_debug(pipeline, quantity='ne', step=step, r_target=r_shell)
            save_loss_curve(pipeline)

        if ckpt_every and (step % ckpt_every == 0):
            save_checkpoint(pipeline, step = step, max_keep = max_keep)

    # final saves
    save_loss_curve(pipeline)
    save_checkpoint(pipeline, step = step, max_keep = max_keep)

    lg.info('finished density-field fit.')

