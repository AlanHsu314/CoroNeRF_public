#################################################################
### EVALUATION SCRIPTS
#################################################################

from __future__ import annotations

import logging
import yaml
import json
from pathlib import Path
from typing import Union

import torch
import numpy as np

from ..artifacts.scalar_fields import available_scalar_fields, evaluate_scalar_shells
from ..artifacts.intensity import render_rays_chunked

# coronerf
from .. import ExperimentDirs, DotDict
from ..experiments.resolve_paths import resolve_cfg_paths, remap_cfg_paths
from ..pipeline.state import PipelineState
from ..pipeline.builders.data import load_data
from ..pipeline.builders.fields import build_fields
from ..pipeline.builders.model import build_model
from ..pipeline.builders.renderer import build_renderer
from ..train.checkpoints import load_checkpoint

# metrics
from .metrics import (
    masked_mse, 
	masked_mae,
    psnr_from_mse, 
    masked_log_mse,
	masked_log_mae,
	summarize_image_metric_matrix,
)
from ..plot.plot_eval_snapshot import plot_eval_snapshot
from ..plot.plot_eval_history import plot_eval_history

# bencharmking
from ..benchmark.config import normalize_config

###############################################
### builders
###############################################

def build_state_from_run(run_dir: Path, 
						 path_remap: dict | None = None,
						 ) -> PipelineState:
	'''
	builds pipeline state object, includes

	[1] experiment dirs
	[2] loads data
	[3] fields
	[4] model architecture
	[5] renderer

	Note: does NOT load checkpoint, need to do that afterwards
	'''
	logger = logging.getLogger('coroNeRF.tools.evaluate_coronerf.build_state_from_run')
	with open(run_dir / 'config.yaml', 'r') as f:
		raw_cfg = yaml.safe_load(f)

	# apply semantic renormalization for safety
	raw_cfg = normalize_config(raw_cfg)

	# apply remapping for local runs instead of cluster
	raw_cfg = remap_cfg_paths(raw_cfg, path_remap=path_remap, root=run_dir)

	# for new runs, config.yaml should have absolute input paths
	# only resolve if its a legacy run with relative paths
	dataset_dir = raw_cfg.get('dataset_dir', None)
	if isinstance(dataset_dir, str) and not Path(dataset_dir).is_absolute():
		raw_cfg = resolve_cfg_paths(raw_cfg, root = run_dir)
	
	cfg = DotDict(raw_cfg)

	# resolve relative paths against run_dir (where config.yaml is)	
	#resolve_cfg_paths(cfg, root = run_dir.parent)

	## IMPORTANT 
	'''
	load_data() only loads gt data if its 'train' or 'resume'
	in fact, exp with results should be in these 2 cases
	so, we set mode to train right now, although it should already be in train
	'''
	if cfg.mode not in ['train', 'resume']:
		logger.info(f'cfg.mode is {cfg.mode}, not train or resume. changing mode to train, proceed with caution')
		cfg.mode = 'train'

	# dirs
	dirs = ExperimentDirs(
		run_dir = run_dir,
		checkpoints = run_dir / 'checkpoints',
		logs = run_dir / 'logs',
		figs = run_dir / 'figs',
		artifacts = run_dir / 'artifacts',
	)

	# state
	state = PipelineState(cfg, dirs = dirs, wb_run = None)

	# data
	load_data(state, **state.cfg.load_data_kwargs)

	# builders
	build_fields(state)
	build_model(state)
	build_renderer(state)

	logger.info(f'successfully loaded and built pipeline state')
	return state

def attach_optimizer_for_ckpt_loading(state: PipelineState):
	# b/c load_checkpoint() requires optimizer to exist, add on one
	# create same optimizer structure as training (but lr doesn't matter for evaluation)

	renderer_model = state.renderer.cfg['renderer_model']
	if renderer_model == 'uniform':
		params = state.renderer.model.parameters()
	elif renderer_model == 'nerfacc':
		params = state.renderer.parameters() # includes occ grid
	else:
		raise ValueError(f'Unknown renderer_model: {renderer_model}')
	
	state.optimizer = torch.optim.AdamW(params, lr=1e-6, weight_decay = 0.0)

def load_ckpt(state: PipelineState, ckpt_path: Path):
	'''
	given pipeline state, loads checkpoint into that
	'''

	# load "artificial" optimizer in
	attach_optimizer_for_ckpt_loading(state)
	
	# call loading ckpt function
	load_checkpoint(state, ckpt_path)

	# set to eval
	state.model.eval()
	state.renderer.eval()

###############################################
### renderers and evaluation
###############################################

@torch.no_grad()
def render_view(state: PipelineState, rays_o: torch.Tensor, rays_d: torch.Tensor, chunk_size: int = 8192) -> np.ndarray:
	'''
	given a viewpoint, render it

	rays_o: (3,) or (N, 3)
	rays_d: (H, W, 3)
	returns I: (H, W, C)
	'''

	# checks
	if rays_d.ndim == 3:
		H, W, _ = rays_d.shape
		rd = rays_d.reshape(-1, 3)
	else:
		raise ValueError(f'Expected rays_d shape (H,W,3), got {rays_d.shape}')
	
	N = rd.shape[0] # number of total rays, H*W

	if rays_o.ndim == 1:
		ro = rays_o[None, :].repeat(N, 1)
	else:
		ro = rays_o
		assert ro.shape[0] == N

	I_flat = render_rays_chunked(
		state,
		rays_o=ro,
		rays_d=rd,
		chunk_rays=chunk_size,
	).numpy()

	return I_flat.reshape(H, W, -1)

def eval_images(state: PipelineState, chunk_size: int = 8192, log_eps: float = 1e-12) -> dict:
	'''
	evaluate images on test data

	chunk size: batch size for rays into renderer
	'''
	logger = logging.getLogger('coroNeRF.tools.evaluate_coronerf.eval_images')

	# define data
	test = state.test_data # change to test data in future datasets
	imgs_gt = test['imgs'].numpy()  # (V, H, W, C)
	mask = test['mask'].numpy()     # (H, W), boolean
	rays_o = test['rays_o']         # (V, 3)
	rays_d = test['rays_d']         # (V, H, W, 3)

	V, H, W, C = imgs_gt.shape

	# per view diagnostics
	per_view = []
	for v in range(V):
		logger.debug(f'view {v}/{V}')

		# forward pass through render
		pred = render_view(state, rays_o[v], rays_d[v], chunk_size = chunk_size) # (H, W, C)
		gt = imgs_gt[v]

		# per channel metrics
		mse_c = []
		mae_c = []
		psnr_c = []
		logmse_c = []
		logmae_c = []

		# loop through channel
		for c in range(C):
			# metrics
			mse = masked_mse(pred[..., c], gt[..., c], mask)
			mae = masked_mae(pred[..., c], gt[..., c], mask)
			logmse = masked_log_mse(pred[..., c], gt[..., c], mask, eps = log_eps)
			logmae = masked_log_mae(pred[..., c], gt[..., c], mask, eps = log_eps)


			# get data range for psnr
			vals = gt[..., c][mask]
			dr = float(np.percentile(vals, 99) - np.percentile(vals, 1))
			if dr <= 0:
				dr = float(np.max(vals) - np.min(vals)) if vals.size else 1.0
			if dr <= 0: # if its still less than 0, just set as 1
				dr = 1.0

			# append
			mse_c.append(mse)
			mae_c.append(mae)
			psnr_c.append(psnr_from_mse(mse, data_range = dr))
			logmse_c.append(logmse)
			logmae_c.append(logmae)
		

		# append
		per_view.append({
			'view': int(v),
			'mse_per_channel': mse_c,
			'mae_per_channel': mae_c,
			'psnr_per_channel': psnr_c,
			'logmse_per_channel': logmse_c,
			'logmae_per_channel': logmae_c,
		})
	
	# aggregate, (V, C)
	mse_all = np.array([x['mse_per_channel'] for x in per_view], dtype = float)
	mae_all = np.array([x['mae_per_channel'] for x in per_view], dtype = float)
	psnr_all = np.array([x['psnr_per_channel'] for x in per_view], dtype = float) 
	logmse_all = np.array([x['logmse_per_channel'] for x in per_view], dtype = float) 
	logmae_all = np.array([x['logmae_per_channel'] for x in per_view], dtype = float)

	# summary (C,)
	mse_summary = summarize_image_metric_matrix(mse_all)
	mae_summary = summarize_image_metric_matrix(mae_all)
	psnr_summary = summarize_image_metric_matrix(psnr_all)
	logmse_summary = summarize_image_metric_matrix(logmse_all)
	logmae_summary = summarize_image_metric_matrix(logmae_all)

	# overall summary with per-view stats too
	summary = {
		'num_views': int(V),

		'mse_mean_per_channel': mse_summary['mean_per_channel'],
		'mse_std_per_channel': mse_summary['std_per_channel'],
		'mse_median_per_channel': mse_summary['median_per_channel'],
		'mse_mean_over_channels': mse_summary['mean_over_channels'],
		'mse_median_over_channels': mse_summary['median_over_channels'],

		'mae_mean_per_channel': mae_summary['mean_per_channel'],
		'mae_std_per_channel': mae_summary['std_per_channel'],
		'mae_median_per_channel': mae_summary['median_per_channel'],
		'mae_mean_over_channels': mae_summary['mean_over_channels'],
		'mae_median_over_channels': mae_summary['median_over_channels'],

		'psnr_mean_per_channel': psnr_summary['mean_per_channel'],
		'psnr_std_per_channel': psnr_summary['std_per_channel'],
		'psnr_median_per_channel': psnr_summary['median_per_channel'],
		'psnr_mean_over_channels': psnr_summary['mean_over_channels'],
		'psnr_median_over_channels': psnr_summary['median_over_channels'],

		'logmse_mean_per_channel': logmse_summary['mean_per_channel'],
		'logmse_std_per_channel': logmse_summary['std_per_channel'],
		'logmse_median_per_channel': logmse_summary['median_per_channel'],
		'logmse_mean_over_channels': logmse_summary['mean_over_channels'],
		'logmse_median_over_channels': logmse_summary['median_over_channels'],

		'logmae_mean_per_channel': logmae_summary['mean_per_channel'],
		'logmae_std_per_channel': logmae_summary['std_per_channel'],
		'logmae_median_per_channel': logmae_summary['median_per_channel'],
		'logmae_mean_over_channels': logmae_summary['mean_over_channels'],
		'logmae_median_over_channels': logmae_summary['median_over_channels'],

		'per_view': per_view,
	}

	logger.debug(f'done evaluating images')

	return summary

def run_evaluation(post_training: bool = False,
				   state: Union[PipelineState, None] = None, 
				   run_dir: Union[Path, None] = None, 
				   step: Union[int, None] = None,  
				   ckpt_path: Union[Path, None] = None):
	
	'''
	2 modes, during training, and post training

	during training:
	[1] post_training = False
	[2] state is passed in
	[3] run_dir = None
	[4] step is passed in 
	[5] ckpt_path = None

	post training:
	[1] post_training = True
	[2] state is loaded from run_dir
	[3] run_dir is passed in
	[4] step is None 
	[5] ckpt_path can be passed in (or if not, automatically loads latest from run_dir)
	
	'''
	
	# logging
	logger = logging.getLogger('coroNeRF.tools.evaluate_coronerf')

	if post_training: # need to load in state
		assert run_dir is not None

		if ckpt_path is None: # get latest one
			latest_ckpt = sorted((run_dir / 'checkpoints').glob('ckpt_step*.pt'))
			if not latest_ckpt:
				raise ValueError(f'Cannot find latest checkpoint in run {run_dir}')
			else:
				ckpt_path = latest_ckpt[-1]

		logger.info(f'post training mode, loading in state and checkpoint...')

		# build pipeline state from run dir
		state = build_state_from_run(run_dir)

		# just the state, still need to load checkpoint
		load_ckpt(state, ckpt_path)

		logger.info(f'successfully loaded checkpoint from {ckpt_path}')

	else:
		assert state is not None
		assert step is not None

	if getattr(state, "no_save", False):
		logger.debug("io.no_save = True -> skipping evaluation")
		return

	# create eval folder out dir
	out_dir = state.dirs.artifacts / 'eval' / 'json'
	out_dir.mkdir(exist_ok = True, parents = True)

	# new summary pipeline
	# grab eval config
	if 'eval' not in state.cfg.keys():
		cfg_eval = {
			'chunk_size': 8192,
			'shell_radii': [1.1, 1.5, 2.0, 2.6, 3.0, 4.0],
			'log_eps': 1e-12,
			'shell_metric_bands': {
				'inner': [1.1, 2.0],
				'mid': [1.1, 2.6],
				'full': [None, None],
			},
		}
	else:
		cfg_eval = state.cfg.eval

	# get metric bands
	bands_cfg = cfg_eval.get('shell_metric_bands', None)
	if bands_cfg is not None:
		metric_bands = {
			name: (
				vals[0] if vals[0] is not None else None,
				vals[1] if vals[1] is not None else None,
			)
			for name, vals in bands_cfg.items()
		}
	else:
		metric_bands = None
	
	# finally evaluate imgs and density
	image_summary = eval_images(
		state,
		chunk_size = cfg_eval['chunk_size'],
		log_eps = cfg_eval.get('log_eps', 1e-12)
	)

	# new eval scalar shells
	scalar_field_eval = {}
	for quantity in available_scalar_fields(state):
		scalar_field_eval[quantity] = evaluate_scalar_shells(
			state,
			quantity = quantity,
			radii = cfg_eval['shell_radii'],
			metric_bands = metric_bands,
		)

	# save it
	if post_training:
		report = {
			'run_dir': str(state.dirs.run_dir),
			'ckpt': str(ckpt_path),
			'image_eval': image_summary,
			'scalar_field_eval': scalar_field_eval,
		}

		out_path = out_dir / f'eval_final_report.json'
		with open(out_path, 'w') as f:
			json.dump(report, f, indent = 2)

		logger.info(f'wrote summary to {out_path}')

		# eventually move plotting here
		plot_eval_history(run_dir = state.dirs.run_dir)
		logger.debug('Plotted shell eval history. Exiting evaluation now.')

	else:
		report = {
			'run_dir': str(state.dirs.run_dir),
			'step': f'{step:06d}',
			'image_eval': image_summary,
			'scalar_field_eval': scalar_field_eval,
		}

		out_path = out_dir / f'eval_step{step:06d}.json'
		with open(out_path, 'w') as f:
			json.dump(report, f, indent = 2)

		logger.info(f'wrote summary to {out_path}')

		# plotting
		plot_eval_snapshot(json_path = out_path)

		logger.debug('Plotted shell error vs radius. Exiting evaluation now.')

