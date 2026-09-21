##############################################################
### define pipeline state object
##############################################################

from typing import Dict, Any
import torch
from pathlib import Path

from ..experiments.dirs import ExperimentDirs

class PipelineState:
	def __init__(self, cfg: Dict[str, Any], dirs: ExperimentDirs, wb_run = None):
		self.cfg = cfg
		self.dirs = dirs
		self.device = torch.device(cfg.get('device', 'cpu'))
		self.wb_run = wb_run

		# data
		self.train_data = None
		self.test_data = None
		self.ds_train = None
		self.ds_test = None

		# observational noise
		self.observation_noise_meta = None

		# observation model mismatch (abundance)
		self.observation_mismatch_meta = None

		# channel selection
		self.selected_channel_indices = None
		self.selected_wavelengths = None
		self.selected_line_names = None

		# fields
		self.temp_field = None
		self.ccoef_field = None
		self.ne_field = None

		# ML
		self.model = None
		self.renderer = None
		self.optimizer = None

		# callbacks
		self.loss_hist = []
		self.step_hist = []

		# some initialization calls
		self._setup()

	def _setup(self):
		# some quick initializations
		self.debug = self.cfg.get('debug', False)
		self.no_save = bool(self.cfg.get('io', {}).get('no_save', False))
		self.mode = self.cfg.get('mode', 'train')

		# reconstruction mode
		#self.reconstruction = self.cfg.get('reconstruction', {})
		#self.reconstruction_target = self.reconstruction.get('target', 'ne')
		#self.predicts_temperature = (self.reconstruction_target == 'ne_t')

		# set mode     
		if self.mode == 'vpgen':
			if self.no_save:
				lut_dir = self.cfg.get('LUT_dir', None)
				if lut_dir is None:
					raise ValueError("LUT dir must be set for mode = 'vpgen'")
				self.dataset_dir = Path(lut_dir)
			else:
				if self.dirs.datasets is None:
					raise ValueError("dirs.datasets must be created for mode = 'vpgen'")
				self.dataset_dir = Path(self.dirs.datasets)
		else:
			dataset_dir = self.cfg.get('dataset_dir', None)
			if dataset_dir is None:
				raise ValueError(f"dataset_dir must be set for mode = '{self.mode}'")
			self.dataset_dir = Path(dataset_dir)

		# forward model backend mode
		self.forward_backend = self.cfg.get('forward_backend', 'lut')

		# setup shared params
		self.shared = self.build_shared_context(self.cfg)

	def build_shared_context(self, cfg):
		shared_cfg = cfg.get("shared", {})
		required = ['aabb_scale', 
			  	    'aabb_min', 
					'aabb_max',
					'reconstruction_target']

		missing = [k for k in required if k not in shared_cfg]
		if missing:
			raise KeyError(f"Missing required shared config keys: {missing}")

		return {
			'aabb_scale': float(shared_cfg['aabb_scale']),
			'aabb_min': float(shared_cfg['aabb_min']),
			'aabb_max': float(shared_cfg['aabb_max']),
			'reconstruction_target': shared_cfg['reconstruction_target'],
		}



