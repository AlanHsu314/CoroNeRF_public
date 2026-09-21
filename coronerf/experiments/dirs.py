##################################################################
### file directory helpers
##################################################################

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union, Any
import logging, contextlib
from pathlib import Path
from datetime import datetime
import shutil


import yaml
import pickle

VERBOSITY_TO_LEVEL = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

'''
class to handle all directories during an experiment. notable ones include
run_dir: main dir for the experiment
checkpoints: models
logs: logging and debugging
figs: figures
datasets: mock datasets generated for training
'''
@dataclass
class ExperimentDirs:
	run_dir: Path
	checkpoints: Path
	logs: Path
	figs: Path
	artifacts: Path
	datasets: Union[Path, None] = None

	@staticmethod
	def create(
		cfg: Dict[str, Any],
		cfg_path: Union[str, Path],
		create_dataset_dir: bool = False
	) -> "ExperimentDirs":
		
		# base dir
		base = Path(cfg.get('base_dir', '../runs'))
		base.mkdir(parents=True, exist_ok=True)

		# create experiment id
		explicit_run_dir_name = cfg.get('run_dir_name', None)

		if explicit_run_dir_name:
			run_id = str(explicit_run_dir_name)
		elif cfg['use_dummy_dir']:
			run_id = 'test'
		else:
			ts = datetime.now().strftime("%Y%m%d-%H%M%S")
			name = cfg['run_id_suffix']
			run_id = f'{ts}{("-" + name) if name else ""}'
		
		# create run directory to save everything in
		run_dir = base / run_id

		# Just in case run dir already exists
		# for benchmarked runs we want determinitsic names
		if (not cfg['use_dummy_dir']) and (not explicit_run_dir_name):
			suffix = 1
			while run_dir.exists():
				run_dir = base / f"{run_id}-{suffix:02d}"
				suffix += 1

		# create sub directories
		ckpts = run_dir / 'checkpoints'
		logs  = run_dir / 'logs'
		figs  = run_dir / "figs"
		artifacts  = run_dir / "artifacts"
		for p in (ckpts, logs, figs, artifacts):
			p.mkdir(parents=True, exist_ok=True)

		# if in vp gen mode, need to create data dir
		if create_dataset_dir:
			dataset_base_raw = cfg.get('dataset_base_dir', '../datasets')
			dataset_base = Path(dataset_base_raw)

			benchmark_name = cfg.get('benchmark_meta', {}).get('benchmark_name', None)
			if benchmark_name:
				dataset_base = dataset_base / benchmark_name

			dataset_base.mkdir(exist_ok = True, parents = True)

			datasets = dataset_base / run_id
			datasets.mkdir(exist_ok = True, parents = True)
		else:
			datasets = None

		# save config
		cfg_out = run_dir / 'config.yaml'
		with cfg_out.open('w') as f:
			yaml.safe_dump(cfg, f, sort_keys=True)

		# Copy original, just in case the loaded one in changes significantly
		cfg_path = Path(cfg_path)
		try:
			if cfg_path.exists():
				shutil.copyfile(cfg_path, run_dir / 'config_original.yaml')
		except Exception as e:
			logging.getLogger('coroNeRF.experiments').debug(f'could not copy original config: {e}')

		# return expdir object
		return ExperimentDirs(run_dir, ckpts, logs, figs, artifacts, datasets)
		
	def setup_logging(self, verbosity: int = 0, 
				   			filename: str = 'train.log',
							enable_file: bool = True) -> None:
		'''
		enable_file: if we want to save the log
		'''
		# define basic level parameters
		level = VERBOSITY_TO_LEVEL.get(verbosity, logging.WARNING)
		fmt = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
		datefmt = '%H:%M:%S'

		# clear old handlers
		root = logging.getLogger()
		if root.handlers:
			for h in root.handlers[:]:
				root.removeHandler(h)

		# add console handler
		ch = logging.StreamHandler()
		ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
		root.addHandler(ch)

		# add file handler
		if enable_file:
			log_path = self.logs / "train.log"
			fh = logging.FileHandler(log_path, mode='a')   # append mode
			fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
			root.addHandler(fh)

		# set default level
		root.setLevel(level)

		# add title to log
		logging.getLogger('coroNeRF').info(f'Logging initialized at level={logging.getLevelName(level)} | run_dir={self.run_dir}')

		# suppress common libraries
		logging.getLogger('matplotlib').setLevel(logging.WARNING)
		logging.getLogger('PIL').setLevel(logging.WARNING)
		logging.getLogger('pycelp').setLevel(logging.WARNING)

	def write_text(self, text: str, name: str, subdir: str = 'logs') -> Path:
		'''
		generic helper to write text to a specific file in a specific folder
		'''
		if subdir != 'logs':
			raise ValueError("Only 'logs' is available. Enable figs/artifacts if needed.")
		
		target_dir = getattr(self, subdir)
		target_path = target_dir / name
		target_path.write_text(text)
		logging.getLogger('coroNeRF.experiments').debug(f'Wrote text file: {target_path}')
		return target_path

class DotDict(dict):
	"""dot-access dict: cfg.x.y instead of cfg['x']['y']"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

def save_pickle(obj, path):
	with open(path, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
	with open(path, 'rb') as f:
		return pickle.load(f)

#################################
### logging
#################################
@contextlib.contextmanager
def spherical_debug(field):
	old_level = field.logger.level
	field.logger.setLevel(logging.DEBUG)
	field.debug_every = 1
	try:
		yield
	finally:
		field.logger.setLevel(old_level)
		field.debug_every = 0

@contextlib.contextmanager
def regular_debug(field):
	old_level = field.logger.level
	field.logger.setLevel(logging.DEBUG)
	try:
		yield
	finally:
		field.logger.setLevel(old_level)

@contextlib.contextmanager
def renderer_debug(renderer):
	old_level = renderer.logger.level
	renderer.logger.setLevel(logging.DEBUG)
	renderer.debug_every = 1
	try:
		yield
	finally:
		renderer.logger.setLevel(old_level)
		renderer.debug_every = 0

@contextlib.contextmanager
def preserve_root_logging():
	"""
	Preserve root logger level + handlers across a block.
	For CHIANT/pycelp calls
	"""
	root = logging.getLogger()
	old_level = root.level
	old_handlers = list(root.handlers)
	try:
		yield
	finally:
		root.handlers = old_handlers
		root.setLevel(old_level)



