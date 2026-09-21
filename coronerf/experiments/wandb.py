#####################################################################
### weights and biases
#####################################################################

import logging
import wandb
from .dirs import ExperimentDirs
from pathlib import Path

# check if we are to use wandb
def wandb_enabled(cfg) -> bool:
	w = cfg.get('wandb', {})
	return bool(w.get('enabled', False))

def _merge_tags(*tag_lists):
	out = []
	seen = set()

	for tags in tag_lists:
		if not tags:
			continue
		for tag in tags:
			if tag is None:
				continue
			s = str(tag)
			if s not in seen:
				seen.add(s)
				out.append(s)
	return out

def wandb_update_summary(run, payload: dict):
	'''
	organizational helper
	'''
	if run is None:
		return
	
	for k, v in payload.items():
		if isinstance(v, (int, float, str, bool)) or v is None:
			run.summary[k] = v

# initialize the logging object
def wandb_init_run(cfg, dirs: ExperimentDirs, runtime: dict, mode: str):
	logger = logging.getLogger('coroNeRF.wandb_init_run')
	
	# configs
	w = cfg.get('wandb', {})
	benchmark_meta = cfg.get('benchmark_meta', {})

	# cfg mode
	configured_mode = w.get('mode', None)

	# respect config mode if given
	if configured_mode is None:
		effective_mode = 'online' if runtime.get('is_windows') else 'offline'
	else:
		effective_mode = configured_mode

	# guardrail warning
	if (effective_mode == 'online') and runtime.get('is_cluster', False):
		logger.warning('W&B mode = online ON cluster:  ensure compute nodes have internet access')
	
	# wandb group = benchmark name by default
	benchmark_name = benchmark_meta.get('benchmark_name')
	experiment_name = benchmark_meta.get('experiment_name')
	run_name = benchmark_meta.get('run_name', str(dirs.run_dir.name))

	configured_group = w.get('group', None)
	effective_group = configured_group if configured_group is not None else benchmark_name

	# wandb tags 
	base_tags = w.get('tags', None)
	benchmark_tags = []
	if benchmark_name is not None:
		benchmark_tags.append('benchmark')
		benchmark_tags.append(str(benchmark_name))
	if experiment_name is not None:
		benchmark_tags.append(str(experiment_name))
	model_name = cfg.get('model', {}).get('name')
	if model_name is not None:
		benchmark_tags.append(str(model_name))
	reconstruction_target = cfg.get('shared', {}).get('reconstruction_target')
	if reconstruction_target is not None:
		benchmark_tags.append(str(reconstruction_target))

	forward_backend = cfg.get('forward_backend')
	if forward_backend is not None:
		benchmark_tags.append(str(forward_backend))

	effective_tags = _merge_tags(base_tags, benchmark_tags)

	# create run obj
	run = wandb.init(project = w.get('project', 'coroNeRF'),             # top level bucket
				  	 entity = w.get('entity', None),                     # who owns project
					 group = effective_group,                            # logical grouping of runs
					 tags = effective_tags if effective_tags else None,  # free form tags for filtering    
					 job_type = mode,                                    # semantic label also for filtering
					 name = run_name,                                    # human-readable run name
					 dir = str(dirs.run_dir),                            # actual run dir folder
					 config = dict(cfg),                                 # immutable config
					 settings = wandb.Settings(start_method = 'thread'), # thread for HPC safety, avoids deadlocks and hangs
					 mode = effective_mode,)                             # online | offline | disabled
	
	logger.info(f'created W&B run')
	
	# update config with some extra info
	run.config.update(
		{
			'run_dir': str(dirs.run_dir.resolve()),
			'dataset_dir': str(dirs.datasets.resolve()) if dirs.datasets else None,
			'hostname': runtime.get('hostname'),
			'is_cluster': runtime.get('is_cluster'),
			'wandb_effective_group': effective_group,
			'wandb_effective_mode': effective_mode,
		}, 
		allow_val_change = True)
	
	# add sweep values to wandb log
	sweep_values = benchmark_meta.get('sweep_values', {})
	if sweep_values:
		run.config.update(
			{f'sweep.{k}': v for k, v in sweep_values.items()},
			allow_val_change = True,
		)
	
	return run

# helper to log dataset artifact
def wandb_log_dataset_artifact(run, dataset_dir: Path, name: str):
	logger = logging.getLogger('coroNeRF.wandb_log_dataset_artifact')

	art = wandb.Artifact(name = name, type = 'dataset')
	art.add_dir(str(dataset_dir), name = 'dataset_dir')
	run.log_artifact(art) # logs and versions the dataset creation

	logger.info(f'W&B artifact log: dataset {str(dataset_dir)}')

# similar helper to log a checkpoint
def wandb_log_checkpoint_artifact(run, ckpt_path: Path, name: str):
	logger = logging.getLogger('coroNeRF.wandb_log_checkpoint_artifact')

	art = wandb.Artifact(name = name, type = 'model')
	art.add_file(str(ckpt_path), name = ckpt_path.name)
	run.log_artifact(art)

	logger.info(f'W&B artifact log: ckpt {str(ckpt_path)}')




