#####################################################################
### runtime detection (running on different machines)
#####################################################################

import os, sys, platform
import logging
import random
import numpy as np
import torch

# detect operating system
def detect_runtime():
	'''
	get metadata for running experiment
	'''
	is_slurm = 'SLURM_JOB_ID' in os.environ
	is_linux = sys.platform.startswith('linux')
	is_windows = sys.platform.startswith('win')
	is_cluster = is_linux and is_slurm # only run on slurm jobs

	return {
		'is_linux': is_linux,
		'is_windows': is_windows,
		'is_slurm': is_slurm,
		'is_cluster': is_cluster,
		'hostname': platform.node()
	}


# chatGPT function
def set_global_seed(seed: int) -> None:
	logger = logging.getLogger("coroNeRF.seed")

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	logger.info(f"Global seed set to {seed}")

	# Enforce deterministic algorithms where supported
	# try:
	# 	torch.use_deterministic_algorithms(True)
	# except Exception as e:
	# 	logging.getLogger("coroNeRF.seed").warning(
	# 		f"torch.use_deterministic_algorithms(True) failed: {e}"
	# 	)