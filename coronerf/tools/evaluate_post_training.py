###############################################################
### EVALUATION SCRIPT for CoroNeRF models
###############################################################


################################
### USAGE
################################
'''
evaluation script

python -m coronerf.tools.evaluate_post_training --run_dir runs/20251216-191710_density_v1


'''

from __future__ import annotations

# organization
import argparse
import logging
from pathlib import Path

from ..eval.run_eval import run_evaluation

##########################
# main functions
##########################

def main(args):
	# logging
	logging.basicConfig(level=logging.DEBUG)
	
	# directories
	run_dir = Path(args.run_dir)
	
	# find ckpt
	if args.ckpt is None:
		ckpts = sorted((run_dir / 'checkpoints').glob('ckpt_step*.pt'))
		if not ckpts:
			raise FileNotFoundError(f'no checkpoints found in {run_dir}/checkpoints')
		ckpt_path = ckpts[-1]
	else:
		ckpt_path = Path(args.ckpt)

	# run evaluation scripts
	run_evaluation(post_training = True,
				   state = None,
				   run_dir = run_dir,
				   step = None,
				   ckpt_path = ckpt_path,
				   )

if __name__ == '__main__':
	# load args
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_dir', type = str, required = True, help = 'path to the specific run experiment')
	parser.add_argument('--ckpt', type = str, default = None, help = 'path to skpt .pt, defaults to latest ckpt' )
	args = parser.parse_args()

	main(args = args)



