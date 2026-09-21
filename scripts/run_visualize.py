############################################################
### visualize script
############################################################

import argparse
import yaml
import logging

from coronerf.pipeline.run import run_visualize

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
#RUNS_BENCHMARKS_DIR = REPO_ROOT / "runs_benchmarks"

def main(args):
	logging.basicConfig(level=logging.INFO)
	
	cfg_path = CONFIGS_DIR / args.config

	if cfg_path is not None:
		with open(cfg_path, 'r') as f:
			raw_cfg = yaml.safe_load(f)
	else:
		raw_cfg = {}

	raw_cfg['mode'] = 'visualize'
	#raw_cfg['resume_run_dir'] = RUNS_BENCHMARKS_DIR / raw_cfg['resume_run_dir']

	if 'visualize' not in raw_cfg:
		raw_cfg['visualize'] = {}

	if args.ckpt is not None:
		raw_cfg['visualize']['checkpoint'] = args.ckpt

	run_visualize(raw_cfg = raw_cfg, cfg_path = cfg_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#parser.add_argument('--run_dir', type=str, required=True, help='path to the specific finished run')
	parser.add_argument('--config', type=str, required=True, help='yaml with run dir and visualize block')
	parser.add_argument('--ckpt', type=str, default=None, help='optional explicit checkpoint path')
	args = parser.parse_args()

	main(args)

