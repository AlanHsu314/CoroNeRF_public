############################################################
### train script
############################################################

import argparse
import yaml

from coronerf.pipeline.run import run_vpgen

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--config",
		type=str,
		default=None,
		help="Configuration file to run from.",
	)
	args = parser.parse_args()

	cfg_path = CONFIGS_DIR / args.config
	with open(cfg_path, "r") as f:
		raw_cfg = yaml.safe_load(f)

	run_vpgen(raw_cfg, cfg_path)
	
if __name__ == '__main__':
	main()



