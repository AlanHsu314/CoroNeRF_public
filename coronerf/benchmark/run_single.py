###############################################################################
## wrap one resolved config around existing pipeline from pipeline/run.py
###############################################################################

from __future__ import annotations
from typing import Union

import logging
import traceback
from pathlib import Path

# builders
from ..pipeline.run import teardown

# benchmarking
from .config import save_json
from .summary import build_summary
from .dispatch import run_resolved_config
from ..experiments.wandb import wandb_update_summary
from ..eval.run_eval import run_evaluation

def run_single_experiment(raw_cfg: dict, cfg_path: Union[Path, str] = '<resolved>') -> dict:
    '''
    run exactly one fully-resolved config using the exisiting pipeline in run.py
    '''
    logger = logging.getLogger('coroNeRF.benchmark.run_single')
    state = None
    run_dir = None

    try:
        # setup pipeline state
        state = run_resolved_config(raw_cfg = raw_cfg, cfg_path = cfg_path)
        run_dir = state.dirs.run_dir
        mode = state.cfg.get('mode', 'train')

        # only training and resume needs final evaluation
        if mode in ['train', 'resume']: 
            # make sure eval file exists even if inline eval was disabled
            final_eval_path = run_dir / 'artifacts' / 'eval' / 'json' / 'eval_final_report.json'
            if not final_eval_path.exists() and not getattr(state, 'no_save', False): # grab latest ckpt
                latest_ckpt = sorted((run_dir / 'checkpoints').glob('ckpt_step*.pt'))
                if latest_ckpt:
                    run_evaluation(
                        post_training = True,
                        run_dir = run_dir,
                        ckpt_path = latest_ckpt[-1],
                    )

        # benchmark summary
        summary = build_summary(state = state, run_dir = run_dir, status = 'completed')
        save_json(summary, run_dir / 'summary.json')

        # wandb save, only summary stats
        if state.wb_run is not None:
            wandb_update_summary(state.wb_run, summary)

        return summary

    except Exception as e:
        error_message = ''.join(traceback.format_exception_only(type(e), e)).strip()
        logger.info(f'run failed to complete, with error_message {error_message}')

        if run_dir is not None and state is not None:
            summary = build_summary(
                state = state,
                run_dir = run_dir,
                status = 'failed',
                error_message = error_message,
            )
            save_json(summary, Path(run_dir) / 'summary.json')

            #wandb update
            if state.wb_run is not None:
                wandb_update_summary(state.wb_run, summary)
        else:
            summary = {
                'status': 'failed',
                'run_dir': None,
                'run_name': raw_cfg.get('benchmark_meta', {}).get('run_name'),
                'benchmark_name': raw_cfg.get('benchmark_meta', {}).get('benchmark_name'),
                'experiment_name': raw_cfg.get('benchmark_meta', {}).get('experiment_name'),
                'error_message': error_message,
            }

        return summary

    finally:
        if state is not None:
            teardown(state)
