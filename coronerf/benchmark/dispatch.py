###########################################################
### dispatcher, based on config.mode
###########################################################

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from ..pipeline.run import setup, prepare_vpgen_dataset_root
from ..pipeline.builders.data import load_data
from ..pipeline.builders.fields import build_fields
from ..pipeline.builders.model import build_model
from ..pipeline.builders.renderer import build_renderer

from ..data.vpgen import generate_psi_viewpoints, preprocess_viewpoints
from ..experiments.wandb import wandb_log_dataset_artifact
from ..train.loops import train_loop

def run_resolved_config(raw_cfg: dict, cfg_path: Union[Path, str] = '<resolved>'):
    '''
    dispatcher, based on mode
    '''
    logger = logging.getLogger('coroNeRF.benchmark.dispatch')
    mode = raw_cfg.get('mode', 'train')

    state, _runtime = setup(raw_cfg = raw_cfg, cfg_path = cfg_path, mode = mode)

    if mode in ['train', 'resume']:
        # build pipeline
        load_data(state, **state.cfg.load_data_kwargs)
        build_fields(state)
        build_model(state)
        build_renderer(state)


        # train
        logger.info(f"Mode={'train' if mode == 'train' else 'resume'}: training")
        train_loop(state, resume = (mode == 'resume'))
    elif mode == 'vpgen':
        # mirrors pipeline.run.run_vpgen
        prepare_vpgen_dataset_root(state)

        '''
        still sets up model in vpgen, which is not needed
        but renderer is needed, but renderer needs a model
        '''
        build_fields(state)
        build_model(state)
        build_renderer(state)

        # main
        logger.info("Mode='vpgen': generating new viewpoints and exiting.")
        generate_psi_viewpoints(state)
        preprocess_viewpoints(state)

        # wandb
        if (not state.no_save) and (state.wb_run is not None):
            ds_name = state.cfg.get('wandb', {}).get('dataset_name', f'dataset-{state.dirs.run_dir.name}')
            wandb_log_dataset_artifact(state.wb_run, state.dirs.datasets, ds_name)
    else:
        raise NotImplementedError(f'Benchmark dispatch does not support mode = {mode}')
    
    return state






































