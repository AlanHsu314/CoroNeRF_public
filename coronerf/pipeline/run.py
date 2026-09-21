###########################################################
### main dispatch script
###########################################################

# organization
from __future__ import annotations
import logging
from pathlib import Path
import shutil
import os
import json

# own module
from .. import ExperimentDirs, DotDict                                                                     # experiments/dirs.py               
from .. import detect_runtime, set_global_seed                                                             # experiments/runtime.py
from ..experiments.wandb import wandb_enabled, wandb_init_run, wandb_log_dataset_artifact

from ..train.loops import train_loop, fit_ne_loop

from ..data.vpgen import regenerate_psi_viewpoints, generate_psi_viewpoints, preprocess_viewpoints

from .builders.data import load_data
from .builders.fields import build_fields
from .builders.model import build_model
from .builders.renderer import build_renderer

from .state import PipelineState

# eval resolving paths
from ..experiments.resolve_paths import resolve_cfg_paths

# benchmark stuff
from ..benchmark.config import normalize_config

# visualization 
import copy
import yaml

from ..eval.run_eval import load_ckpt
from ..visualization.explorer import CoroNeRFExplorer

# helpers and setup for run scripts

def prepare_effective_cfg(raw_cfg, cfg_path, mode: str) -> dict:
    '''
    prepares effective config to be used in pipeline everywhere
    '''

    # normalize cfg
    cfg = normalize_config(raw_cfg)

    # add mode
    cfg['mode'] = mode

    # resolve only input/resource paths relative to source config location
    cfg_root = Path(cfg_path).resolve().parent if cfg_path is not None else Path.cwd()
    cfg = resolve_cfg_paths(cfg, root = cfg_root)

    # fix base dir relative to the repo root
    base_dir = cfg.get("base_dir", None)
    if base_dir is not None:
        base_dir = Path(base_dir)

        if not base_dir.is_absolute():
            # repo root = parent of configs/
            base_dir = (cfg_root / base_dir).resolve()

        cfg["base_dir"] = str(base_dir)

    return cfg

def prepare_vpgen_dataset_root(state):
    '''
    resolves dataset paths
    '''
    logger = logging.getLogger('coroNeRF.prepare_vpgen_dataset_root')

    lut_src_dir = state.cfg.get('LUT_dir', None)
    if lut_src_dir is None:
        raise ValueError("LUT_dir must be set for mode = 'vpgen'")
    lut_src_dir = Path(lut_src_dir)

    if state.no_save:
        logger.info('io.no_save = True: using LUT_dir directly as dataset root')
        return
    
    for name in ['ccoef_LUT.npy', 'grid_vals.npz', 'psi_fields.npz']:
        src = lut_src_dir / name
        dst = state.dataset_dir / name
        shutil.copy(src, dst)

    logger.info(f'copied PSI/LUT resources into dataset root: {state.dataset_dir}')

def setup(raw_cfg, cfg_path, mode: str):
    logger = logging.getLogger('coroNeRF.run.setup')
    
    # config object (for easy attr access)
    raw_cfg = prepare_effective_cfg(raw_cfg, cfg_path, mode)
    cfg = DotDict(raw_cfg)

    logger.debug(f"shared config: {cfg.shared}")

    # do we save
    no_save = bool(cfg.get('io', {}).get('no_save', False))

    # add mode, before this was in raw_cfg but now wer delegate responsibility to run script
    #cfg.mode = mode # moved to prepare_effective_cfg()

    # case on no_save
    if no_save:
        # create dummy directories (guard all saving elsewhere)
        dirs = ExperimentDirs(
            run_dir = Path('.'),
            checkpoints = Path('./__nosave__/checkpoints'),
            logs = Path('./__nosave__/logs'),
            figs = Path('./__nosave__/figs'),
            artifacts = Path('./__nosave__/artifacts'),
        )

        # specific modes that should not allow no_save
        if mode == 'resume':
            raise ValueError('io.no_save = True is not compatible with mode = "resume" (resume requires checkpoints on disk).')
    else:
        # dirs object
        if mode == 'resume':
            # when resuming, need to give the run directory
            run_dir = Path(raw_cfg["resume_run_dir"])
            dirs = ExperimentDirs(
                run_dir = run_dir,
                checkpoints = run_dir / "checkpoints",
                logs = run_dir / "logs",
                figs = run_dir / "figs",
                artifacts = run_dir / "artifacts",
            )
            # reuse the directories, do not create new ones
        elif mode == 'vpgen':
            dirs = ExperimentDirs.create(cfg=raw_cfg, cfg_path=cfg_path, create_dataset_dir = True)
        else:
            dirs = ExperimentDirs.create(cfg=raw_cfg, cfg_path=cfg_path)
        
    # seed
    seed = cfg.get("seed", 0xC0FFEE)
    set_global_seed(int(seed))
    
    # logging
    dirs.setup_logging(verbosity = cfg.get('verbosity', 1), enable_file = (not no_save))
    logger = logging.getLogger('coroNeRF.setup')
    
    if no_save:
        logger.info('io.no_save = True, proceeding without saving any directories')
    
    # runtime dict
    runtime = detect_runtime()

    force = cfg.get('runtime', {}).get('force_cluster', None)
    if force is not None:
        runtime['is_cluster'] = bool(force)

    if runtime['is_windows']:
        if runtime['is_linux']: # probably wont ever happen
            raise ValueError('operating system cannot be both windows and linux')
        if runtime['is_slurm']: # weird case but safety guard
            logger.warning("SLURM_JOB_ID present on Windows; ignoring cluster logic")
            runtime['is_cluster'] = False
        
        logger.info('experiment running on windows, setting os.environ HOME --> USERPROFILE')
        os.environ['HOME'] = os.environ['USERPROFILE'] # for chianti in pycelp
    elif runtime['is_linux']:
        logger.info('experiment running on linux')
    
    runtime_out = dict(runtime)

    logger.info(
        f"Runtime detected: "
        f"os={'windows' if runtime['is_windows'] else 'linux'}, "
        f"cluster={runtime['is_cluster']}, "
        f"hostname={runtime['hostname']}"
    )

    if runtime['is_cluster']:
        runtime_out.update({
            'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID'),
            'SLURM_JOB_NAME': os.environ.get('SLURM_JOB_NAME'),
            'SLURM_NODELIST': os.environ.get('SLURM_NODELIST'),
            'SLURM_GPUS': os.environ.get('SLURM_GPUS'),
            'hostname': runtime['hostname'],
        })

    # write runtime json
    if not no_save:
        with open(dirs.run_dir / 'runtime.json', 'w') as f:
            json.dump(runtime_out, f, indent=2)

    # w&B
    wb_run = None
    if (not no_save) and wandb_enabled(cfg):
        wb_run = wandb_init_run(cfg, dirs, runtime_out, mode)
        logger.info(f'W&B enabled: {wb_run.project}/{wb_run.name}')

    # create pipeline state
    state = PipelineState(cfg = cfg, dirs = dirs, wb_run = wb_run)

    return state, runtime_out

# visualization helpers, courtesy of chatgpt so we dont need to wire into setup()

def _deep_update(base: dict, override: dict) -> dict:
	'''
	recursive dict update
	'''
	out = copy.deepcopy(base)
	for k, v in override.items():
		if isinstance(v, dict) and isinstance(out.get(k, None), dict):
			out[k] = _deep_update(out[k], v)
		else:
			out[k] = copy.deepcopy(v)
	return out

def build_visualize_state(raw_cfg: dict, cfg_path) -> tuple[PipelineState, Path, dict]:
	'''
	Load the saved config from a finished run, then overlay the incoming
	visualize config and a few runtime options.

	This avoids creating a new experiment directory and keeps visualization
	tied to the selected finished run.
	'''
	logger = logging.getLogger('coroNeRF.run.build_visualize_state')

	run_dir_raw = raw_cfg.get('resume_run_dir', None)
	if run_dir_raw is None or str(run_dir_raw).strip() == '':
		raise ValueError("mode='visualize' requires resume_run_dir to point to an existing finished run")

	run_dir = Path(run_dir_raw).resolve()
	cfg_run_path = run_dir / 'config.yaml'
	if not cfg_run_path.exists():
		raise FileNotFoundError(f'Could not find saved run config: {cfg_run_path}')

	with open(cfg_run_path, 'r') as f:
		run_cfg = yaml.safe_load(f)

	run_cfg = normalize_config(run_cfg)

	# only overlay a small set of fields from the incoming config
	override_cfg = {
		'mode': 'visualize',
		'resume_run_dir': str(run_dir),
		'device': raw_cfg.get('device', run_cfg.get('device', 'cuda')),
		'verbosity': raw_cfg.get('verbosity', run_cfg.get('verbosity', 1)),
		'debug': raw_cfg.get('debug', run_cfg.get('debug', False)),
		'visualize': raw_cfg.get('visualize', {}),
		'wandb': {
			'enabled': False,
		},
	}

	merged_cfg = _deep_update(run_cfg, override_cfg)
	cfg = DotDict(merged_cfg)

	# console logging only; do not overwrite train.log / runtime.json / config.yaml
	root = logging.getLogger()
	if root.handlers:
		for h in root.handlers[:]:
			root.removeHandler(h)

	ch = logging.StreamHandler()
	fmt = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
	datefmt = '%H:%M:%S'
	ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
	root.addHandler(ch)
	root.setLevel(logging.INFO if cfg.get('verbosity', 1) >= 1 else logging.WARNING)

	logger.info(f'building visualize state from run_dir={run_dir}')

	dirs = ExperimentDirs(
		run_dir = run_dir,
		checkpoints = run_dir / 'checkpoints',
		logs = run_dir / 'logs',
		figs = run_dir / 'figs',
		artifacts = run_dir / 'artifacts',
	)

	state = PipelineState(cfg = cfg, dirs = dirs, wb_run = None)

	# no dataset loading needed for post-training visualization
	build_fields(state)
	build_model(state)
	build_renderer(state)

	viz_cfg = cfg.get('visualize', {})
	ckpt_cfg = viz_cfg.get('checkpoint', 'latest')

	if ckpt_cfg == 'latest':
		ckpts = sorted((run_dir / 'checkpoints').glob('ckpt_step*.pt'))
		if not ckpts:
			raise FileNotFoundError(f'No checkpoints found in {run_dir / "checkpoints"}')
		ckpt_path = ckpts[-1]
	else:
		ckpt_path = Path(ckpt_cfg).resolve()

	load_ckpt(state, ckpt_path)

	return state, ckpt_path, viz_cfg

# main run scripts that builds, trains, evals, exits

def run_train(raw_cfg, cfg_path):
    logger = logging.getLogger('coroNeRF.run_train')

    state, runtime_out = setup(raw_cfg, cfg_path, mode = 'train')

    load_data(state, **state.cfg.load_data_kwargs)

    build_fields(state)
    build_model(state)
    build_renderer(state)

    # main 
    logger.info("Mode='train': training")
    train_loop(state)

    # exiting
    teardown(state)
    
def run_resume(raw_cfg, cfg_path):
    logger = logging.getLogger('coroNeRF.run_resume')

    state, runtime_out = setup(raw_cfg, cfg_path, mode = 'resume')

    load_data(state, **state.cfg.load_data_kwargs)

    build_fields(state)
    build_model(state)
    build_renderer(state)

    # main
    logger.info("Mode='resume': resuming training")
    train_loop(state, resume = True)

    # exiting
    teardown(state)
    
def run_fit_ne(raw_cfg, cfg_path):
    logger = logging.getLogger('coroNeRF.run_fit_ne')

    state, runtime_out = setup(raw_cfg, cfg_path, mode = 'fit_ne')

    build_fields(state)
    build_model(state)
    build_renderer(state)

    # main
    logger.info("Mode='fit_ne': fitting DensityMLP directly with PSI ne_field data")
    fit_ne_loop(state)

    # exiting
    teardown(state)
    
def run_regen(raw_cfg, cfg_path):
    logger = logging.getLogger('coroNeRF.run_regen')

    state, runtime_out = setup(raw_cfg, cfg_path, mode = 'regen')

    load_data(state, **state.cfg.load_data_kwargs)

    build_fields(state)
    build_model(state)
    build_renderer(state)

    # main
    logger.info("Mode='regen': regenerating viewpoints_gt and exiting.")
    # make sure renderer uses ground-truth ne
    state.renderer.use_gt_ne = True
    regenerate_psi_viewpoints(state)

    # exiting
    teardown(state)
    
def run_vpgen(raw_cfg, cfg_path):
    logger = logging.getLogger('coroNeRF.run_vpgen')

    state, runtime_out = setup(raw_cfg, cfg_path, mode = 'vpgen')
    
    # copy over files to new dataset directory, needed for aux fields later on
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

    if (not getattr(state, "no_save", False)) and state.wb_run is not None:
        ds_name = state.cfg.get('wandb', {}).get('dataset_name', f'dataset-{state.dirs.run_dir.name}')
        wandb_log_dataset_artifact(state.wb_run, state.dirs.datasets, ds_name)

    # exiting
    teardown(state)

def run_visualize(raw_cfg, cfg_path):
	logger = logging.getLogger('coroNeRF.run_visualize')

	state, ckpt_path, viz_cfg = build_visualize_state(raw_cfg, cfg_path)

	logger.info(f"Mode='visualize': loading checkpoint {ckpt_path}")

	explorer = CoroNeRFExplorer(
		state = state,
		ckpt_path = ckpt_path,
		out_dir_name = viz_cfg.get('out_dir_name', 'visualize'),
        viz_cfg = viz_cfg,
	)

	explorer.run_from_config(viz_cfg)

	teardown(state)

def teardown(state):
    # ending procedures

    # w&B
    if state.wb_run is not None:
        state.wb_run.finish()

'''
[.] In the future, if we want to merge pycelp-style LOS generation with this pipeline, use the forward model class below
[.] just note: pycelp uses numba, which requires lower than numpy 2.0, while most of my torch env uses torch versions that build with numpy > 2.0
[.] so, we leave this out for now

##############################################
### psi and pycelp classes
##############################################

@dataclass
class LineModel: # wrapper for each emussion line
    name: str
    wavelength: float
    eline: object         # pycelp emission line object
    ccoef_interp: object  # rgi instance
    align_interp: object  # rgi instance

class PycelpForward:
    def __init__(self, ion_kwargs: dict, psi_kwargs: dict, rgi_kwargs: dict): 
        # based on initialize_psi_model of viewpoint_pipeline_V1.py
        self.logger = logging.getLogger('coroNeRF.PycelpForward')
        self.logger.debug('initializing PycelpForward model')
        
        # create pycelp ion object
        self.ion_kwargs = ion_kwargs
        self.ion = pycelp.Ion(ion_name = self.ion_kwargs['ion_name'],
                              nlevels=self.ion_kwargs['n_levels'])
        
        # load psi cube
        self.psi_kwargs = psi_kwargs
        self.psi_cube_dir = f'../data/corona/CR{self.psi_kwargs.get("CR_number", "2149")}/'
        self.psi_cube = psi.Model(self.psi_cube_dir)

        # grid points
        lons, colats, rs = self.psi_cube.lons,self.psi_cube.lats,self.psi_cube.rs # colats here are from 0 to pi!

        self.logger.debug(f'lons range: [{lons.min()}, {lons.max()}]')
        self.logger.debug(f'colat range: [{colats.min()}, {colats.max()}]')
        self.logger.debug(f'r range: [{rs.min()}, {rs.max()}]')

        points = (lons, colats, rs)

        # interpolator constructor
        self.rgi_kwargs = rgi_kwargs
        construct_rgi = lambda values: rgi(points = points, values = values, **self.rgi_kwargs)

        # load wavelengths
        wvls = self.ion_kwargs.get('wavelengths', None)
        if wvls is None:
            raise ValueError(f'must include list of wavelengths of emission lines')
        
        line_names = self.ion_kwargs.get('line_names', [])
        if len(line_names) and len(line_names) != len(wvls):
            raise ValueError('line_names, if given, must match wwavelengths length!')
        
        # create emission line objects
        self.lines: list[LineModel] = []

        for i, wvl in enumerate(wvls):
            self.logger.debug(f'loading pops for {self.ion_kwargs["ion_name"]} {wvl:.1f} A')

            eline = self.ion.get_emissionLine(wvl)
            pop_path = f'../data/pops/CR{self.psi_kwargs["CR_number"]}/{self.ion_kwargs["ion_name"]}_{str(wvl)}.npz'
            npzd = np.load(self.pop_dir)
            ccoef = npzd['ccoef']
            align = npzd['align']

            ccoef_interp = construct_rgi(ccoef)
            align_interp = construct_rgi(align)

            name = line_names[i] if i < len(line_names) else f'{self.ion_kwargs["ion_name"]} {wvl:.1f}'
            self.lines.append(LineModel(name = name,
                                           wavelength = wvl,
                                        eline = eline,
                                        ccoef_interp = ccoef_interp,
                                        align_interp = align_interp))
            
        self.logger.debug('created interpolators for emission lines')
            
        # create interps for non-line-dependent values
        self.interp_dict = dict()
        self.interp_dict['bx'] = construct_rgi(self.psi_cube.bx)
        self.interp_dict['by'] = construct_rgi(self.psi_cube.by)
        self.interp_dict['bz'] = construct_rgi(self.psi_cube.bz)
        self.interp_dict['temp'] = construct_rgi(self.psi_cube.temp)
        self.interp_dict['ne'] = construct_rgi(self.psi_cube.ne)

        self.logger.debug('created interpolators for psi field values')

'''