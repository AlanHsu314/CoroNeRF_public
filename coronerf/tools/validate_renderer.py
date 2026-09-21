#############################################################
### validatioon scripts for renderer
#############################################################

import logging
import torch

from ..experiments.dirs import renderer_debug
from .debug_renders import render_view_debug

def test_renderer(pipeline):
    lg = logging.getLogger('coroNeRF.test_renderer')
    single_ray_test = True
    gt_ne_test = pipeline.renderer.use_gt_ne

    with renderer_debug(pipeline.renderer):
        if single_ray_test:
            lg.debug('single ray test...')

            rays_o = torch.tensor([[1.5, 0, 0], [2, 2, 2]], dtype = pipeline.renderer.dtype, device = pipeline.renderer.device)
            rays_d = torch.tensor([[-1, 0, 0], [-0.5, -0.5, -0.5]], dtype = pipeline.renderer.dtype, device = pipeline.renderer.device)

            pipeline.renderer.forward(rays_o, rays_d)
        if gt_ne_test: # means we are testing using gt_ne
            lg.debug('gt ne render test...')
            render_view_debug(pipeline, -1, vp_idx=0) # test with ne_gt


