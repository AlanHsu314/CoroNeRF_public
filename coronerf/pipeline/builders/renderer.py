#######################################################
### build renderers
#######################################################

import logging

from ...render.emissivity import EmissivityRenderer
from ...tools.validate_renderer import test_renderer

def build_renderer(state):
    logger = logging.getLogger('coroNeRF.setup_renderer')

    # create instance of renderer
    # # override aabb scale if given scene aabb scale
    renderer_cfg = dict(state.cfg.renderer) # avoid mutating config
    # renderer_cfg['aabb_scale'] = state.aabb_scale
    state.renderer = EmissivityRenderer(model = state.model,
                                        temp_field = state.temp_field,
                                        ccoef_field = state.ccoef_field,
                                        cfg = renderer_cfg,
                                        ne_field = state.ne_field,
                                        shared_context= state.shared)

    # unit testing emissivity renderer
    if state.cfg.renderer.get('test', False):
        logger.info('Testing Emissivity Renderer Class')
        test_renderer(state)




