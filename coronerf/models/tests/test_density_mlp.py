########################################################
## testing densty mlp
########################################################

import logging
import torch

from ...util.torch import get_torch_dtype
from ..density_mlp import DensityMLP


def test_mlp(pipeline):
    cfg = pipeline.cfg

    lg = logging.getLogger('coroNeRF.test_model_class')
    dtype = get_torch_dtype(cfg.model['dtype'])
    single_point_test = True

    if single_point_test:
        lg.debug('single point test...')
        x = torch.tensor([[0.1*1/30, 0.1*1/30, 0.1*1/30]], dtype = dtype)
        lg.debug(f'x_AABB: {x}, dtype: {x.dtype}')

        # feed through model
        model = DensityMLP(hidden_dim = 128,
                            num_hidden = 3,
                            L = 6,
                            in_dim = 3,
                            dtype = dtype,
                            strict_types = cfg.model['strict_types'],
                            use_sh_encoding= cfg.model.get('use_sh_encoding', False),
                            sh_degree = cfg.model.get('sh_degree', 2))
        
        ne_out = model(x)
        lg.debug(f'log10(ne) out: {ne_out}, dtype: {ne_out.dtype}') # model isnt trained, so don't expect meaningful value here