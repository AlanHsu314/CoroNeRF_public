########################################################
## testing positional (and eventually angular) encoding
########################################################

import logging
import torch

from ..encoding import PosEnc
from ...util.torch import get_torch_dtype

def test_posenc(pipeline):
    cfg = pipeline.cfg

    lg = logging.getLogger('coroNeRF.test_posenc')
    dtype = get_torch_dtype(cfg.model['dtype'])
    single_point_test = True

    if single_point_test:
        lg.debug('single point test...')
        x = torch.tensor([[0.4, 0.5, 0.6]], dtype = dtype)
        lg.debug(f'in x shape: {x.shape}, dtype: {x.dtype}')

        # encode
        pe = PosEnc(in_dim = 3, L = 6, dtype = dtype, strict_types = cfg.model['strict_types'])
        x_out = pe.forward(x)
        lg.debug(f'out x shape = {x_out.shape}, dtype: {x_out.dtype}')