###############################################
## small unit test for density hash MLP
###############################################

import logging 
import torch

from ...util.torch import get_torch_dtype
from ..density_hash_mlp import DensityHashMLP

def test_density_hash_mlp(pipeline):
    '''
    quick unit test for hash grid sanity
    '''
    cfg = pipeline.cfg

    lg = logging.getLogger('coroNeRF.test_density_hash_mlp')
    dtype = get_torch_dtype(cfg.model['dtype'])

    x = torch.tensor([
        [0.1 * 1/30, 0.1 * 1/30, 0.1 * 1/30],
        [0.2 * 1/30, -0.1 * 1/30, 0.05 * 1/30],
    ], dtype = dtype)

    lg.debug(f'x_AABB: {x}, dtype: {x.dtype}')

    # instantiate model
    model = DensityHashMLP(
        **cfg.DensityHashMLP_kwargs,
        dtype = dtype,
    )

    # single forward pass
    ne_out = model(x)
    lg.debug(f'log10(ne) out: {ne_out}, shape={ne_out.shape}, dtype={ne_out.dtype}')

    assert ne_out.shape == (2,), f'Expected output shape (2,), got {tuple(ne_out.shape)}'
    assert torch.isfinite(ne_out).all(), 'DensityHashMLP produced non-finite outputs'

    # gradient smoke test
    loss = ne_out.mean()
    loss.backward()

    num_grads = sum(p.grad is not None for p in model.parameters())
    lg.debug(f'num params with grad = {num_grads}')
    assert num_grads > 0, 'no gradients flowed through DensityHashMLP'
    








