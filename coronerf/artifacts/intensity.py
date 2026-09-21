######################################################
### intensity plots
######################################################

from __future__ import annotations
from typing import Union

import torch

@torch.no_grad()
def render_rays_chunked(state, rays_o, rays_d, chunk_rays: int, **renderer_kwargs):
    '''
    render rays by chunks
    '''
    device = state.renderer.device
    dtype = state.renderer.dtype

    rays_o = rays_o.to(device=device, dtype=dtype)
    rays_d = rays_d.to(device=device, dtype=dtype)

    outs = []
    N = rays_o.shape[0]
    for s in range(0, N, chunk_rays):
        e = min(s + chunk_rays, N)
        out = state.renderer.render(
            rays_o[s:e],
            rays_d[s:e],
            backend=state.forward_backend,
            **renderer_kwargs,
        )
        outs.append(out.detach().cpu())

    return torch.cat(outs, dim=0)








