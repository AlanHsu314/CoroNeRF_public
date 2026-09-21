"""
.. include:: ../README.md
"""
# make njit a harmless dectorator
try:
    import numba
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap
else:
    # override njit to be a no-op decorator
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap

from .ion import Ion

import numpy as np 

class vanVleck_angles:
    def __init__(self):  
        self.rad = np.arccos(1./np.sqrt(3.))
        self.deg = np.rad2deg(self.rad)
    
vanVleck = vanVleck_angles()

__all__ = ['Ion']
