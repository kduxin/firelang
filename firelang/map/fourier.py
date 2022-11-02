import torch
from .rect import (
    SmoothedRectMap,
)

__all__ = [
    'SmoothedRectFourier2DMap',
]

class SmoothedRectFourier2DMap(SmoothedRectMap):

    @property
    def grid(self):
        g = SmoothedRectMap.get_grid(self)
        assert g.ndim == 2

        g = torch.fft.fft2(g).real