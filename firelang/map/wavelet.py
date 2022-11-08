from __future__ import annotations
from typing import List, Tuple, Iterable
from typing_extensions import Literal
import numpy as np
import torch
from torch import Tensor, dtype, device
from torch.nn import Parameter
import pywt
import ptwt
from ._grid import Grid
from .rect import (
    SmoothedRectMap,
)
from firelang.function import Functional
from firelang.utils.limits import parse_rect_limits

__all__ = [
    "SmoothedRectWavelet2DMap",
]


class Wavelet2DMap(Grid):
    def __init__(
        self,
        dim_sizes: List[int],
        wavelet: Literal["haar"] = "haar",
        level: int = 3,
        dtype: dtype = torch.float32,
        device: device = "cuda",
        shape: Tuple[int] = (1,),
    ):
        Grid.__init__(
            self,
            dim_sizes=dim_sizes,
            dtype=dtype,
            device=device,
            shape=shape,
        )
        self.register_extra_init_kwargs(
            wavelet=wavelet,
            level=level,
        )
        self._wavelet = wavelet
        self.wavelet = pywt.Wavelet(wavelet)
        self.level = level

    @property
    def gridvals(self) -> Tensor:
        assert self.ndim == 2
        g: Parameter = self._gridvals  # (size, n1, n2)

        g = g[:, None, :, :]  # (size, 1, n1, n2)
        sizes: Tensor = self.dim_sizes
        sizes = sizes.data.cpu().numpy().tolist()

        coeffs = []
        for _ in range(self.level - 1):
            h, w = (sizes[0] + 1) // 2, (sizes[1] + 1) // 2
            coeff1 = g[:, :, -h:, :w]
            coeff2 = g[:, :, -h:, -w:]
            coeff3 = g[:, :, :h, -w:]
            coeffs.append((coeff1, coeff2, coeff3))
            sizes = [h, w]

        coeffs.append(g[:, :, :h, :w])
        coeffs = list(reversed(coeffs))
        return ptwt.waverec2(coeffs, self.wavelet)  # (size, 1, n1, n2)


class SmoothedRectWavelet2DMap(SmoothedRectMap):
    def __init__(
        self,
        limits: Tuple[float, float] | List[Tuple[float, float]],
        grid_dim_sizes: List[int],
        rect_dim_sizes: int | List[int] = 3,
        rect_weight_decay: Literal["gauss", "exp"] = "gauss",
        bandwidth_mode: Literal["parameter", "constant"] = "parameter",
        bandwidth_lb: float = 0.3,
        wavelet: str = "haar",
        level: str = 3,
        dtype: dtype = torch.float32,
        device: device = "cuda",
        shape: Tuple[int] = (1,),
    ):
        Functional.__init__(self, locals())

        self.shape = shape
        size = int(np.prod(shape))
        self.ndim = len(grid_dim_sizes)
        self._grid = Wavelet2DMap(
            grid_dim_sizes,
            wavelet=wavelet,
            level=level,
            dtype=dtype,
            device=device,
            shape=shape,
        )

        def _sizes_to_tensor(sizes: int | List[int], ndim: int) -> Tensor:
            if not isinstance(sizes, Iterable):
                sizes = [sizes] * ndim
            sizes = torch.tensor(sizes, dtype=torch.long, device=device)
            return sizes

        self.grid_dim_sizes = _sizes_to_tensor(grid_dim_sizes, self.ndim)
        self.rect_dim_sizes = _sizes_to_tensor(rect_dim_sizes, self.ndim)
        self.limits = torch.tensor(
            parse_rect_limits(limits, self.ndim),
            dtype=dtype,
            device=device,
        )

        self.rect_weight_decay = rect_weight_decay
        self.bandwidth_mode = bandwidth_mode
        self.bandwidth_lb = bandwidth_lb
        if bandwidth_mode == "constant":
            self._bandwidth = bandwidth_lb * torch.ones(size, dtype=dtype)
        elif bandwidth_mode == "parameter":
            self._bandwidth = Parameter(torch.zeros(size, dtype=dtype))
        else:
            raise ValueError(bandwidth_mode)
