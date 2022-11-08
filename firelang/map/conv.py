from __future__ import annotations
from typing import List, Tuple, Iterable
from typing_extensions import Literal
import numpy as np
import torch
from torch import dtype, device, Tensor
from torch.nn import Module, Parameter, ModuleList

from firelang.function import Functional
from firelang.utils.limits import parse_rect_limits
from ._grid import Grid
from .rect import (
    SmoothedRectMap,
)

__all__ = [
    "SmoothedRectConv2DMap",
]


class Conv2DGrid(Grid):
    def __init__(
        self,
        dim_sizes: List[int],
        conv_size: int | List[int] = 3,
        conv_chans: int = 1,
        conv_layers: int = 1,
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
            conv_size=conv_size,
            conv_chans=conv_chans,
            conv_layers=conv_layers,
        )
        self.conv_size = conv_size
        self.conv_chans = conv_chans
        self.conv_layers = conv_layers

        self.conv = ModuleList()
        for l in range(conv_layers):
            in_chans = 1 if l == 0 else conv_chans
            self.conv.append(
                torch.nn.Conv2d(
                    in_chans,
                    conv_chans,
                    conv_size,
                    padding="same",
                    device=device,
                )
            )
            self.unsliceable_params.add(f"conv.{l}.weight")
            self.unsliceable_params.add(f"conv.{l}.bias")

    @property
    def gridvals(self) -> Tensor:
        assert self.ndim == 2
        g: Parameter = self._gridvals  # (size, n1, n2)

        g = g[:, None, :, :]  # (size, shape, 1, n1, n2)
        for layer in self.conv:
            g = layer(g)  # (size, channels, n1, n2)
        g = g.sum(dim=1)
        return g


class SmoothedRectConv2DMap(SmoothedRectMap):
    def __init__(
        self,
        limits: Tuple[float, float] | List[Tuple[float, float]],
        grid_dim_sizes: List[int],
        rect_dim_sizes: int | List[int] = 3,
        rect_weight_decay: Literal["gauss", "exp"] = "gauss",
        bandwidth_mode: Literal["parameter", "constant"] = "parameter",
        bandwidth_lb: float = 0.3,
        conv_size: int | List[int] = 3,
        conv_chans: int = 1,
        conv_layers: int = 1,
        dtype: dtype = torch.float32,
        device: device = "cuda",
        shape: Tuple[int] = (1,),
    ):
        Functional.__init__(self, locals())

        self.shape = shape
        size = int(np.prod(shape))
        self.ndim = len(grid_dim_sizes)
        self._grid = Conv2DGrid(
            grid_dim_sizes,
            conv_size=conv_size,
            conv_chans=conv_chans,
            conv_layers=conv_layers,
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
