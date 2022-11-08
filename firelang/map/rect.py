from __future__ import annotations
from typing import List, Tuple, Union, Iterable
from typing_extensions import Literal
import numpy as np
import torch
from torch import Tensor, dtype, device
from torch.nn import Parameter

from firelang.function import Functional
from firelang.utils.timer import elapsed, Timer
from firelang.utils.limits import parse_rect_limits
from ._grid import Grid

__all__ = [
    "SmoothedRectMap",
]


class SmoothedRectMap(Functional):
    def __init__(
        self,
        limits: float | List[float] | Tuple[float, float] | List[Tuple[float, float]],
        grid_dim_sizes: List[int],
        rect_dim_sizes: int | List[int] = 3,
        rect_weight_decay: Literal["gauss", "exp"] = "gauss",
        bandwidth_mode: Literal["parameter", "constant"] = "parameter",
        bandwidth_lb: float = 0.3,
        dtype: dtype = torch.float32,
        device: device = "cuda",
        shape: Tuple[int] = (1,),
    ):
        Functional.__init__(self, locals())

        self.shape = shape
        size = int(np.prod(shape))
        self.ndim = len(grid_dim_sizes)
        self._grid: Grid = Grid(grid_dim_sizes, shape=shape, dtype=dtype, device=device)

        def _sizes_to_tensor(sizes: int | List[int], ndim: int) -> Tensor:
            if not isinstance(sizes, Iterable):
                sizes = [sizes] * ndim
            sizes = torch.tensor(sizes, dtype=torch.long, device=device)
            return sizes

        self.grid_dim_sizes = _sizes_to_tensor(grid_dim_sizes, self.ndim)
        self.rect_dim_sizes = _sizes_to_tensor(rect_dim_sizes, self.ndim)
        self.limits = torch.tensor(
            parse_rect_limits(limits, self.ndim), dtype=dtype, device=device
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

    @property
    def bandwidth(self):
        bandwidth = self._bandwidth
        bandwidth_mode = self.bandwidth_mode
        if bandwidth_mode == "constant":
            bandwidth.to(self.detect_device())
            return bandwidth
        elif bandwidth_mode == "parameter":
            return bandwidth.abs() + self.bandwidth_lb
        else:
            raise ValueError(bandwidth_mode)

    def get_bandwidth(self):
        return self.bandwidth

    @property
    def grid(self):
        return self._grid.gridvals

    def get_grid(self):
        return self.grid

    def detect_device(self):
        return self._grid.detect_device()

    def forward(self, locs: Tensor) -> Tensor:
        """Interpolate grid values at `locs`

        Args:
            - locs (Tensor): (...shape, dim).
                Locations in the grid

        Returns:
            Tensor: (...shape,)
        """

        device = locs.device
        ndim = locs.shape[-1]

        with Timer(elapsed, "gridmap/location_transform"):
            """Transform values of `locs` from the range
            [limits_lower, limits_upper] to [0, dim_size]
            """
            limits = self.limits.to(device)  # (ndim, 2)
            limits_lower = limits[:, 0]
            limits_upper = limits[:, 1]
            assert (
                locs >= limits_lower
            ).all(), f"locs: {locs}\nSmallest value = {locs.min().item()}"
            assert (
                locs <= limits_upper
            ).all(), f"locs: {locs}\nLargest value = {locs.max().item()}"
            dim_sizes = self.grid_dim_sizes.to(device)  # (ndim,)
            locs = (locs - limits_lower) / (limits_upper - limits_lower)
            locs = locs * dim_sizes.type(torch.float32)

        with Timer(elapsed, "gridmap/vertices"):
            rect_dim_sizes = self.rect_dim_sizes.to(device)
            rect_dim_sizes = rect_dim_sizes  # (ndim,)

            lower = torch.ceil(locs.data - rect_dim_sizes / 2).type(torch.long)
            lower = torch.maximum(lower, torch.zeros_like(lower))
            upper = torch.minimum(lower + rect_dim_sizes, dim_sizes)
            lower = upper - rect_dim_sizes

            corners = lower
            # (...shape, ndim, 2)

        with Timer(elapsed, "gridmap/subgrid_weights", sync_cuda=True):

            rect_dim_sizes = self.rect_dim_sizes.to(device)
            distances = rectangle_distance_to_loc(locs, corners, rect_dim_sizes)
            # (...shape, n1, ..., nd)
            bandwidth = self.bandwidth.to(device).view(*self.shape, *[1] * ndim)
            # (...shape,  1, ...,  1)
            subgrid_weights = weights_from_distances(
                distances, ndim, self.rect_weight_decay, bandwidth
            )  # (...shape, n1, ..., nd)

        with Timer(elapsed, "gridmap/subgrid_values", sync_cuda=True):
            subgrid_vals = self._grid.slice_rectangle(
                corners,
                rect_dim_sizes,
            )
            # (...shape, n1, ..., nd)

        with Timer(elapsed, "gridmap/weighted_sum"):
            vals = (subgrid_vals * subgrid_weights).sum(dim=list(range(-ndim, 0)))
            # (...shape,)

        return vals


def rectangle_distance_to_loc(
    locs: Tensor, corners: Tensor, rect_dim_sizes: Tensor
) -> Tensor:
    """Compute the distance between `locs` and the grid points within the
    (hyper-)rectangle that are specified by `corners` and `rect_dim_sizes`.

    Args:
        - locs (Tensor): (...shape, ndim) locations
        - corners (Tensor): (...shape, ndim).
        - rect_dim_sizes (Tensor): rect_dim_sizes at each dimension of the rectangle.

    Returns:
        Tensor: let each rectangle be represented by (n1, n2, ..., nd), returns \
            a Tensor with shape (...shape, n1, n2, ..., nd).
    """
    device = locs.device
    (*shape, ndim) = locs.shape

    shifts = corners.to(locs) - locs  # (...shape, ndim)
    distsq = 0
    for d in range(ndim):
        shifts_at_dim_d = shifts[..., d : d + 1] + torch.arange(
            rect_dim_sizes[d], device=device, dtype=torch.long
        )  # (...shape, nd=rect_dim_sizes[d])
        distsq_at_dim_d = (shifts_at_dim_d**2).reshape(
            *shape, *([1] * d), -1, *([1] * (ndim - d - 1))
        )  # (...shape, 1, ..., 1, nd, 1, ..., 1)
        distsq = distsq + distsq_at_dim_d
    return distsq**0.5


def weights_from_distances(
    distances: Tensor,
    ndim: int,
    decay: Literal["gauss", "exp"],
    bandwidth: Tensor,
) -> Tensor:
    """Compute weights from distances.

    Args:
        - distances (Tensor): (...shape, *)
        - decay ("gauss" | "exp"): different weight decaying patterns with respect to distance.
            - "gauss": proportional to $exp(-d^2)$
            - "exp": proportional to $exp(-d)$
        - bandwidth (Tensor): (...shape,) the weights decay slower with a larger `bandwidth`

    Returns:
        Tensor: (...shape, *)

    """
    distances = distances / bandwidth
    if decay == "gauss":
        logweights = -(distances**2)
    elif decay == "exp":
        logweights = -distances
    else:
        raise ValueError(decay)

    """ numerical-safer normalization """
    ndim_ids = list(range(-ndim, 0))
    logweights = logweights - logweights.amax(dim=ndim_ids, keepdim=True)
    weights = logweights.exp()
    weights = weights / weights.sum(dim=ndim_ids, keepdim=True)

    return weights
