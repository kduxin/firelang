from __future__ import annotations
from typing import List, Tuple, Union, Iterable
from typing_extensions import Literal
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
        stack_size: int = 1,
    ):
        Functional.__init__(self, locals())

        self.stack_size = stack_size
        self.ndim = len(grid_dim_sizes)
        self._grid: Grid = Grid(
            grid_dim_sizes, stack_size=stack_size, dtype=dtype, device=device
        )

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
            self._bandwidth = bandwidth_lb * torch.ones(stack_size, dtype=dtype)
        elif bandwidth_mode == "parameter":
            self._bandwidth = Parameter(torch.ones(stack_size, dtype=dtype))
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

    def forward(self, locs: Tensor, cross: bool = False) -> Tensor:
        """Interpolate grid values at `locs`

        Args:
            - locs (Tensor): (measure_stack, batch_size, dim).
                Locations in the grid
                - If cross == False:
                    measure_stack must be equal to self.stack_size

            - cross (bool, optional): Defaults to False.

        Returns:
            Tensor:
            - If cross == False:
                (measure_stack, batch_size)
            - If cross == True:
                (self.stack_size, measure_stack, batch_size)
        """

        device = locs.device
        measure_stack, batch_size, ndim = locs.shape
        map_stack = self.stack_size
        if not cross:
            assert map_stack == measure_stack

        with Timer(elapsed, "gridmap/location_transform"):
            """Transform values of `locs` from the range
            [limits_lower, limits_upper] to [0, dim_size]
            """
            limits = self.limits.to(device)  # (ndim, 2)
            eps = 0
            limits_lower = limits[:, 0].reshape(1, 1, -1) - eps
            limits_upper = limits[:, 1].reshape(1, 1, -1) + eps
            # print('locs:', locs)
            assert (
                locs >= limits_lower
            ).all(), f"locs: {locs}\nSmallest value = {locs.min().item()}"
            assert (
                locs <= limits_upper
            ).all(), f"locs: {locs}\nLargest value = {locs.max().item()}"
            dim_sizes = self.grid_dim_sizes.to(device)  # (ndim,)
            locs = (locs - limits_lower) / (limits_upper - limits_lower)
            locs = locs * dim_sizes.type(torch.float32).reshape(1, 1, -1)

        with Timer(elapsed, "gridmap/vertices"):
            rect_dim_sizes = self.rect_dim_sizes.to(device)
            rect_dim_sizes = rect_dim_sizes.reshape(1, 1, -1)  # (ndim,)  # (1, 1, ndim)

            lower = torch.ceil(locs.data - rect_dim_sizes / 2).type(torch.long)
            lower = torch.maximum(lower, torch.zeros_like(lower))
            upper = torch.minimum(lower + rect_dim_sizes, dim_sizes.reshape(1, 1, -1))
            lower = upper - rect_dim_sizes

            corners = lower
            # (measure_stack, batch_size, ndim, 2)

        with Timer(elapsed, "gridmap/subgrid_weights", sync_cuda=True):

            rect_dim_sizes = self.rect_dim_sizes.to(device)
            # distances = self._grid.rectangle_distance_to_loc(locs, corners, rect_dim_sizes)
            distances = rectangle_distance_to_loc(locs, corners, rect_dim_sizes)
            # (measure_stack, batch_size, n1, ..., nd)
            bandwidth = self.bandwidth.to(device)
            subgrid_weights = weights_from_distances(
                distances, self.rect_weight_decay, bandwidth, cross=cross
            )
            # If cross==False: (measure_stack, batch_size, n1, ..., nd)
            # Else: (self.stack_size, measure_stack, batch_size, n1, ..., nd)

        if not cross:
            with Timer(elapsed, "gridmap/subgrid_values", sync_cuda=True):
                subgrid_vals = self._grid.slice_rectangle(
                    corners, rect_dim_sizes, cross=cross
                )
                # (measure_stack, batch_size, n1, ..., nd)

            with Timer(elapsed, "gridmap/weighted_sum"):
                vals = (subgrid_vals * subgrid_weights).sum(
                    dim=list(range(2, ndim + 2))
                )
                # (measure_stack, batch_size)
        else:
            with Timer(elapsed, "gridmap/subgrid_values", sync_cuda=True):
                subgrid_vals = self._grid.slice_rectangle(
                    corners, rect_dim_sizes, cross=cross
                )
                # (self.stack_size, measure_stack, batch_size, n1, ..., nd)

            with Timer(elapsed, "gridmap/weighted_sum"):
                vals = (subgrid_vals * subgrid_weights).sum(
                    dim=list(range(3, ndim + 3))
                )
                # (self.stack_size, measure_stack, batch_size)

        return vals


def rectangle_distance_to_loc(
    locs: Tensor, corners: Tensor, rect_dim_sizes: Tensor
) -> Tensor:
    """Compute the distance between `locs` and the grid points within the
    (hyper-)rectangle that are specified by `corners` and `rect_dim_sizes`.

    Args:
        - locs (Tensor): (measure_stack, batch_size, ndim) locations
        - corners (Tensor): (measure_stack, batch_size, ndim). \
            If cross == False: measure_stack must be equal to self.stack_size
        - rect_dim_sizes (Tensor): rect_dim_sizes at each dimension of the rectangle.

    Returns:
        Tensor: let each rectangle be represented by (n1, n2, ..., nd), returns \
            a Tensor with shape (measure_stack, batch_size, n1, n2, ..., nd).
    """
    measure_stack, batch_size, ndim = locs.shape
    device = locs.device

    shifts = corners.to(locs) - locs  # (measure_stack, batch_size, ndim)
    distsq = 0
    for d in range(ndim):
        shifts_at_dim_d = shifts[:, :, d : d + 1] + torch.arange(
            rect_dim_sizes[d], device=device, dtype=torch.long
        ).reshape(
            1, 1, -1
        )  # (measure_stack, batch_size, nd=rect_dim_sizes[d])
        distsq_at_dim_d = (shifts_at_dim_d**2).reshape(
            measure_stack, batch_size, *([1] * d), -1, *([1] * (ndim - d - 1))
        )  # (measure_stack, batch_size, 1, ..., 1, nd, 1, ..., 1)
        distsq = distsq + distsq_at_dim_d
    return distsq**0.5


def weights_from_distances(
    distances: Tensor,
    decay: Literal["gauss", "exp"],
    bandwidth: Tensor,
    cross: bool = False,
) -> Tensor:
    """Compute weights from distances.

    Args:
        - distances (Tensor): (measure_stack, batch_size, *)
        - decay ("gauss" | "exp"): different weight decaying patterns with respect to distance.
            - "gauss": proportional to $exp(-d^2)$
            - "exp": proportional to $exp(-d)$
        - bandwidth (Tensor): (map_stack,) the weights decay slower with a larger `bandwidth`

    Returns:
        Tensor:
        - If cross == False, measure_stack must be equal to map_stack.
          Returns (measure_stack, *)
        - Else, returns (map_stack, measure_stack, *)

    """
    ndim = distances.ndim - 2
    measure_stack = distances.shape[0]
    map_stack = bandwidth.shape[0]

    bandwidth = bandwidth.reshape(map_stack, 1, *[1 for _ in range(ndim)])
    if not cross:
        assert map_stack == measure_stack
    else:
        distances = distances[None]
        bandwidth = bandwidth[:, None]

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
