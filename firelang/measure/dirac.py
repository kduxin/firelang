from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from .base import Measure
from firelang.utils.limits import parse_rect_limits
from firelang.function import Functional

__all__ = [
    "DiracMixture",
]


class DiracMixture(Measure):
    _x: Parameter
    _m: Parameter

    def __init__(
        self,
        dim: int,
        k: int,
        limits: float | Tuple[float, float] | List[Tuple[float, float]] = None,
        mfix: bool = False,
        shape: Tuple[int] = (1,),
    ):
        Measure.__init__(self, locals())
        size = int(np.prod(shape))
        if limits is None:
            self._x = Parameter(torch.randn(size, k, dim, dtype=torch.float32))
        else:
            limits = torch.tensor(
                parse_rect_limits(limits, dim), dtype=torch.float32
            )  # (dim, 2)
            ranges = (limits[:, 1] - limits[:, 0])[None, None]  # (1, 1, dim)
            starts = limits[:, 0][None, None]  # (1, 1, dim)
            self._x = Parameter(
                torch.rand(size, k, dim, dtype=torch.float32) * ranges + starts
            )
        self._m = 1.0 if mfix else Parameter(torch.ones(size, k, dtype=torch.float32))

        self.dim = dim
        self.k = k
        self.limits = limits
        self.mfix = mfix
        self.shape = shape

    def integral(
        self,
        func: Functional,
        cross: bool = False,
        batch_size: int = 1000000,
        sum: bool = True,
    ) -> Tensor:
        if not cross:
            m = self.m.view(*self.shape, self.k)  # (...shape, k)
            x = self.x.view(*self.shape, self.k, self.dim)  # (...shape, k, dim)
            func = func.view(*func.shape, 1)
            fx = func(x) * m
        else:
            assert (
                self.shape[:-1] == func.shape[:-1]
            ), f"Shape inconsistent: {func.shape[:-1]} ({func.shape}) != {self.shape[:-1]} ({self.shape})."

            measure_size = self.shape[-1]
            func_size = func.shape[-1]

            m = self.m.view(*self.shape[:-1], 1, measure_size, self.k)
            x = self.x.view(*self.shape[:-1], 1, measure_size, self.k, self.dim)
            func = func.view(*func.shape[:-1], func_size, 1, 1)

            size = func_size * self.k
            nrow_per_batch = (batch_size + size - 1) // size
            fx = []
            for i in range(0, measure_size, nrow_per_batch):
                _x = x[..., i : i + nrow_per_batch, :, :]
                _m = m[..., i : i + nrow_per_batch, :]
                _fx = func(_x) * _m  # (...shape[:-1], _nrow_per_batch, measure_size, k)
                fx.append(_fx)
            fx = torch.cat(fx, dim=-2)

        if sum:
            fx = fx.sum(-1)  # (...shape[:-1], func_size, measure_size, k)
        return fx

    def get_x(self):
        # _x: (*shape, k, dim)
        if self.limits is None:
            return self._x
        else:
            limits = self.limits.to(self.detect_device())
            ranges = (limits[:, 1] - limits[:, 0])[None, None]  # (1, 1, dim)
            _x = self._x / (ranges / 2)
            _x = torch.tanh(_x)
            _x = _x * (ranges / 2)
            return _x

    @property
    def x(self):
        return self.get_x()

    def get_m(self):
        if isinstance(self._m, Tensor):
            return self._m.abs()
        else:
            return self._m

    @property
    def m(self):
        return self.get_m()

    def __repr__(self):
        segs = [f"DiracMixture(shape={self.shape}, k={self.k}"]
        if self.mfix:
            segs.append(f", m=1.0")

        if self.limits is not None:
            limits = self.limits.data.cpu().numpy().tolist()
        else:
            limits = None
        if limits is not None:
            segs.append(f", limits={limits}")
        segs.append(")")
        return "".join(segs)

    def _parameter_shape_hash(self):
        hsh = Measure._parameter_shape_hash(self)
        hsh += hash(self.mfix)
        hsh += hash(self.limits)
        return hash(hsh)