from __future__ import annotations
from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Parameter
from .base import Measure
from firelang.utils.limits import parse_rect_limits

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
        stack_size: int = 1,
    ):
        Measure.__init__(self, locals())
        if limits is None:
            self._x = Parameter(torch.randn(stack_size, k, dim, dtype=torch.float32))
        else:
            limits = torch.tensor(
                parse_rect_limits(limits, dim), dtype=torch.float32
            )  # (dim, 2)
            ranges = (limits[:, 1] - limits[:, 0])[None, None]  # (1, 1, dim)
            starts = limits[:, 0][None, None]  # (1, 1, dim)
            self._x = Parameter(
                torch.rand(stack_size, k, dim, dtype=torch.float32) * ranges + starts
            )
        self._m = (
            1.0 if mfix else Parameter(torch.ones(stack_size, k, dtype=torch.float32))
        )

        self.dim = dim
        self.k = k
        self.stack_size = stack_size
        self.limits = limits
        self.mfix = mfix

    def integral(self, func, cross=False, batch_size=1000000, sum=True):
        x = self.get_x()

        func_stack_size = len(func)
        col_stride = (batch_size + func_stack_size - 1) // func_stack_size

        if cross:
            res = []
            for i in range(0, self.stack_size, col_stride):
                m = (
                    self._m
                    if isinstance(self._m, float)
                    else self._m[i : i + col_stride].abs()
                )
                batch = func(x[i : i + col_stride], cross=cross) * m
                if sum:
                    batch = batch.sum(dim=-1)
                res.append(batch)
            res = torch.cat(res, dim=-1)
        else:
            m = self._m if isinstance(self._m, float) else self._m.abs()
            res = func(x, cross=cross) * m
            if sum:
                res = res.sum(dim=-1)
        return res

    def get_x(self):
        # _x: (stack_size, k, dim)
        if self.limits is not None:
            limits = self.limits.to(self.detect_device())
            ranges = (limits[:, 1] - limits[:, 0])[None, None]  # (1, 1, dim)
            _x = self._x / (ranges / 2)
            _x = torch.tanh(_x)
            _x = _x * (ranges / 2)

            if _x.isnan().any():
                print(f"_x has NaN: {_x}")
                print(f"self._x has NaN ?: {self._x.isnan().any()}")
                print(f"ranges: {ranges}")
                exit()
            return _x
        else:
            return self._x

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
        segs = ["DiracMixture("]
        segs.append(f"stack_size={self.stack_size}")
        segs.append(f", k={self.k}")
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
