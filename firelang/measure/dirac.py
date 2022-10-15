import torch
from torch.nn import Parameter
from .base import Measure

__all__ = ["DiracMixture"]


class DiracMixture(Measure):
    _x: Parameter
    _m: Parameter

    def __init__(
        self,
        dim: int,
        k: int,
        range: float = None,
        mfix: bool = False,
        stack_size: int = 1,
    ):
        Measure.__init__(self, locals())
        assert range is None or range > 0
        self._x = (
            Parameter(torch.randn(stack_size, k, dim, dtype=torch.float32))
            if range is None
            else Parameter(
                torch.rand(stack_size, k, dim, dtype=torch.float32) * (2 * range)
                - range
            )
        )
        self._m = (
            1.0 if mfix else Parameter(torch.ones(stack_size, k, dtype=torch.float32))
        )

        self.dim = dim
        self.k = k
        self.stack_size = stack_size
        self.range = range
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
        if hasattr(self, "range") and self.range is not None:
            return torch.tanh(self._x / self.range) * self.range
        else:
            return self._x

    @property
    def x(self):
        return self.get_x()

    def get_m(self):
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
        if self.range is not None:
            segs.append(f", range=[-{self.range}, {self.range}]")
        segs.append(")")
        return "".join(segs)

    def _parameter_shape_hash(self):
        return hash(
            Measure._parameter_shape_hash(self) + hash(self.mfix) + hash(self.range)
        )
