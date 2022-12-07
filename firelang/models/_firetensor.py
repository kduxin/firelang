from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import Module
from firelang.measure import Measure
from firelang.function import Functional
from firelang.stack import IndexLike

__all__ = [
    "FireTensor",
    "FIRETensor",
]


class FireTensor(Module):
    def __init__(self, funcs: Functional, measures: Measure):
        Module.__init__(self)
        # check_shape_consistency(funcs.shape, measures.shape)
        self.funcs: Functional = funcs
        self.measures: Measure = measures

    def __getitem__(self, index: IndexLike) -> FireTensor:
        return FireTensor(self.funcs[index], self.measures[index])

    def view(self, *shape, inplace=False) -> FireTensor:
        if inplace:
            self.funcs.view(*shape, inplace=True)
            return self
        else:
            return FireTensor(
                funcs=self.funcs.view(*shape, inplace=False),
                measures=self.measures.view(*shape, inplace=False),
            )

    def __add__(self, other: FireTensor | Functional) -> FireTensor:
        if isinstance(other, FireTensor):
            return FireTensor(
                funcs=self.funcs + other.funcs, measures=self.measures + other.measures
            )
        elif isinstance(other, Functional):
            return FireTensor(funcs=self.funcs + other, measures=self.measures)
        else:
            raise TypeError(other)

    def __mul__(self, other: FIRETensor) -> Tensor:
        if id(other) == id(self):
            return self.measures.integral(self.funcs) * 2
        else:
            return other.measures.integral(self.funcs) + self.measures.integral(
                other.funcs
            )

    def __matmul__(self, other: FIRETensor) -> Tensor:
        if id(other) == id(self):
            mat = self.measures.integral(self.funcs, cross=True)
            return mat + torch.transpose(mat, -2, -1)
        else:
            return other.measures.integral(self.funcs, cross=True) + torch.transpose(
                self.measures.integral(other.funcs, cross=True), -2, -1
            )

    def __repr__(self):
        return (
            f"<FIRETensor(funcs={self.funcs.__class__.__name__}, "
            f"measures={self.measures.__class__.__name__}), "
            f"shape={self.funcs.shape}>"
        )

    def split(self, n_heads: int):
        return FireTensor(
            funcs=self.funcs.vmap(), measures=self.measures.split(n_heads)
        )

    def size(self):
        return self.funcs.shape

    @property
    def shape(self):
        return self.size()

    def detect_device(self):
        return self.funcs.detect_device()

    def detect_dtype(self):
        return self.funcs.detect_dtype()

    def flatten_parameter(self):
        return torch.cat(
            [self.funcs.flatten_parameter(), self.measures.flatten_parameter()], dim=-1
        )

    def load_flatten_parameter(self, flattened: Tensor) -> int:
        offset = self.funcs.load_flatten_parameter(flattened, offset=0)
        offset = self.measures.load_flatten_parameter(flattened, offset=offset)
        return offset

    def restack(self, shape):
        return FireTensor(
            funcs=self.funcs.restack(shape), measures=self.measures.restack(shape)
        )

    stack = restack


FIRETensor = FireTensor
