from __future__ import annotations
from typing import List
from typing_extensions import Literal
import torch
from torch import Tensor
from torch.nn import ModuleList
from firelang.utils.shape import check_shape_consistency
from firelang.utils.index import normalize_index
from .base import Functional


class Add(Functional):
    def __init__(self, f1: Functional, f2: Functional | float):
        if isinstance(f2, Functional):
            assert check_shape_consistency(f1.shape, f2.shape)
            assert check_shape_consistency(f1.shape_out, f2.shape_out)
            assert f1.dim == f2.dim
            self.dim = f1.dim
        Functional.__init__(
            self,
            locals_={"shape": f1.shape, "shape_out": f1.shape_out},
            is_fleaf=False,
        )
        self.f1 = f1
        self.f2 = f2

    def forward(self, x, *args, **kwargs):
        fx1 = self.f1.forward(x, *args, **kwargs)
        fx2 = (
            self.f2.forward(x, *args, **kwargs)
            if isinstance(self.f2, Functional)
            else self.f2
        )
        return fx1 + fx2

    def restack(self, shape):
        f1stack = self.f1.restack(shape)
        f2stack = self.f2.restack(shape) if isinstance(self.f2, Functional) else self.f2
        return Add(f1stack, f2stack)


class Sub(Functional):
    def __init__(self, f1: Functional, f2: Functional | float):
        if isinstance(f2, Functional):
            assert check_shape_consistency(f1.shape, f2.shape)
            assert check_shape_consistency(f1.shape_out, f2.shape_out)
            assert f1.dim == f2.dim
            self.dim = f1.dim
        Functional.__init__(
            self,
            locals_={"shape": f1.shape, "shape_out": f1.shape_out},
            is_fleaf=False,
        )
        self.f1 = f1
        self.f2 = f2

    def forward(self, x, *args, **kwargs):
        fx1 = self.f1.forward(x, *args, **kwargs)
        fx2 = (
            self.f2.forward(x, *args, **kwargs)
            if isinstance(self.f2, Functional)
            else self.f2
        )
        return fx1 - fx2

    def restack(self, shape):
        f1stack = self.f1.restack(shape)
        f2stack = self.f2.restack(shape) if isinstance(self.f2, Functional) else self.f2
        return Sub(f1stack, f2stack)


class Mul(Functional):
    def __init__(self, f1: Functional, f2: Functional | float):
        if isinstance(f2, Functional):
            assert check_shape_consistency(f1.shape, f2.shape)
            assert check_shape_consistency(f1.shape_out, f2.shape_out)
            assert f1.dim == f2.dim
            self.dim = f1.dim
        Functional.__init__(
            self,
            locals_={"shape": f1.shape, "shape_out": f1.shape_out},
            is_fleaf=False,
        )
        self.f1 = f1
        self.f2 = f2

    def forward(self, x, *args, **kwargs):
        fx1 = self.f1.forward(x, *args, **kwargs)
        fx2 = (
            self.f2.forward(x, *args, **kwargs)
            if isinstance(self.f2, Functional)
            else self.f2
        )
        return fx1 * fx2

    def restack(self, shape):
        f1stack = self.f1.restack(shape)
        f2stack = self.f2.restack(shape) if isinstance(self.f2, Functional) else self.f2
        return Mul(f1stack, f2stack)


class TrueDiv(Functional):
    def __init__(self, f1: Functional, f2: Functional | float):
        if isinstance(f2, Functional):
            assert check_shape_consistency(f1.shape, f2.shape)
            assert check_shape_consistency(f1.shape_out, f2.shape_out)
            assert f1.dim == f2.dim
            self.dim = f1.dim
        Functional.__init__(
            self,
            locals_={"shape": f1.shape, "shape_out": f1.shape_out},
            is_fleaf=False,
        )
        self.f1 = f1
        self.f2 = f2

    def forward(self, x, *args, **kwargs):
        fx1 = self.f1.forward(x, *args, **kwargs)
        fx2 = (
            self.f2.forward(x, *args, **kwargs)
            if isinstance(self.f2, Functional)
            else self.f2
        )
        return fx1 / fx2

    def restack(self, shape):
        f1stack = self.f1.restack(shape)
        f2stack = self.f2.restack(shape) if isinstance(self.f2, Functional) else self.f2
        return TrueDiv(f1stack, f2stack)


class Pow(Functional):
    def __init__(self, f: Functional, pow: float):
        Functional.__init__(
            self,
            locals_={"shape": f.shape, "shape_out": f.shape_out},
            is_fleaf=False,
        )
        self.f = f
        self.pow = pow
        self.dim = f.dim

    def forward(self, x, *args, **kwargs):
        fx = self.f.forward(x, *args, **kwargs)
        return fx**self.pow

    def restack(self, shape):
        fstack = self.f.restack(shape)
        return Pow(fstack, self.pow)


class Neg(Functional):
    def __init__(self, f: Functional):
        Functional.__init__(
            self,
            locals_={"shape": f.shape, "shape_out": f.shape_out},
            is_fleaf=False,
        )
        self.f = f
        self.dim = f.dim

    def forward(self, x, *args, **kwargs):
        return -self.f.forward(x, *args, **kwargs)

    def restack(self, shape):
        fstack = self.f.restack(shape)
        return Neg(fstack)


class Abs(Functional):
    def __init__(self, f: Functional):
        Functional.__init__(
            self,
            locals_={"shape": f.shape, "shape_out": f.shape_out},
            is_fleaf=False,
        )
        self.f = f
        self.dim = f.dim

    def forward(self, x, *args, **kwargs):
        return self.f.forward(x, *args, **kwargs).abs()

    def restack(self, shape):
        fstack = self.f.restack(shape)
        return Abs(fstack)


class Sequential(Functional):
    def __init__(self, fs: List[Functional]):
        for f1, f2 in zip(fs[:-1], fs[1:]):
            assert f1.shape_out == f2.shape
            assert f1.dim == f2.dim
            f2.is_fleaf = False
        self.dim = fs[0].dim
        Functional.__init__(
            self,
            locals_={"shape": fs[0].shape, "shape_out": fs[-1].shape_out},
            is_fleaf=False,
        )
        self.fs = ModuleList(fs)

    def forward(self, x, *args, **kwargs):
        for f in self.fs:
            x = f.forward(x, *args, **kwargs)
        return x

    def restack(self, shape):
        return Sequential([f.restack(shape) for f in self.fs])


