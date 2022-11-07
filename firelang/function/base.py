from __future__ import division, annotations
from typing import Tuple, Mapping, Any, Iterable, Callable, Union
import operator as op
import numpy as np
from torch.nn import Module, ModuleList
import firelang
from firelang.stack import StackingSlicing, _parse_shape

__all__ = ["Functional"]


class Functional(StackingSlicing):
    def __init__(
        self,
        locals_: Mapping[str, Any],
        unsliceable_params: Iterable[str] = [],
        prev=[],
        operator=None,
    ):
        StackingSlicing.__init__(
            self, locals_=locals_, unsliceable_params=unsliceable_params
        )
        prevm = [p for p in prev if isinstance(p, Module)]
        self.prevm = ModuleList(prevm) if len(prevm) else prevm
        self.prev = prev
        self.operator = operator

    def __add__(self, func_or_scalar):
        if isinstance(func_or_scalar, StackingSlicing):
            assert self.shape == func_or_scalar.shape
        return Functional(
            locals_={"shape": self.shape},
            prev=[self, func_or_scalar],
            operator=op.add,
        )

    def __sub__(self, func_or_scalar):
        if isinstance(func_or_scalar, StackingSlicing):
            assert self.shape == func_or_scalar.shape
        return Functional(
            locals_={"shape": self.shape},
            prev=[self, func_or_scalar],
            operator=op.sub,
        )

    def __mul__(self, other: Union[float, Functional, firelang.Measure]):
        """
        Args:
            other (Union[float, Functional, Measure]):
            - if is `float` or `Functional`: generate a new Functional.
            - if is `Measure`, compute the paired integral.
        """
        if isinstance(other, float) or isinstance(other, Functional):
            return Functional(
                locals_={"shape": self.shape},
                prev=[self, other],
                operator=op.mul,
            )
        elif isinstance(other, firelang.Measure):
            return other.integral(self)
        else:
            raise TypeError(
                f"`other` must be a float or Functional or Measure object, not {type(other)}."
            )

    def __truediv__(self, func_or_scalar):
        if isinstance(func_or_scalar, StackingSlicing):
            assert self.shape == func_or_scalar.shape
        return Functional(
            locals_={"shape": self.shape},
            prev=[self, func_or_scalar],
            operator=op.truediv,
        )

    def __pow__(self, pow: float):
        return Functional(
            locals_={"shape": self.shape},
            prev=[self, pow],
            operator=op.pow,
        )

    def __neg__(self):
        return Functional(locals_={"shape": self.shape}, prev=[self], operator=op.neg)

    def neg(self):
        return Functional(locals_={"shape": self.shape}, prev=[self], operator=op.neg)

    def __abs__(self):
        return Functional(locals_={"shape": self.shape}, prev=[self], operator=op.abs)

    def abs(self):
        return Functional(locals_={"shape": self.shape}, prev=[self], operator=op.abs)

    def apply_op(self, mapping: Callable, *other_nodes):
        return Functional(
            locals_={"shape": self.shape},
            prev=[self, *other_nodes],
            operator=mapping,
        )

    def forward(self, x, *args, **kwargs):
        operator = self.operator
        prev = self.prev
        if operator is None:
            return NotImplementedError
        elif operator is op.add:
            return prev[0].forward(x, *args, **kwargs) + prev[1].forward(
                x, *args, **kwargs
            )
        elif operator is op.sub:
            return prev[0].forward(x, *args, **kwargs) - prev[1].forward(
                x, *args, **kwargs
            )
        elif operator is op.mul:
            return prev[0].forward(x, *args, **kwargs) * prev[1].forward(
                x, *args, **kwargs
            )
        elif operator is op.truediv:
            return prev[0].forward(x, *args, **kwargs) / prev[1].forward(
                x, *args, **kwargs
            )
        elif operator is op.pow:
            return prev[0].forward(x, *args, **kwargs) ** prev[1]
        elif operator is op.neg:
            return -prev[0].forward(x, *args, **kwargs)
        elif operator is op.abs:
            return prev[0].forward(x, *args, **kwargs).abs()
        elif isinstance(operator, Callable):
            return operator(prev, x, *args, **kwargs)
        else:
            raise ValueError(f"Unrecognized operator: {operator}")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def is_leaf(self):
        return not hasattr(self, "prev") or not len(self.prev)

    def __getitem__(self, idx):
        if self.is_leaf():
            newop = StackingSlicing.__getitem__(self, idx)
        else:
            sliced = [
                node[idx] if isinstance(node, StackingSlicing) else node
                for node in self.prev
            ]
            newop = sliced[0].apply_op(self.operator, *sliced[1:])
        return newop

    def view(self, *shape, inplace: bool = False):
        shape = _parse_shape(shape, num_elements=int(np.prod(self.shape)))

        if inplace:
            if self.is_leaf():
                StackingSlicing.view(self, shape, inplace=True)
            else:
                for node in self.prev:
                    if isinstance(node, Functional):
                        Functional.view(node, shape, inplace=True)
            return self
        else:
            if self.is_leaf():
                newop = StackingSlicing.view(self, shape)
            else:
                prev = [
                    Functional.view(node, shape) if isinstance(node, Functional) else node
                    for node in self.prev
                ]
                newop = prev[0].apply_op(self.operator, *prev[1:])
            return newop

    def __repr__(self):
        if self.is_leaf():
            return Module.__repr__(self) + f", shape={self.shape}"
        else:
            segs = [f"{self.__class__.__name__}("]
            for i, node in enumerate(self.prev):
                for j, line in enumerate(repr(node).split("\n")):
                    if j == 0:
                        segs.append("  " + f"prev[{i}]: " + line)
                    else:
                        segs.append("  " + line)
            op_name = (
                self.operator.__name__
                if hasattr(self.operator, "__name__")
                else self.operator.__class__.__name__
            )
            segs.append(f"), operator={op_name}")
            return "\n".join(segs)

    def restack(self, shape: Tuple[int] = None):
        if self.is_leaf():
            newop = StackingSlicing.restack(self, shape)
        else:
            stacked = [
                node.restack(shape) if hasattr(node, "restack") else node
                for node in self.prev
            ]
            newop = stacked[0].apply_op(self.operator, *stacked[1:])
        return newop

    stack = restack

    def __matmul__(self, measure: firelang.Measure):
        assert isinstance(measure, firelang.Measure)
        return measure.integral(self, cross=True)