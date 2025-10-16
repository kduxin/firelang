from __future__ import division, annotations
from typing import Mapping, Any, Iterable, Callable
import copy
from collections import OrderedDict
from torch.nn import Module, ModuleList, ModuleDict
import firelang
from firelang.stack import StackingSlicing

__all__ = ["Functional"]


_vmap_log = set()


class cached_forward:
    def __init__(self, callable: Callable, identifier: str):
        self.callable = callable
        self.identifier = identifier
        self.cache = OrderedDict()

    def __call__(self, x, *args, **kwargs):
        key = (id(x), *args, *sorted(kwargs.items(), key=lambda x: x[0]))
        if key in self.cache:
            print(f"Triggered cache at {self.identifier}")
            return self.cache[key]

        fx = self.callable(x, *args, **kwargs)
        while len(self.cache) >= 1:
            self.cache.popitem(last=False)
        self.cache[key] = fx
        return fx

    def __getstate__(self):
        return {
            "cache": OrderedDict(),
            **{key: val for key, val in self.__dict__.items() if key != "cache"},
        }


class Functional(StackingSlicing):
    def __init__(
        self,
        locals_: Mapping[str, Any],
        unsliceable_params: Iterable[str] = [],
        operator: Callable = None,
        is_fleaf: bool = True,
    ):
        if "shape_out" not in locals_:
            locals_["shape_out"] = locals_["shape"]
        StackingSlicing.__init__(
            self, locals_=locals_, unsliceable_params=unsliceable_params
        )
        self._sanity_check()
        self.shape_out = locals_["shape_out"]
        # prevm = [p for p in prev if isinstance(p, Module)]
        # self.prevm = ModuleList(prevm) if len(prevm) else prevm
        # self.prev = prev
        self.operator = cached_forward(
            self.forward if operator is None else operator,
            identifier=f"{self.__class__.__name__} (id={id(self)})",
        )
        self.is_fleaf = is_fleaf

    def _sanity_check(self):
        assert "shape_out" in self.init_locals, (
            "A `StackingSlicing` subclass must accept `shape_out` as an "
            "initialization argument."
        )

    def __add__(self, other: Functional | float):
        return firelang.function.Add(self, other)

    def __sub__(self, other: Functional | float):
        return firelang.function.Sub(self, other)

    def __mul__(self, other: Functional | float | firelang.Measure):
        """
        Args:
            other (Union[float, Functional, Measure]):
            - if is `float` or `Functional`: generate a new Functional.
            - if is `Measure`, compute the paired integral.
        """
        if isinstance(other, Functional) or isinstance(other, float):
            return firelang.function.Mul(self, other)
        elif isinstance(other, firelang.Measure):
            return other.integral(self)
        else:
            raise TypeError(
                f"`other` must be a float or Functional or Measure object, not {type(other)}."
            )

    def __truediv__(self, other: Functional | float):
        return firelang.function.TrueDiv(self, other)

    def __pow__(self, pow: float):
        return firelang.function.Pow(self, pow)

    def __neg__(self):
        return firelang.function.Neg(self)

    def neg(self):
        return self.__neg__()

    def __abs__(self):
        return firelang.function.Abs(self)

    def abs(self):
        return self.__abs__()

    def __matmul__(self, other: firelang.Measure):
        return other.integral(self, cross=True)

    def __call__(self, x, *args, **kwargs):
        operator = self.operator
        if operator is None:
            return NotImplementedError
        elif isinstance(operator, Callable):
            fx = operator(x, *args, **kwargs)
        else:
            raise ValueError(f"Unrecognized operator: {operator}")
        return fx

    def clear_cache(self):
        self.operator.clear_cache()
        for child in self.func_children():
            child.clear_cache()

    def vmap(self, num_extra_dimensions: int = 1, inplace: bool = False) -> Functional:
        _vmap_log.clear()
        if inplace:
            self._vmap_(num_extra_dimensions)
            new = self
        else:
            new = copy.deepcopy(self)
            for name, p_from in self.named_parameters():
                p_to = new.get_parameter(name)
                p_to.requires_grad_(False)
                p_to.copy_(p_from)
            new._vmap_(num_extra_dimensions)
        _vmap_log.clear()
        return new

    def _vmap_(self, num_extra_dimensions: int = 1):
        if id(self) not in _vmap_log:
            _vmap_log.add(id(self))
            self.shape = (*[1] * num_extra_dimensions, *self.shape)
            self.shape_out = (*[1] * num_extra_dimensions, *self.shape_out)
            for func in self.func_children():
                func._vmap_(num_extra_dimensions)

    def func_children(self):
        return (child for name, child in self.named_func_children())

    def named_func_children(self):
        for name, child in self.named_children():
            if isinstance(child, Functional):
                yield name, child
            elif isinstance(child, ModuleList):
                for i, c in enumerate(child):
                    yield f"{name}.{i}", c
            elif isinstance(child, ModuleDict):
                for n, c in child.items():
                    yield f"{name}.{n}", c
        return StopIteration

    def __repr__(self):
        if self.is_fleaf:
            reprstr = Module.__repr__(self)
        else:
            segs = [f"{self.__class__.__name__} {self.shape}->{self.shape_out} ("]
            for name, child in self.named_children():
                s = repr(child)
                for j, line in enumerate(s.split("\n")):
                    if j == 0:
                        segs.append("  " + f"{name}: " + line)
                    else:
                        segs.append("  " + line)
            segs.append(")")
            reprstr = "\n".join(segs)
        return reprstr
