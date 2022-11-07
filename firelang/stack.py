from __future__ import annotations
from typing import List, Tuple, Mapping, Set, Iterable, Any, Union
from copy import deepcopy
from collections import OrderedDict, defaultdict
import inspect
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ModuleDict

__all__ = [
    "clear_cache",
    "current_cache_sizes",
    "StackingSlicing",
]

_cache = defaultdict(OrderedDict)


def clear_cache():
    _cache.clear()


def current_cache_sizes():
    return {key: len(dct) for key, dct in _cache.items()}


IndexLike = Union[
    int, slice, List, Tensor, None, Tuple[Union[int, slice, List, Tensor, None], ...]
]


class StackingSlicing(Module):
    init_locals: Mapping[str, Any]
    unsliceable_params: Set[str]
    skip_keys = ["shape"]

    def __init__(
        self,
        locals_: Mapping[str, Any],
        unsliceable_params: Iterable[str] = [],
    ):
        Module.__init__(self)
        self.init_locals = locals_
        self._sanity_check()
        self.shape = locals_["shape"]
        self.unsliceable_params = set(unsliceable_params)

    def register_extra_init_kwargs(self, **kwargs):
        for key, val in kwargs.items():
            self.init_locals[key] = val

    def register_extra_unsliceable_params(self, *names):
        for name in names:
            self.unsliceable_params.add(name)

    def _sanity_check(self):
        assert (
            type(self) != StackingSlicing
        ), "StackingSlicing must be initialized from a subclass"

        assert "shape" in self.init_locals, (
            "A `StackingSlicing` subclass must accept `shape` as an "
            "initialization argument."
        )

    def view(self, *shape, inplace: bool = False):
        shape = _parse_shape(shape, int(np.prod(self.shape)))

        if inplace:
            self.shape = shape
            return self

        else:
            new = deepcopy(self)
            new.shape = shape

            for module in new.children():
                if isinstance(module, StackingSlicing):
                    StackingSlicing.view(module, *shape, inplace=True)
                elif isinstance(module, ModuleList):
                    for m in module:
                        if isinstance(m, StackingSlicing):
                            StackingSlicing.view(m, *shape, inplace=True)
                elif isinstance(module, ModuleDict):
                    for _, m in module.items():
                        if isinstance(m, StackingSlicing):
                            StackingSlicing.view(m, *shape, inplace=True)
            return new

    def __getitem__(self, index: IndexLike):
        idtensor: Tensor = _parse_index(index, self.shape)
        ids = idtensor.reshape(-1)
        shape = tuple(idtensor.shape)

        to: StackingSlicing = self.restack(shape)

        # A parameter not listed in `unsliceable_params` should be
        # sliced and copied. Otherwise, the whole parameter is copied.
        for name, param in self.named_parameters(recurse=False):
            param_to = to.get_parameter(name)
            param_to.requires_grad_(False)
            if name in self.unsliceable_params:
                param_to.copy_(param)
            else:
                param_to.copy_(param[ids])

        # A submodule that is a `StackingSlicing` should be sliced
        # and copied. Otherwise, the whole submodule is copied.
        for name, module in self.named_children():
            submod_to: Module = to.get_submodule(name)
            submod_to.requires_grad_(False)

            if isinstance(module, StackingSlicing):
                submod_from: Module = module.__getitem__(index)
            elif isinstance(module, ModuleList):
                submod_from = ModuleList(
                    [
                        entry[index] if isinstance(entry, StackingSlicing) else entry
                        for entry in module
                    ]
                )
            elif isinstance(module, ModuleDict):
                submod_from = ModuleDict(
                    {
                        key: entry[index]
                        if isinstance(entry, StackingSlicing)
                        else entry
                        for key, entry in module.items()
                    }
                )
            else:
                submod_from: Module = module
            submod_from.shape = shape
            setattr(to, name, submod_from)

        to.shape = shape
        return to

    def detect_device(self):
        return next(iter(self.parameters())).device

    def _parameter_shape_hash(self):
        name_shapes = [(name, p.shape) for name, p in self.named_parameters()]
        return hash(tuple(name_shapes))

    def restack(
        self,
        shape: int | Tuple[int],
        use_cached: bool = True,
        max_cached_copies: int = 100,
    ):
        if not isinstance(shape, Tuple):
            shape = (shape,)

        tag = f"stacked/{self.__class__.__name__}-{self._parameter_shape_hash()}"
        if use_cached and shape in _cache[tag]:
            new = deepcopy(_cache[tag][shape])
        else:
            positional, keywords = self._recover_args_from_locals(
                locals_=self.init_locals,
            )
            new = self.__class__(
                *positional,
                **keywords,
                shape=shape,
            )
            new = new.to(self.detect_device())

            _cache[tag][shape] = new
            while len(_cache[tag]) > max_cached_copies:
                _cache[tag].popitem(last=False)  # pop the earliest

        new.shape = shape
        return new.to(self.detect_device())

    stack = restack

    def _recover_args_from_locals(
        self,
        locals_: Mapping[str, Any],
    ):
        signature = inspect.signature(self.__init__)

        positional = []
        keywords = {}
        for key, sign in signature.parameters.items():
            if key in self.skip_keys:
                continue

            value = locals_[key]
            if sign.kind not in [
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ]:
                positional.append(value)
            elif sign.kind == inspect.Parameter.VAR_POSITIONAL:
                positional.extend(value)
            elif sign.kind == inspect.Parameter.VAR_KEYWORD:
                keywords = {**keywords, **value}
        return positional, keywords


def _complete_ellipsis(index: Tuple[Any | Ellipsis], ndim: int):
    num_ellip = index.count(Ellipsis)
    assert num_ellip <= 1, f"Invalid index {index}"
    if num_ellip == 0:
        return index

    i = index.index(Ellipsis)
    completed = (
        list(index[:i]) + [slice(None)] * (ndim - len(index) + 1) + list(index[i + 1 :])
    )
    return tuple(completed)


def _parse_index(index: IndexLike, shape: Tuple[int]) -> Tensor:
    if not isinstance(index, tuple):
        index = (index,)
    index = _complete_ellipsis(index, ndim=len(shape))

    nindex = len(index)

    nindex_notnan = len([idx for idx in index if idx is not None])
    stride = int(np.prod(shape[nindex_notnan:]))

    ids = torch.tensor([0], dtype=torch.long)
    slice_shape = []
    dim = len(index) - 1
    shape_dim = nindex_notnan - 1
    for dim in range(nindex - 1, -1, -1):
        if index[dim] is None:
            slice_shape.append(1)
            continue

        index_at_dim = index[dim]
        size_at_dim = shape[shape_dim]
        if isinstance(index_at_dim, int):
            ids = ids + index_at_dim * stride
        elif isinstance(index_at_dim, slice):
            offsets = torch.arange(size_at_dim)[index_at_dim] * stride
            ids = (offsets[:, None] + ids[None, :]).reshape(-1)
            slice_shape.append(len(offsets))
        elif isinstance(index_at_dim, list):
            offsets = torch.tensor(index_at_dim) * stride
            ids = (offsets[:, None] + ids[None, :]).reshape(-1)
            slice_shape.append(len(offsets))
        elif isinstance(index_at_dim, Tensor):
            assert index_at_dim.ndim == 1, (
                f"Index at dimension {dim} should be 1-dimensional, "
                f"not {index_at_dim.ndim}-d."
            )
            ids = ids.to(index_at_dim.device)
            offsets = index_at_dim * stride
            ids = (offsets[:, None] + ids[None, :]).reshape(-1)
            slice_shape.append(len(offsets))
        else:
            raise TypeError(
                f"Index at dimension {dim} should be "
                f"a `int`, a `slice`, or a `Tensor`, not {type(index_at_dim)}"
            )

        stride *= size_at_dim
        shape_dim -= 1

    slice_shape = list(reversed(slice_shape))
    return ids.reshape(slice_shape)


def _replace_minus_one(shape: Tuple[int], num_elements: int):
    num_minus_one = shape.count(-1)
    assert num_minus_one <= 1, f"Invalid shape {shape}"
    if num_minus_one == 0:
        return shape

    otherdim = int(np.prod([size for size in shape if size >= 1]))
    inferred = num_elements // otherdim
    assert (
        inferred * otherdim == num_elements
    ), f"Invalid new shape {shape} for {num_elements} elements"
    i = shape.index(-1)
    replaced = list(shape[:i]) + [inferred] + list(shape[i + 1 :])
    return tuple(replaced)


def _parse_shape(shape, num_elements):
    if len(shape) == 1:
        assert isinstance(shape, Iterable)
        shape = tuple(shape[0])
    shape = _replace_minus_one(shape, num_elements=num_elements)
    given_num_elements = int(np.prod(shape))
    assert (
        given_num_elements == num_elements
    ), f"Inconsistent shape: should have {num_elements} elements, not {given_num_elements}."
    return shape

