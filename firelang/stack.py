from __future__ import annotations
from typing import List, Tuple, Mapping, Set, Iterable, Any, Union
from copy import deepcopy
from collections import OrderedDict, defaultdict
import inspect
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ModuleDict
from .utils.index import IndexLike
from .utils.timer import Timer, elapsed

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

    @Timer(elapsed, "slice", relative=False)
    def __getitem__(self, index: IndexLike):
        new_shape = tuple(torch.empty(self.shape)[index.cpu()].shape)
        to: StackingSlicing = self.restack(new_shape)

        with Timer(elapsed, "copy", relative=False):
            # A parameter not listed in `unsliceable_params` should be
            # sliced and copied. Otherwise, the whole parameter is copied.
            for name, param in self.named_parameters(recurse=False):
                param_to = to.get_parameter(name)
                param_to.requires_grad_(False)
                if name in self.unsliceable_params:
                    param_to.copy_(param)
                else:
                    param_to.copy_(param[index])

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
                            entry[index]
                            if isinstance(entry, StackingSlicing)
                            else entry
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
                submod_from.shape = new_shape
                setattr(to, name, submod_from)

            to.shape = new_shape
        return to

    def detect_device(self):
        for m in self.modules():
            try:
                return next(m.parameters()).device
            except StopIteration:
                if hasattr(m, "_former_parameters"):
                    # `self` is an instance from torch.nn.parallel.replicate
                    fp = m._former_parameters
                    if len(fp):
                        return next(iter(fp.values())).device
        raise ValueError("Failed to detect the device.")

    def detect_dtype(self):
        for m in self.modules():
            try:
                return next(m.parameters()).dtype
            except StopIteration:
                if hasattr(m, "_former_parameters"):
                    # `self` is an instance from torch.nn.parallel.replicate
                    fp = m._former_parameters
                    if len(fp):
                        return next(iter(fp.values())).dtype
        raise ValueError("Failed to detect the dtype.")

    def _parameter_shape_hash(self):
        name_shapes = [(name, p.shape) for name, p in self.named_parameters()]
        return hash(tuple(name_shapes))

    @Timer(elapsed, "restack", relative=False)
    def restack(
        self,
        *shape: int | Tuple[int],
        use_cached: bool = True,
        max_cached_copies: int = 100,
    ):
        if len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = tuple(shape[0])

        device = self.detect_device()
        init_locals = self.init_locals
        if "device" in init_locals:
            init_locals["device"] = device

        tag = f"stacked/{self.__class__.__name__}-{self._parameter_shape_hash()}"
        if use_cached and shape in _cache[tag]:
            new = deepcopy(_cache[tag][shape])
        else:
            positional, keywords = self._recover_args_from_locals(
                locals_=init_locals,
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
                keywords[key] = value
            elif sign.kind == inspect.Parameter.VAR_POSITIONAL:
                positional.extend(value)
            elif sign.kind == inspect.Parameter.VAR_KEYWORD:
                keywords = {**keywords, **value}
        return positional, keywords

    @property
    def ndim(self) -> int:
        return len(self.shape)

