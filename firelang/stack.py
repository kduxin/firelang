from typing import Mapping, Set, Iterable, Any
from copy import deepcopy
from collections import OrderedDict, defaultdict
import inspect
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


class StackingSlicing(Module):
    init_locals: Mapping[str, Any]
    unsliceable_params: Set[str]
    skip_keys = ["stack_size"]

    def __init__(
        self,
        locals_: Mapping[str, Any],
        unsliceable_params: Iterable[str] = [],
    ):
        Module.__init__(self)
        self.init_locals = locals_
        self._sanity_check()
        self.stack_size = locals_["stack_size"]
        self.unsliceable_params = set(unsliceable_params)

    def _sanity_check(self):
        assert (
            type(self) != StackingSlicing
        ), "StackingSlicing must be initialized from a subclass"

        assert "stack_size" in self.init_locals, \
            "A `StackingSlicing` subclass must accept `stack_size` as an " \
            "initialization argument."

    def __getitem__(self, ids: Tensor):
        """
        ids: 1d Tensor of torch.Long
        """
        assert ids.ndim == 1
        new_stack_size = len(ids)
        to: StackingSlicing = self.restack(new_stack_size)

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
                submod_from: Module = module[ids]
            elif isinstance(module, ModuleList):
                submod_from = ModuleList([entry[ids] for entry in module])
            elif isinstance(module, ModuleDict):
                submod_from = ModuleDict({key: entry[ids] for key, entry in module})
            else:
                submod_from: Module = module
            submod_from.stack_size = new_stack_size
            setattr(to, name, submod_from)

        to.stack_size = new_stack_size
        return to

    def detect_device(self):
        return next(iter(self.parameters())).device

    def _parameter_shape_hash(self):
        name_shapes = [(name, p.shape) for name, p in self.named_parameters()]
        return hash(tuple(name_shapes))

    def restack(
        self, stack_size: int, use_cached: bool = True, max_cached_copies: int = 100
    ):

        tag = f"stacked/{self.__class__.__name__}-{self._parameter_shape_hash()}"
        if use_cached and stack_size in _cache[tag]:
            new = deepcopy(_cache[tag][stack_size])
        else:
            positional, keywords = self._recover_args_from_locals(
                locals_=self.init_locals,
            )
            new = self.__class__(
                *positional,
                **keywords,
                stack_size=stack_size,
            )
            new = new.to(self.detect_device())
            new.stack_size = stack_size

            _cache[tag][stack_size] = new
            while len(_cache[tag]) > max_cached_copies:
                _cache[tag].popitem(last=False)  # pop the earliest

        return new

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

    def __len__(self):
        return self.stack_size
