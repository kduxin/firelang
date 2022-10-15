from typing import Iterable
from .base import Functional

__all__ = ["Sequential", "sequential"]


class Sequential(Functional):
    def __init__(self, funcs, stack_size=1):
        Functional.__init__(
            self,
            locals(),
            prev=[func.restack(stack_size) for func in funcs],
            operator=sequential,
        )


def sequential(other_funcs: Iterable[Functional], x, *args, **kwargs):
    for f in other_funcs:
        x = f(x, *args, **kwargs)
    return x
