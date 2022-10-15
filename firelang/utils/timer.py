from typing import Mapping
from time import time
from functools import wraps
from torch.cuda import synchronize

__all__ = ["Timer", "elapsed"]


class _Timer:
    t0: float
    elapsed: Mapping[str, float]

    def __init__(self, elapsed: Mapping, tag: str):
        self.elapsed = elapsed
        self.tag = tag

    def __enter__(self):
        self.t0 = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.update()

    def __call__(self, func):
        @wraps(func)
        def with_timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return with_timer

    def update(self):
        self.elapsed[self.tag] = self.elapsed.get(self.tag, 0) + time() - self.t0
        self.t0 = time()


class Timer(_Timer):
    def __init__(self, elapsed: Mapping, tag: str, sync_cuda=True):
        _Timer.__init__(self, elapsed, tag)
        self.sync_cuda = sync_cuda

    def __enter__(self):
        if self.sync_cuda:
            synchronize()
        self.t0 = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sync_cuda:
            synchronize()
        self.update()


elapsed = {}
