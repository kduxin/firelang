from typing import Mapping
from numbers import Number
from time import time
from functools import wraps
from torch.cuda import synchronize

__all__ = ["Timer", "elapsed", "format_elapsed"]


class Elapsed(dict):
    def format(elapsed, thresh=1.0, level=99):
        elapsed = [(k, v) for k, v in elapsed.items() if isinstance(v, Number)]
        elapsed = sorted(elapsed, key=lambda x: x[0], reverse=False)

        """ Build a tree """
        root = ["", [], 0.0]  # Each node is (prefix, subnodes, total_elapsed)
        stack = [root]
        for tag, elap in elapsed:
            node = [tag, [], elap]

            while True:
                prefix, tree, total_elap = stack[-1]
                if tag.startswith(prefix):
                    break
                else:
                    stack.pop()

            tree.append(node)
            stack.append(node)

        """ Convert the tree to string """
        indent = 2

        def _format(node, curlevel, parent_tag=""):
            tag, childs, total_elapsed = node
            s = " " * (indent * curlevel) + tag[len(parent_tag) :]
            segs = [f"{s:<20}\t{total_elapsed:>8.2f}"]
            if curlevel < level:
                childs = sorted(
                    childs, key=lambda x: x[2], reverse=True
                )  # sort by elapsed
                cum_elapsed = 0.0
                for child in childs:
                    _, _, elap = child
                    if cum_elapsed / total_elapsed < thresh:
                        segs.extend(_format(child, curlevel + 1, tag))
                        cum_elapsed += elap
                    else:
                        s = " " * (indent * (curlevel + 1)) + "others"
                        remained_elapsed = total_elapsed - cum_elapsed
                        if remained_elapsed > 0.01:
                            segs.append(f"{s:<20}\t{remained_elapsed:>8.2f}")
                        break
            return segs

        segs = []
        for child in root[1]:
            seg = _format(child, 0)
            segs.extend(seg)
        return "\n".join(segs)


elapsed = Elapsed()


class _Timer:
    t0: float
    elapsed: Mapping[str, float]

    def __init__(self, elapsed: Mapping, tag: str, relative: bool = True):
        """
        Args:
            - elapsed (Mapping): where to store results of the timer.
            - tag (str): name of the timer.
            - relative (bool, optional): generate path-like tag when using nested Timers.
              Defaults to True.
        """
        self.elapsed = elapsed
        self.tag = self.original_tag = tag
        self.relative = relative

    def __enter__(self):
        self.t0 = time()

        if self.relative:
            self.original_prefix = self.elapsed.get("__prefix", "")
            # create temporary attributes
            self.tag = f"{self.original_prefix}/{self.original_tag}"
            self.elapsed["__prefix"] = self.tag

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.update()

        if self.relative:
            # recover original attributes
            self.tag = self.original_tag
            self.elapsed["__prefix"] = self.original_prefix

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
    def __init__(self, elapsed: Mapping, tag: str, sync_cuda=True, relative=True):
        _Timer.__init__(self, elapsed, tag, relative=relative)
        self.sync_cuda = sync_cuda

    def __enter__(self):
        if self.sync_cuda:
            synchronize()
        _Timer.__enter__(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sync_cuda:
            synchronize()
        _Timer.__exit__(self, exc_type, exc_val, exc_tb)
