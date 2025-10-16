from __future__ import annotations
from typing import List, Tuple, Union, Any, Iterable
import numpy as np
import torch
from torch import Tensor
from .shape import ShapeLike

IndexLike = Union[
    int, slice, List, Tensor, None, Tuple[Union[int, slice, List, Tensor, None]]
]


def flatten_index(index: IndexLike, shape: Tuple[int]) -> Tensor:
    if not isinstance(index, Iterable):
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


def normalize_index(index: IndexLike, shape: ShapeLike):
    if not isinstance(shape, Iterable):
        shape = [shape]
    if not isinstance(index, Iterable):
        normalized = normalize_index([index], shape)
        return normalized[0]
    
    index = _complete_ellipsis(index, ndim=len(shape))
    assert len(index) == len(shape)
    normalized = []
    for dim, (index_at_dim, size_at_dim) in enumerate(zip(index, shape)):

        if isinstance(index_at_dim, int):
            assert -size_at_dim <= index_at_dim < size_at_dim
            if index_at_dim < 0:
                index_at_dim += size_at_dim
            normalized.append(index_at_dim)
        elif isinstance(index_at_dim, slice):
            index_at_dim = list(range(size_at_dim))[index_at_dim]
            normalized.append(index_at_dim)
        elif isinstance(index_at_dim, list):
            nonneg = []
            for idx in index_at_dim:
                assert -size_at_dim <= idx < size_at_dim
                if idx < 0:
                    idx += size_at_dim
                nonneg.append(idx)
            normalized.append(nonneg)
        elif isinstance(index_at_dim, Tensor):
            assert index_at_dim.ndim == 1, (
                f"Index at dimension {dim} should be 1-dimensional, "
                f"not {index_at_dim.ndim}-d."
            )
            assert (-size_at_dim <= index_at_dim).all() and (index_at_dim < size_at_dim).all()
            index_at_dim = index_at_dim.data.cpu().numpy()
            index_at_dim[index_at_dim < 0] += size_at_dim
            normalized.append(index_at_dim.tolist())
        else:
            raise TypeError(
                f"Index at dimension {dim} should be "
                f"a `int`, a `slice`, or a `Tensor`, not {type(index_at_dim)}"
            )

    return normalized