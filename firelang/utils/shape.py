from __future__ import annotations
from typing import Tuple, Iterable, Union
import numpy as np

ShapeLike = Union[int, Tuple[int]]

def check_shape_consistency(shape1: Tuple[int], shape2: Tuple[int]):
    assert len(shape1) == len(
        shape2
    ), f"Shape inconsistent in number of dimension between {shape1} and {shape2}."
    for d1, d2 in zip(shape1, shape2):
        if d1 != 1 and d2 != 1 and d1 != d2:
            raise ValueError(f"Inconsistent shape: {shape1} and {shape2}")
    return True


def parse_shape(shape, num_elements):
    if len(shape) == 1:
        shape = shape[0]
    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, Iterable):
        shape = tuple(shape)
    else:
        raise TypeError(f"Invalid shape {shape}")
    shape = _replace_minus_one(shape, num_elements=num_elements)
    given_num_elements = int(np.prod(shape))
    assert (
        given_num_elements == num_elements
    ), f"Inconsistent shape: should have {num_elements} elements, not {given_num_elements}."
    return shape


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
