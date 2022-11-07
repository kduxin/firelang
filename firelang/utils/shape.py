from typing import Tuple

def check_shape_consistency(shape1: Tuple[int], shape2: Tuple[int]):
    assert len(shape1) == len(
        shape2
    ), f"Shape inconsistent in number of dimension between {shape1} and {shape2}."
    for d1, d2 in zip(shape1, shape2):
        if d1 != 1 and d2 != 1 and d1 != d2:
            raise ValueError(f"Inconsistent shape: {shape1} and {shape2}")