from __future__ import annotations
from typing import List, Tuple, Iterable

__all__ = [
    "parse_rect_limits",
]


def parse_rect_limits(
    limits: float | Tuple[float, float] | List[Tuple[float, float]],
    dim: int,
) -> List[Tuple[float, float]]:
    if not isinstance(limits, Iterable):
        return [[-limits, limits] for _ in range(dim)]
    elif not isinstance(limits[0], Iterable):
        assert len(limits) == 2
        return [limits for _ in range(dim)]
    else:
        assert len(limits) == dim
        return limits
