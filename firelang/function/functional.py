"""
This file defines scalar-valued functionals.
"""
from .components import (
    _MLPInit,
    _MLPlanarInit,
)
from .components import (
    _Forward,
    _Divergence,
    _DivergenceViaQuadform,
    _Jacdet,
    _Jaclogdet,
)

__all__ = [
    "MLP",
    "MLPDiv",
    "MLPlanarDiv",
    "MLPlanarDivFast",
    "MLPlanarJacdet",
    "MLPlanarJaclogdet",
]


""" -------------- MLP --------------- """


class MLP(_Forward, _MLPInit):
    pass


class MLPDiv(_Divergence, _MLPInit):
    pass


""" -------------- Planar Transforms --------------- """


class MLPlanarDiv(_Divergence, _MLPlanarInit):
    pass


class MLPlanarDivFast(_DivergenceViaQuadform, _MLPlanarInit):
    pass


class MLPlanarJacdet(_Jacdet, _MLPlanarInit):
    pass


class MLPlanarJaclogdet(_Jaclogdet, _MLPlanarInit):
    pass
