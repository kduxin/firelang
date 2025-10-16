from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from firelang.utils.shape import check_shape_consistency
from .common import identity, identity_deriv, sigmoid_deriv, tanh_deriv
from ..base import Functional

__all__ = [
    "PseudoPlanarTransform",
]


class PseudoPlanarTransform(Functional):
    def __init__(self, dim, activation="tanh", shape: int | Tuple[int] = (1,)):
        Functional.__init__(self, locals())

        scale = 0.1 / dim**0.5
        self.v = Parameter(torch.empty(*shape, dim).normal_(0, scale))
        self.b = Parameter(torch.empty(*shape).normal_(0, scale))
        self.u = Parameter(torch.empty(*shape, dim).normal_(0, scale))

        if activation is None:
            self.act, self.actderiv = identity, identity_deriv
        elif activation == "sigmoid":
            self.act, self.actderiv = torch.sigmoid, sigmoid_deriv
        elif activation == "tanh":
            self.act, self.actderiv = torch.tanh, tanh_deriv
        else:
            raise ValueError(activation)

        self.dim = dim
        self.activation = activation
        self.shape = shape

    def __mul__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """Compute f(x)

        Args:
            x (Tensor): (...shape, dim)

        Returns:
            Tensor: (...shape, dim)
        """
        fshape = self.shape
        (*xshape, dim) = x.shape
        check_shape_consistency(fshape, xshape)

        u, v, b = self.u, self.v, self.b

        a = torch.einsum("...i,...i->...", x, v) + b  # (...shape,)
        fx = x + self.act(a)[..., None] * u  # (...shape, dim)
        return fx

    def jacob(self, x: Tensor, return_fx: bool = False) -> Tensor:
        """Compute df(x) / dx

        Args:
            x (Tensor): (...shape, dim)
            return_fx (bool, optional): whether returns f(x) or not. Defaults to False.

        Returns:
            Tensor: (...shape, dim, dim)
        """

        fshape = self.shape
        (*xshape, dim) = x.shape
        check_shape_consistency(fshape, xshape)

        I = torch.eye(dim, device=x.device, dtype=x.dtype).reshape(
            *[1 for _ in xshape], dim, dim
        )

        u, v, b = self.u, self.v, self.b

        a = torch.einsum("...i,...i->...", x, v) + b
        ad = self.actderiv(a)
        jacob = (
            I + ad[..., None, None] * u[..., :, None] * v[..., None, :]
        )  # (fshape, dim, dim)

        if return_fx:
            fx = x + self.act(a)[..., None] * u  # (...shape, dim)
            return jacob, fx
        else:
            return jacob

    def jacdet(self, x: Tensor, return_fx: bool = False) -> Tensor:
        """Compute det(df(x) / dx)

        Args:
            x (Tensor): (...shape, dim)
            return_fx (bool, optional): whether returns f(x) or not. Defaults to False.

        Returns:
            Tensor: (...shape)
        """
        fshape = self.shape
        (*xshape, dim) = x.shape
        check_shape_consistency(fshape, xshape)

        u, v, b = self.u, self.v, self.b

        u_dot_v = torch.einsum("...i,...i->...", u, v)  # (...fshape,)
        a = torch.einsum("...i,...i->...", x, v) + b  # (...fshape,)
        ad = self.actderiv(a)  # (...fshape,)
        jacdet = 1 + ad * u_dot_v

        if return_fx:
            fx = x + self.act(a)[..., None] * u  # (...shape, dim)
            return jacdet, fx
        else:
            return jacdet

    def jacob_mul_vecs(
        self, x: Tensor, vecs: Tensor, return_fx: bool = False
    ) -> Tensor:
        """Compute (df(x) / dx) @ vec

        Args:
            x (Tensor): (...shape, dim)
            vecs (Tensor): (...shape, nvecs, dim)
            return_fx (bool, optional): whether returns f(x) or not. Defaults to False.

        Returns:
            Tensor: (...shape, nvecs, dim)
        """
        fshape = self.shape
        (*xshape, dim) = x.shape
        check_shape_consistency(fshape, xshape)

        u, v, b = self.u, self.v, self.b

        a = torch.einsum("...i,...i->...", x, v) + b  # (...fshape,)
        ad = self.actderiv(a)  # (...fshape,)

        v_dot_vecs = torch.einsum("...i,...ni->...n", v, vecs)  # (...fshape, nvecs)
        scalar = ad[..., None] * v_dot_vecs  # (...fshape, nvecs)

        jacob_mul_vecs = (
            vecs + scalar[..., None] * u[..., None, :]
        )  # (...fshape, nvecs, dim)

        if return_fx:
            fx = x + self.act(a)[..., None] * u  # (...shape, dim)
            return jacob_mul_vecs, fx
        else:
            return jacob_mul_vecs
