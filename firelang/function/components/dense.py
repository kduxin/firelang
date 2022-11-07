from typing_extensions import Literal
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from firelang.utils.shape import check_shape_consistency
from .common import identity, identity_deriv, sigmoid_deriv, tanh_deriv
from ..base import Functional

__all__ = [
    "Perceptron",
]


class Perceptron(Functional):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation="sigmoid",
        norm: Literal[None, "batch", "layer"] = None,
        shape=(1,),
    ):
        Functional.__init__(self, locals())

        size = np.prod(shape)
        scale = 0.1 / (input_dim + output_dim) ** 0.5
        self.A = Parameter(torch.empty(size, output_dim, input_dim).normal_(0, scale))
        self.b = Parameter(torch.zeros(size, output_dim))

        if activation is None:
            self.act, self.actderiv = identity, identity_deriv
        elif activation == "sigmoid":
            self.act, self.actderiv = torch.sigmoid, sigmoid_deriv
        elif activation == "tanh":
            self.act, self.actderiv = torch.tanh, tanh_deriv
        else:
            raise ValueError(activation)

        if norm is None:
            self.normalizer = identity
        elif norm == "batch":
            self.normalizer = nn.BatchNorm1d(output_dim)
        elif norm == "layer":
            self.normalizer = nn.LayerNorm(output_dim)
        else:
            raise ValueError(norm)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shape = shape
        self.activation = activation
        self.norm = norm

    def forward(self, x) -> Tensor:
        """
        Parameters:
            x: (...shape, input_dim)
        Returns:
            (...shape, output_dim)
        """
        fshape = self.shape
        (*xshape, input_dim) = x.shape
        check_shape_consistency(fshape, xshape)

        output_dim = self.output_dim
        A = self.A.view(*fshape, output_dim, input_dim)
        b = self.b.view(*fshape, output_dim)

        x = torch.einsum("...i,...ji->...j", x, A) + b

        # normalization
        xshape = x.shape
        x = self.normalizer(x.reshape(np.prod(xshape[:-1]), xshape[-1])).reshape(
            *xshape
        )
        x = self.act(x)
        return x

    def jacob(self, x) -> Tensor:
        """
        Parameters:
            x: (...shape, input_dim)
        Returns:
            jacob: (...shape, output_dim, input_dim)
        """
        assert self.norm is None

        fshape = self.shape
        (*xshape, input_dim) = x.shape
        check_shape_consistency(fshape, xshape)

        output_dim = self.output_dim
        A = self.A.view(*fshape, output_dim, input_dim)
        b = self.b.view(*fshape, output_dim)

        a = torch.einsum("...i,...ji->...j", x, A) + b
        ad = self.actderiv(a)  # (...shape, output_dim)
        jacob = ad[..., None] * self.A  # (...shape, output_dim, input_dim)
        return jacob

    def jacdet(self, x) -> Tensor:
        raise NotImplementedError
