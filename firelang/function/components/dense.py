from typing_extensions import Literal
from functools import partial
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
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
        stack_size=1,
    ):
        Functional.__init__(self, locals())

        scale = 0.1 / (input_dim + output_dim) ** 0.5
        self.A = Parameter(
            torch.empty(stack_size, output_dim, input_dim).normal_(0, scale)
        )
        self.b = Parameter(torch.zeros(stack_size, output_dim))

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
        self.stack_size = stack_size
        self.activation = activation
        self.norm = norm

    def forward(self, x, cross=False) -> Tensor:
        """
        Parameters:
            x: (stack2, batch, input_dim)
            when cross=True, also possible to get (stack1, stack2, batch, dim)
        Returns:
            if cross == True:
                (stack1, stack2, batch, output_dim)
            elif cross == False (and stack == stack2):
                (stack1, batch, output_dim)
        """
        if cross == True:
            if x.ndim == 3:
                x = torch.einsum("tbj,sij->stbi", x, self.A) + self.b[:, None, None, :]
            elif x.ndim == 4:
                x = torch.einsum("stbj,sij->stbi", x, self.A) + self.b[:, None, None, :]
            else:
                raise ValueError(x.ndim)
        else:
            x = torch.einsum("sbj,sij->sbi", x, self.A) + self.b[:, None, :]
        xshape = x.shape

        x = self.normalizer(x.reshape(np.prod(xshape[:-1]), xshape[-1])).reshape(
            *xshape
        )
        x = self.act(x)
        return x

    def jacob(self, x) -> Tensor:
        """
        Parameters:
            x: (stack, batch, input_dim)
        Returns:
            jac: (stack, batch, output_dim, input_dim)
        """
        assert self.norm is None
        a = torch.einsum("sbj,sij->sbi", x, self.A) + self.b.unsqueeze(
            1
        )  # (stack, batch, output_dim)
        ad = self.actderiv(a)  # (stack, batch, output_dim)
        jacob = ad[..., None] * self.A.unsqueeze(
            1
        )  # (stack, batch, output_dim, input_dim)
        return jacob

    def jacdet(self, x) -> Tensor:
        raise NotImplementedError
