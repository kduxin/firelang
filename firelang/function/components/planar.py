from functools import partial
import torch
from torch import Tensor
from torch.nn import Parameter
from .common import identity, identity_deriv, sigmoid_deriv, tanh_deriv
from ..base import Functional

__all__ = [
    "PseudoPlanarTransform",
]


class PseudoPlanarTransform(Functional):
    def __init__(self, dim, activation="tanh", stack_size=1):
        Functional.__init__(self, locals())
        scale = 0.1 / dim**0.5
        self.v = Parameter(torch.randn(stack_size, dim).normal_(0, scale))
        self.b = Parameter(torch.randn(stack_size, 1).normal_(0, scale))
        self.u = Parameter(torch.randn(stack_size, dim).normal_(0, scale))
        # v: (stack_size, dim)
        # b: (stack_size)
        # u: (stack_size, dim)

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
        self.stack_size = stack_size

    @property
    def w(self):
        return self.v

    def forward(self, x: Tensor, cross=False) -> Tensor:
        """
        Parameters:
            x: (stack2, batch, dim)
            when cross=True, also possible to get (stack1, stack2, batch, dim)
        Returns:
            cross=False (stack1==stack2):
                x: (stack2, batch, dim)
            cross=True:
                x: (stack1, stack2, batch, dim)
        """
        if not cross:
            assert x.shape[0] == self.stack_size
            a = (
                torch.einsum("sbi,si->sb", x, self.v)  # (stack_size, batch)
                + self.b  # (stack_size, 1)
            )  # (stack_size, batch)
            fx = (
                x
                + self.act(a)[:, :, None]  # (stack_size, batch, 1)
                * self.u[:, None, :]  # (stack_size, 1, dim)
            )  # (stack_size, batch, dim)
        else:
            if x.ndim == 3:
                x = x[None]
            else:
                assert x.ndim == 4
                assert x.shape[0] == self.stack_size

            a = (
                torch.einsum("tsbi,ti->tsb", x, self.v)  # (stack_size, stack2, batch)
                + self.b[:, None, :]  # (stack_size, 1, 1)
            )  # (stack_size, stack2, batch)
            fx = (
                x
                + self.act(a)[:, :, :, None]  # (stack_size, stack2, batch, 1)
                * self.u[:, None, None, :]  # (stack_size, 1, 1, dim)
            )  # (stack_size, stack2, batch, dim)
        return fx

    def jacob(self, x, cross=False, return_fx=False):
        """
        Parameters:
            x: (stack2, batch, dim)
            when cross=True, also possible to get (stack1, stack2, batch, dim)
        Returns:
            cross=False (stack1==stack2):
                jacob: (stack2, batch, dim, dim)
            cross=True:
                jacob: (stack1, stack2, batch, dim, dim)
        """
        dim = x.shape[-1]
        I = torch.eye(dim, device=x.device, dtype=x.dtype)
        if not cross:
            assert x.shape[0] == self.stack_size
            I = I[None, None, :, :]

            a = (
                torch.einsum("sbi,si->sb", x, self.v)  # (stack_size, batch)
                + self.b  # (stack_size, 1)
            )  # (stack_size, batch)
            ad = self.actderiv(a)  # (stack_size, batch)
            jacob = (
                I
                + ad[:, :, None, None]  # (stack_size, batch, 1, 1)
                * self.u[:, None, :, None]  # (stack_size, 1, dim, 1)
                * self.v[:, None, None, :]  # (stack_size, 1, 1, dim)
            )  # (stack_size, batch, dim, dim)
            if return_fx:
                fx = (
                    x
                    + self.act(a)[:, :, None]  # (stack_size, batch, 1)
                    * self.u[:, None, :]  # (stack_size, 1, dim)
                )  # (stack_size, batch, dim)

        else:
            if x.ndim == 3:
                x = x[None]
            else:
                assert x.ndim == 4
                assert x.shape[0] == self.stack_size
            # x: (stack_size, stack, batch, dim)
            I = I[None, None, None, :, :]

            a = (
                torch.einsum("tsbi,ti->tsb", x, self.v)  # (stack_size, stack2, batch)
                + self.b[:, None, :]  # (stack_size, 1, 1)
            )  # (stack_size, stack2, batch)
            ad = self.actderiv(a)  # (stack_size, stack2, batch)
            jacob = (
                I
                + ad[:, :, :, None, None]  # (stack_size, stack2, batch, 1, 1)
                * self.u[:, None, None, :, None]  # (stack_size, 1, 1, dim, 1)
                * self.v[:, None, None, None, :]  # (stack_size, 1, 1, 1, dim)
            )  # (stack_size, stack2, batch, dim, dim)
            if return_fx:
                fx = (
                    x
                    + self.act(a)[:, :, :, None]  # (stack_size, stack2, batch, 1)
                    * self.u[:, None, None, :]  # (stack_size, 1, 1, dim)
                )  # (stack_size, stack2, batch, dim)

        if return_fx:
            return jacob, fx
        else:
            return jacob

    def jacdet(self, x: Tensor, cross=False, return_fx=False):
        """
        Parameters:
            x: (stack2, batch, dim)
            when cross=True, also possible to get (stack1, stack2, batch, dim)
        Returns:
            cross=False (stack1==stack2):
                jacob: (stack2, batch)
            cross=True:
                jacob: (stack1, stack2, batch)
        """
        u_dot_v = torch.einsum("si,si->s", self.u, self.v)  # (stack_size, )
        if not cross:
            assert x.shape[0] == self.stack_size
            # x: (stack_size, batch, dim)

            a = torch.einsum("sbi,si->sb", x, self.v) + self.b
            # a: (stack_size, batch)
            ad = self.actderiv(a)  # (stack_size, batch)
            jacdet = 1 + ad * u_dot_v[:, None]  # (stack_size, )  # (stack_size, batch)
            if return_fx:
                fx = (
                    x
                    + self.act(a)[:, :, None]  # (stack_size, batch, 1)
                    * self.u[:, None, :]  # (stack_size, 1, dim)
                )  # (stack_size, batch, dim)
        else:
            if x.ndim == 3:
                x = x[None]
            elif x.ndim == 4:
                assert x.shape[0] == self.stack_size
            else:
                raise ValueError(x.ndim)
            # x: (stack_size, stack2, batch, dim)

            a = (
                torch.einsum("stbi,si->stb", x, self.v)  # (stack_size, stack2, batch)
                + self.b[:, None, :]  # (stack_size, 1, 1)
            )  # (stack_size, stack2, batch)
            ad = self.actderiv(a)  # (stack_size, stack2, batch)
            jacdet = (
                1 + ad * u_dot_v[:, None, None]  # (stack_size, 1, 1)
            )  # (stack_size, stack2, batch)
            if return_fx:
                fx = (
                    x
                    + self.act(a)[:, :, :, None]  # (stack_size, stack2, batch, 1)
                    * self.u[:, None, None, :]  # (stack_size, 1, 1, dim)
                )  # (stack_size, stack2, batch, dim)

        if return_fx:
            return jacdet, fx
        else:
            return jacdet
