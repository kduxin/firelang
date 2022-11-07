from __future__ import annotations
from typing import List, Tuple
from typing_extensions import Literal
import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from firelang.function.base import Functional
from . import (
    Perceptron,
    PseudoPlanarTransform,
)

__all__ = [
    "_MultiLayerBase",
    "_MLPInit",
    "_MLPlanarInit",
    "_Forward",
    "_Divergence",
    "_Jacdet",
    "_Jaclogdet",
]


LOGABS_EPS = 1e-1

""" ------------- init ------------- """


class _MLPInit(Functional):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation="sigmoid",
        norm: Literal[None, "batch", "layer"] = None,
        shape: int | Tuple[int] = (1,),
    ):
        Functional.__init__(self, locals())
        dims = [input_dim] + hidden_dims
        layer_kwargs = {"activation": activation, "norm": norm}
        last_layer_kwargs = {"activation": None, "norm": None}
        self.layers = ModuleList(
            [
                Perceptron(
                    input_dim=idim,
                    output_dim=odim,
                    shape=shape,
                    **(layer_kwargs if i < len(dims) - 2 else last_layer_kwargs),
                )
                for i, (idim, odim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dims = dims
        self.shape = shape
        self.activation = activation
        self.norm = norm


class _MLPlanarInit(Functional):
    def __init__(self, dim, nlayers, shape=(1,), **planar_kwargs):
        Functional.__init__(self, locals())
        self.layers = ModuleList(
            [
                PseudoPlanarTransform(dim=dim, shape=shape, **planar_kwargs)
                for _ in range(nlayers)
            ]
        )

        self.shape = shape
        self.dim = dim
        self.nlayers = nlayers
        self.planar_kwargs = planar_kwargs


""" -------------- Stacked forward pass -------------- """


class _MultiLayerBase:
    """
    Should has a .layers attribute that is a list of layers.
    Each layer should implement .forward method.
    """

    layers: List[Module]

    def forward(self, x):
        """
        Args:
            x (Tensor): (...shape, dim)
        Returns:
            Tensor: (...shape,)
        """
        raise NotImplementedError


class _Forward(_MultiLayerBase):
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        assert x.shape[-1] == 1, (
            f"Output of the last layer should has only one dimension, "
            f"not {x.shape[-1]}."
        )
        return x.squeeze(-1)


class _Divergence(_MultiLayerBase):
    """Each layer should implement .jacob method."""

    def forward(self, x: Tensor) -> Tensor:
        """Compute tr(df(x) / dx)

        Args:
            x (Tensor): (...shape, dim)

        Returns:
            Tensor: (...shape,)
        """
        cumjacob: Tensor
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                jacob, x = layer.jacob(x, return_fx=True)
            else:  # last iter
                jacob = layer.jacob(x, return_fx=False)

            assert jacob.shape[-1] == jacob.shape[-2], (
                f"The Jacobian is non-square, " f"but has a shape {jacob.shape}."
            )

            if i == 0:
                cumjacob = jacob
            else:
                cumjacob = torch.einsum("...ij,...jk->...ik", jacob, cumjacob)
        # take trace of jacobian
        div = torch.einsum("...ii->...", cumjacob)

        dim = x.shape[-1]
        div = div / dim
        return div


        return div


class _Jacdet(_MultiLayerBase):
    """Each layer should implement .jacdet method."""

    def forward(self, x: torch.Tensor):
        cumjacdet: Tensor = 1
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                jacdet, x = layer.jacdet(x, return_fx=True)
                # x: (...shape, dim)
            else:
                jacdet = layer.jacdet(x, return_fx=False)
            # jacdet: (...shape,)

            cumjacdet = cumjacdet * jacdet  # (...shape,)
        return cumjacdet


class _Jaclogdet(_MultiLayerBase):
    """Each layer should implement .jacdet method."""

    def forward(self, x: torch.Tensor, eps=LOGABS_EPS):
        cumjacdet = _Jacdet.forward(self, x)
        cumjaclogdet = (eps + cumjacdet.abs()).log()
        return cumjaclogdet