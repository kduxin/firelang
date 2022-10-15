from typing import List
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
        stack_size=1,
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
                    stack_size=stack_size,
                    **(layer_kwargs if i < len(dims) - 2 else last_layer_kwargs),
                )
                for i, (idim, odim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dims = dims
        self.stack_size = stack_size
        self.activation = activation
        self.norm = norm


class _MLPlanarInit(Functional):
    def __init__(self, dim, nlayers, stack_size=1, **planar_kwargs):
        Functional.__init__(self, locals())
        self.layers = ModuleList(
            [
                PseudoPlanarTransform(dim=dim, stack_size=stack_size, **planar_kwargs)
                for _ in range(nlayers)
            ]
        )

        self.stack_size = stack_size
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

    def forward(self, x, cross=False):
        """
        Args:
            if cross == False:
                x (Tensor): (stack1, batch, dim)
            elif cross == True:
                x (Tensor): (stack2, batch, dim) or (stack1, stack2, batch, dim)
        Returns:
            if cross == False:
                Tensor: (stack1, batch)
            elif cross == True:
                Tensor: (stack1, stack2, batch)
        """
        raise NotImplementedError


class _Forward(_MultiLayerBase):
    def forward(self, x, cross=False):
        for layer in self.layers:
            x = layer(x, cross=cross)
        assert x.shape[-1] == 1, (
            f"Output of the last layer should has only one dimension, "
            f"not {x.shape[-1]}."
        )
        return x.squeeze(-1)


class _Divergence(_MultiLayerBase):
    """Each layer should implement .jacob method."""

    def forward(self, x: torch.Tensor, cross=False):
        cumjacob: Tensor
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                jacob, x = layer.jacob(x, cross=cross, return_fx=True)
            else:  # last iter
                jacob = layer.jacob(x, cross=cross, return_fx=False)

            assert jacob.shape[-1] == jacob.shape[-2], (
                f"The Jacobian is non-square, " f"but has a shape {jacob.shape}."
            )

            if i == 0:
                cumjacob = jacob
            else:
                if not cross:
                    # jacob, cumjacob: (stack2, batch, dim, dim)
                    cumjacob = torch.einsum("sbij,sbjk->sbik", jacob, cumjacob)
                else:
                    # jacob, cumjacob: (stack1, stack2, batch, dim, dim)
                    cumjacob = torch.einsum("tsbij,tsbjk->tsbik", jacob, cumjacob)
        # take trace of jacobian
        div = torch.einsum("...ii->...", cumjacob)
        # (stack2, batch) or (stack1, stack2, batch)
        return div


class _Jacdet(_MultiLayerBase):
    """Each layer should implement .jacdet method."""

    def forward(self, x: torch.Tensor, cross=False):
        cumjacdet: Tensor = 1
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                jacdet, x = layer.jacdet(x, cross=cross, return_fx=True)
                # x: (stack2, batch, dim) or (stack1, stack2, batch, dim)
            else:
                jacdet = layer.jacdet(x, cross=cross, return_fx=False)

            cumjacdet = cumjacdet * jacdet
            # (stack2, batch) or (stack1, stack2, batch)
        return cumjacdet


class _Jaclogdet(_MultiLayerBase):
    """Each layer should implement .jacdet method."""

    def forward(self, x: torch.Tensor, cross=False, eps=LOGABS_EPS):
        cumjacdet = _Jacdet.forward(self, x, cross=cross)
        cumjaclogdet = (eps + cumjacdet.abs()).log()
        return cumjaclogdet