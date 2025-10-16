from __future__ import annotations
import torch
from firelang.function import Functional
from firelang.stack import StackingSlicing

__all__ = ["Measure"]


class Measure(StackingSlicing):
    def integral(self, func: Functional, cross: bool = False, batch_size: int = None, sum: bool = True):
        """ Compute integral $\int f d\mu$
        Args:
            - func (firelang.Functional)
            - cross (bool, optional): _description_. Defaults to False.
            - batch_size (int, optional): _description_. Defaults to None.
            - sum (bool, optional): whether sum the integral values at separate locations. Defaults to True.

        Returns:
            - if cross == False: (...shape, size, dim)
            - else: (...shape, func_size, measure_size, dim)
        """
        raise NotImplementedError

    def __mul__(self, other: Functional):
        """Returns: Tensor of shape (stack, K)"""
        assert isinstance(other, Functional)
        assert self.shape == other.shape
        return self.integral(other)

    def __matmul__(self, other: Functional):
        """Returns: Tensor of shape (func_stack, measure_stack)"""
        assert isinstance(other, Functional)
        return torch.transpose(self.integral(other, cross=True), -2, -1)
