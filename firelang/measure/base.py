from __future__ import annotations
import firelang
from firelang.stack import StackingSlicing

__all__ = ["Measure"]


class Measure(StackingSlicing):
    def integral(self, func: firelang.Functional, cross: bool = False, batch_size: int = None, sum: bool = True):
        """ Compute integral $\int f d\mu$
        Args:
            - func (firelang.Functional)
            - cross (bool, optional): _description_. Defaults to False.
            - batch_size (int, optional): _description_. Defaults to None.
            - sum (bool, optional): whether sum the integral values at separate locations. Defaults to True.

        Returns:
        - if cross == False:
            func should accept Tensor of shape (stack, K, input_dim) and produces (stack, K)
            - if sum == True: returns (stack,)
            - else:           returns (stack, K)
        - if cross == True:
            func should be able to accept both (measure_stack, n_component, input_dim)
            and (func_stack, measure_stack, n_component, input_dim), and produces (func_stack, measure_stack, K)
            - if sum == True: returns (func_stack, measure_stack)
            - else:           returns (func_stack, measure_stack, K)
        """
        raise NotImplementedError

    def __mul__(self, other: firelang.Functional):
        """Returns: Tensor of shape (stack, K)"""
        assert isinstance(other, firelang.Functional)
        assert self.stack_size == other.stack_size
        return self.integral(other, sum=False)

    def __matmul__(self, other: firelang.Functional):
        """Returns: Tensor of shape (func_stack, measure_stack)"""
        assert isinstance(other, firelang.Functional)
        return self.integral(other, cross=True).T
