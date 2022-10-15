from __future__ import annotations
import firelang
from firelang.stack import StackingSlicing

__all__ = ["Measure"]


class Measure(StackingSlicing):
    def integral(self, func, cross=False, batch_size=None, sum=True):
        """
        cross=False:
            func: (stack, K, input_dim) |-> (stack, K)
            integral:                   |-> (stack, )
        cross=True:
            func: (measure_stack, n_component, input_dim) |-> (func_stack, measure_stack, K)
            integral:                                     |-> (func_stack, measure_stack)
        """
        raise NotImplementedError

    def __mul__(self, other: firelang.Functional):
        assert isinstance(other, firelang.Functional)
        assert self.stack_size == other.stack_size
        return self.integral(other)

    def __matmul__(self, other: firelang.Functional):
        assert isinstance(other, firelang.Functional)
        return self.integral(other, cross=True).T