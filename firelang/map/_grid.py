from __future__ import annotations
from typing import List, Tuple, Iterable
import numpy as np
from numba import cuda
import torch
from torch import Tensor, dtype, device
from torch.nn import Parameter
from firelang.stack import StackingSlicing


class Grid(StackingSlicing):

    _gridvals: Tensor

    def __init__(
        self,
        dim_sizes: List[int],
        dtype: dtype = torch.float32,
        device: device = "cuda",
        shape: Tuple[int] = (1,),
    ):
        StackingSlicing.__init__(self, locals())
        self._gridvals = Parameter(
            torch.empty(*shape, *dim_sizes, dtype=dtype, device=device).normal_(0, 0.1)
        )
        self.dim_sizes = torch.tensor(dim_sizes, dtype=torch.long, device=device)
        self.ndim = len(dim_sizes)

    def detect_device(self):
        return self._gridvals.device

    @property
    def gridvals(self):
        return self._gridvals

    def slice_rectangle(
        self,
        corners: Tensor,
        rect_dim_sizes: int | List[int] | Tensor,
    ) -> Tensor:
        """Parallelized slicing of (hyper-)rectangles from the d-dim grid
        that are specified by `corners` and `rect_dim_sizes`.

        Args:
            - corners (Tensor): (...xshape, ndim). The i-th corner out of `xshape` corners
                is aligned with the i-th grid of self._gridvals.
            - rect_dim_sizes (int | List[int] | Tensor): rect_dim_sizes at each dimension of the rectangle. \
                If is a `int`, it gives the size at all dimensions.

            A rectangle at dimension d is located at:
            - If `rect_dim_sizes` is int: [corner : corner + rect_dim_sizes]
            - If `rect_dim_sizes` is List[int]: [corner : corner + rect_dim_sizes[d]]

        Returns:
            - Tensor: let each rectangle be represented by (n1, n2, ..., nd), returns
                a Tensor with shape: (...shape, n1, n2, ..., nd),
        """
        (*shape, ndim) = corners.shape
        size = int(np.prod(shape))
        device, dtype = corners.device, corners.dtype

        grid_dim_sizes = self.dim_sizes.to(device)

        if not isinstance(rect_dim_sizes, Iterable):
            rect_dim_sizes = [rect_dim_sizes] * len(ndim)
        else:
            assert len(rect_dim_sizes) == ndim
        if not isinstance(rect_dim_sizes, Tensor):
            rect_dim_sizes = torch.tensor(
                rect_dim_sizes,
                dtype=torch.long,
                device=device,
            )

        """ Compute the offsets for all grid points within the rectangle,
            which will be used in torch.take(...).
        """
        corners = corners.reshape(size, ndim)
        offsets = torch.zeros(
            size,
            int(np.prod(rect_dim_sizes.data.cpu().numpy())),
            dtype=torch.long,
            device=device,
        )
        BLOCKDIM_X = 512
        n_block = (size + BLOCKDIM_X - 1) // BLOCKDIM_X
        rectangle_offsets_in_grid_kernel[n_block, BLOCKDIM_X](
            cuda.as_cuda_array(corners),
            cuda.as_cuda_array(grid_dim_sizes),
            cuda.as_cuda_array(rect_dim_sizes),
            size,
            ndim,
            cuda.as_cuda_array(offsets),
        )
        offsets = offsets.reshape(*shape, *rect_dim_sizes.tolist())
        # (*shape, n1, n2, ..., nd)

        """ Consider additional offset caused by `shape` """
        grid_size = int(np.prod(grid_dim_sizes.data.cpu().numpy()))
        stack_offsets = (
            torch.arange(int(np.prod(self.shape)), dtype=torch.long, device=device)
            * grid_size
        ).reshape(*self.shape, *[1] * ndim)
        # (*self.shape, 1, ..., 1)
        offsets = offsets + stack_offsets

        rectangle_vals = self.gridvals.take(offsets)
        return rectangle_vals


@cuda.jit("void(int64[:,:], int64[:], int64[:], int32, int32, int64[:,:])")
def rectangle_offsets_in_grid_kernel(
    corners, grid_dim_sizes, rect_dim_sizes, batch_size, ndim, out
):
    """Compute the offsets of grid points from the beginning of the grid,
    where the grid points are those inside rectangles specified by `corners` and `rect_dim_sizes`.

    Args:
        - corner (CudaArray): (batch_size, ndim)
        - grid_dim_sizes (CudaArray): (ndim,)
        - rect_dim_sizes (CudaArray): (ndim,)
        - batch_size (int)
        - ndim (int)
        - out (CudaArray): (batch_size, n1*n2...*nd). Buffer for the output (offsets).
    """

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= batch_size:
        return

    rect_size = 1
    for d in range(ndim):
        rect_size *= rect_dim_sizes[d]

    for j in range(rect_size):
        rect_stride = 1
        grid_stride = 1
        offset = 0
        for d in range(ndim):
            d = ndim - 1 - d
            id_at_dim_d = (j // rect_stride) % rect_dim_sizes[d] + corners[i, d]
            offset += id_at_dim_d * grid_stride
            rect_stride *= rect_dim_sizes[d]
            grid_stride *= grid_dim_sizes[d]
        out[i, j] = offset
