from __future__ import annotations
from typing import List, Tuple
import torch
from torch import Tensor
from torch.nn import Parameter
from .base import Measure
from firelang.utils.limits import parse_rect_limits
from firelang.utils.index import normalize_index
from firelang.utils.shape import check_shape_consistency
from firelang.function import Functional

__all__ = [
    "DiracMixture",
]


class DiracMixture(Measure):
    _x: Parameter
    _m: Parameter

    def __init__(
        self,
        dim: int,
        k: int,
        limits: float | Tuple[float, float] | List[Tuple[float, float]] | Tensor = None,
        mfix: bool = False,
        signed: bool = False,
        shape: Tuple[int] = (1,),
    ):
        Measure.__init__(self, locals())
        if limits is None:
            self._x = Parameter(torch.randn(*shape, k, dim, dtype=torch.float32))
        else:
            if isinstance(limits, Tensor):
                assert (
                    limits.ndim == 2 and limits.shape[0] == dim and limits.shape[1] == 2
                ), f"Invalid limits: {limits}"
            else:
                limits = torch.tensor(
                    parse_rect_limits(limits, dim), dtype=torch.float32
                )  # (dim, 2)
            ranges = (limits[:, 1] - limits[:, 0])[None, None]  # (1, 1, dim)
            starts = limits[:, 0][None, None]  # (1, 1, dim)
            self._x = Parameter(
                torch.rand(*shape, k, dim, dtype=torch.float32) * ranges + starts
            )
        self._m = 1.0 if mfix else Parameter(torch.ones(*shape, k, dtype=torch.float32))

        self.dim = dim
        self.k = k
        self.limits = limits
        self.mfix = mfix
        self.shape = shape
        self.signed = signed

    def integral(
        self,
        func: Functional,
        cross: bool = False,
        batch_size: int = 1000000,
    ) -> Tensor:

        m, x = self.m, self.x
        if not cross:
            func = func.vmap()  # (1, ...shape)
            x = x.permute(-2, *range(x.ndim - 2), -1)  # (k, ...shape, dim)
            if isinstance(m, Tensor):
                m = m.permute(-1, *range(m.ndim - 1))  # (k, ...shape)
            fx = func(x) * m  # (k, ...shape)
            fx = fx.sum(dim=0)
        else:
            assert check_shape_consistency(self.shape[:-1], func.shape[:-1])
            batch_shape = self.shape[:-1]
            msize = self.shape[-1]
            fsize = func.shape[-1]

            func = func.vmap(2)  # (1, 1, ...batch_shape, fsize)
            x = x.permute(-2, -3, *range(x.ndim - 3), -1).unsqueeze(-2)
            # x: (k, msize, ...batch_shape, 1, dim)
            if isinstance(m, Tensor):
                m = m.permute(-1, -2, *range(m.ndim - 2)).unsqueeze(-1)
                # m: (k, msize, ...batch_shape, 1)

            size = fsize * self.k
            nrow_per_batch = (batch_size + size - 1) // size
            fx = []
            for i in range(0, msize, nrow_per_batch):
                _x = x if x.shape[1] == 1 else x[:, i : i + nrow_per_batch, ...]
                _m = m if m.shape[1] == 1 else m[:, i : i + nrow_per_batch, ...]
                _fx = func(_x) * _m  # (k, nrow_per_batch, ...batch_shape, fsize)
                fx.append(_fx)
            fx = torch.cat(fx, dim=1)  # (k, msize, ...batch_shape, fsize)
            fx = fx.sum(dim=0)  # (msize, ...batch_shape, fsize)
            fx = fx.permute(
                *range(1, fx.ndim - 1), -1, 0
            )  # (...batch_shape, fsize, msize)
        return fx

    def get_x(self):
        # _x: (*shape, k, dim)
        if self.limits is None:
            return self._x
        else:
            limits = self.limits.to(self.detect_device())
            ranges = (limits[:, 1] - limits[:, 0])[None, None]  # (1, 1, dim)
            _x = self._x / (ranges / 2)
            _x = torch.tanh(_x)
            _x = _x * (ranges / 2)
            return _x

    @property
    def x(self):
        return self.get_x()

    def get_m(self):
        if isinstance(self._m, Tensor):
            if self.signed:
                return self._m
            else:
                return self._m.abs()
        else:
            return self._m

    @property
    def m(self):
        return self.get_m()

    def __repr__(self):
        segs = [f"DiracMixture(shape={self.shape}, k={self.k}"]
        if self.mfix:
            segs.append(f", m=1.0")

        if self.limits is not None:
            limits = self.limits.data.cpu().numpy().tolist()
        else:
            limits = None
        if limits is not None:
            segs.append(f", limits={limits}")
        segs.append(")")
        return "".join(segs)

    def _parameter_shape_hash(self):
        hsh = Measure._parameter_shape_hash(self)
        hsh += hash(self.mfix)
        hsh += hash(self.limits)
        return hash(hsh)

    def unsqueeze(self, dim: int) -> DiracMixture:
        dim = normalize_index(dim, len(self.shape))
        new_shape = (*self.shape[:dim], 1, *self.shape[dim + 1 :])
        new = DiracMixture(
            dim=self.dim,
            k=self.k,
            limits=self.limits,
            mfix=self.mfix,
            shape=new_shape,
            signed=self.signed,
        )

        """ substitute _x """
        _x = self._x.unsqueeze(dim)
        if new._x.shape != _x.shape:
            new._x.requires_grad_(False)
            new._x = Parameter(torch.empty_like(_x))
        new._x.requires_grad_(False)
        new._x.copy_(_x)

        """ substitute _m """
        if isinstance(new._m, Tensor):
            _m = self.m
            _m = _m.reshape(*new_shape, self.k)
            new._m.requires_grad_(False)
            new._m.copy_(_m)

        return new

    def squeeze(self, dim: int) -> DiracMixture:
        dim = normalize_index(dim, len(self.shape))
        assert (
            self.shape[dim] == 1
        ), f"Unable to squeeze a dimension of size {self.shape[dim]}."
        new_shape = (*self.shape[: dim - 1], *self.shape[dim + 1 :])
        new = DiracMixture(
            dim=self.dim,
            k=self.k,
            limits=self.limits,
            mfix=self.mfix,
            shape=new_shape,
            signed=self.signed,
        )

        """ substitute _x """
        _x = self._x.squeeze(dim)
        if new._x.shape != _x.shape:
            new._x.requires_grad_(False)
            new._x = Parameter(torch.empty_like(_x))
        new._x.requires_grad_(False)
        new._x.copy_(_x)

        """ substitute _m """
        if isinstance(new._m, Tensor):
            _m = self.m
            _m = _m.reshape(*new_shape, self.k)
            new._m.requires_grad_(False)
            new._m.copy_(_m)

        return new

    def split(self, n_heads: int) -> DiracMixture:
        assert (
            self.k % n_heads == 0
        ), f"K ({self.k}) must be multiple of the number of heads ({n_heads})."
        new_k = self.k // n_heads
        new_shape = (n_heads, *self.shape)
        new = DiracMixture(
            dim=self.dim,
            k=new_k,
            limits=self.limits,
            mfix=self.mfix,
            shape=new_shape,
            signed=self.signed,
        )

        """ substitute _x """
        xshape = self._x.shape[:-2]
        _x = self._x.reshape(*xshape, n_heads, new_k, self.dim)
        _x = _x.permute(
            -3, *range(len(self.shape)), -2, -1
        ).contiguous()  # (n_heads, ...shape, new_k, dim)
        if new._x.shape != _x.shape:
            new._x.requires_grad_(False)
            new._x = Parameter(torch.empty_like(_x))
        new._x.requires_grad_(False)
        new._x.copy_(_x)

        """ substitute _m """
        if isinstance(new._m, Tensor):
            _m = self.m
            _m = _m.reshape(*self.shape, n_heads, self.k // n_heads)
            _m = _m.permute(-2, *range(len(self.shape)), -1)  # (n_heads, ...shape, k)
            new._m.requires_grad_(False)
            new._m.copy_(_m)
        return new

    def concat(self, dim: int = 0) -> DiracMixture:
        dim = normalize_index(dim, len(self.shape))
        n_heads = self.shape[dim]
        new_k = self.k * n_heads
        new_shape = (*self.shape[:dim], *self.shape[dim + 1 :])

        new = DiracMixture(
            dim=self.dim,
            k=new_k,
            limits=self.limits,
            mfix=self.mfix,
            shape=new_shape,
            signed=self.signed,
        )

        """ substitute _x """
        _x = self._x  # (...shape_before, n_heads, ...shape_after, k, field_dim)
        _x = _x.permute(
            *range(dim), *range(dim + 1, len(self.shape)), dim, -2, -1
        )  # (...shape_before, ...shape_after, n_heads, k, field_dim)
        _x = _x.reshape(*_x.shape[:-3], new_k, self.dim)
        if new._x.shape != _x.shape:
            new._x.requires_grad_(False)
            new._x = Parameter(torch.empty_like(_x))
        new._x.requires_grad_(False)
        new._x.copy_(_x)

        """ substitute _m """
        if isinstance(new._m, Tensor):
            _m = self.m  # (...shape_before, n_heads, ...shape_after, k)
            _m = _m.permute(
                *range(dim), *range(dim + 1, len(self.shape)), dim, -1
            )  # (...shape_before, ...shape_after, n_heads, k)
            _m = _m.reshape(*new_shape, new_k)
            new._m.requires_grad_(False)
            new._m.copy_(_m)
        return new

    def share_supports(self, dim: int) -> DiracMixture:
        """
        Warning: this function does not support automatic detection / reduction
        of duplicated supports. So you should only call this function once.
        
        Args:
            dim: the axis along which the Measure supports are shared.
        
        Returns:
            DiracMixture: (...shape) of `n*k` mixtures, where `n` is the \
                size at dimension `dim`.
        """
        dim = normalize_index(dim, len(self.shape))

        n = self.shape[dim]
        k = self.k
        nk = n * k

        new = DiracMixture(
            dim=self.dim,
            k=nk,
            limits=self.limits,
            mfix=False,
            shape=self.shape,
            signed=self.signed,
        )

        """ substitute _x: (..., n, ..., k) -> (..., n, ..., nk) """
        _x: Tensor = self._x
        xshape = _x.shape
        xdims = len(xshape)
        dimorder = (
            *range(dim),
            *range(dim + 1, len(self.shape)),
            dim,
            *range(len(self.shape), xdims),
        )
        _x = _x.permute(dimorder)  # (...shape_before, ...shape_after, n, k, field_dim)

        common_x = _x.reshape(
            *xshape[:dim], 1, *xshape[dim + 1 : -2], nk, self.dim
        )  # (...shape_before, 1, ...shape_after, nk, field_dim)

        new._x.requires_grad_(False)
        new._x = Parameter(torch.empty_like(common_x))
        new._x.requires_grad_(False)
        new._x.copy_(common_x)

        """ substitute _m: (..., n, ..., k) -> (..., n, ..., nk) """

        if not isinstance(self._m, Tensor):
            _m = torch.ones(
                *self.shape, self.k, dtype=torch.float32, device=self._x.device
            )
        else:
            _m = self.m
        ids = [slice(None)] * len(_m.shape)
        ids[dim] = torch.arange(n).reshape(n, 1)
        ids[-1] = torch.arange(nk).reshape(n, k)

        newm = torch.zeros(*self.shape, nk, dtype=_m.dtype, device=_m.device)
        if dim + 1 == len(ids) - 1:  # dim is the second to last dimension
            newm[ids] = _m
        else:
            _m = _m.permute(
                dim, len(_m.shape) - 1, *range(dim), *range(dim + 1, len(_m.shape) - 1)
            )
            newm[ids] = _m
        new._m.requires_grad_(False)
        new._m.copy_(newm)

        return new

    def linear(self, A: Tensor) -> DiracMixture:
        """Weighted linear combination of the measures along the last dimension,
        which must have already shared the same set of `x`.

        Args:
            A (Tensor): (..., out_dim, in_dim), where out_dim == in_dim

        Returns:
            DiracMixture: (...shape[:-1], out_dim)
        """
        assert A.shape[-1] == self.shape[-1]
        new_shape = (*self.shape[:-1], A.shape[-2])

        new = DiracMixture(
            dim=self.dim,
            k=self.k,
            limits=self.limits,
            mfix=False,
            shape=new_shape,
            signed=self.signed,
        )

        """ substitute _x """
        if new._x.shape != self._x.shape:
            new._x.requires_grad_(False)
            new._x = Parameter(torch.empty_like(self._x))
        new._x.requires_grad_(False)
        new._x.copy_(self._x)

        """ substitute _m """
        if self.mfix:
            _m = torch.ones(
                *self.shape, self.k, dtype=torch.float32, device=self._x.device
            )
        else:
            _m = self.m  # (...shape[:-1], shape[-1] == in_dim, k)
        _m = (
            _m.unsqueeze(-3)  # (...shape[:-1], 1, in_dim, k)
            * A.unsqueeze(-1)  # (...shape[:-1], out_dim, in_dim, 1)
        ).sum(
            dim=-2
        )  # (...shape[:-1], out_dim, k)
        new._m.requires_grad_(False)
        new._m.copy_(_m)

        return new

    def __add__(self, other: DiracMixture) -> DiracMixture:
        assert self.shape == other.shape
        assert (self._x == other._x).all()

        new = DiracMixture(
            dim=self.dim,
            k=self.k,
            limits=self.limits,
            mfix=False,
            shape=self.shape,
            signed=self.signed,
        )

        if self.mfix:
            _m1 = torch.ones(
                *self.shape, self.k, dtype=torch.float32, device=self._x.device
            )
        else:
            _m1 = self.m

        if other.mfix:
            _m2 = torch.ones(
                *other.shape, other.k, dtype=torch.float32, device=other._x.device
            )
        else:
            _m2 = other.m

        """ substitute _x """
        if new._x.shape != self._x.shape:
            new._x.requires_grad_(False)
            new._x = Parameter(torch.empty_like(self._x))
        new._x.requires_grad_(False)
        new._x.copy_(self._x)

        """ substitute _m """
        new._m.requires_grad_(False)
        new._m.copy_(_m1 + _m2)

        return new
