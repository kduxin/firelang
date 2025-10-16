from typing import Tuple
import torch
from torch import Tensor, isnan
from ..base import Measure
from ..dirac import DiracMixture


def sinkhorn(
    m1: Measure,
    m2: Measure,
    reg: float = 0.1,
    max_iter: int = 20,
    p: float = 2.0,
    tau: float = 1e3,
    stop_threshold: float = 1e-3,
):
    assert isinstance(m1, DiracMixture)
    assert isinstance(m2, DiracMixture)

    device = m1.detect_device()
    batch_size = m1.stack_size
    k1, k2 = m1.k, m2.k
    xw, yw = m1.m, m2.m
    if not isinstance(xw, Tensor):
        xw = torch.ones(batch_size, k1, dtype=torch.float32, device=device) * xw
    if not isinstance(yw, Tensor):
        yw = torch.ones(batch_size, k2, dtype=torch.float32, device=device) * yw

    # s = SinkhornDistanceStablized(
    #     reg=reg,
    #     max_iter=max_iter,
    #     reduction="none",
    #     p=p,
    #     tau=tau,
    #     stop_threshold=stop_threshold,
    # )

    s = SinkhornDistance(
        reg=reg, max_iter=max_iter, reduction="none", p=p, stop_threshold=stop_threshold
    )
    distance = s(m1.x, m2.x, xw, yw)

    return distance


class SinkhornDistance:
    def __init__(self, reg, max_iter, reduction="none", p=2.0, stop_threshold=1e-3):
        self.reg = reg
        self.max_iter = max_iter
        self.reduction = reduction
        self.p = p
        self.stop_threshold = stop_threshold

    def __call__(self, x: Tensor, y: Tensor, xw: Tensor, yw: Tensor) -> Tuple:
        """_summary_

        Args:
            x (Tensor): (*batch_size, n1, dim)
            y (Tensor): (*batch_size, n2, dim)
            xw (Tensor): (*batch_size, n1)
            yw (Tensor): (*batch_size, n2)

        Returns:
            - Tuple: (distance, pi, C)
                - distance (Tensor): (*batch_size,)
        """
        device = x.device
        dim = x.shape[-1]
        n1, n2 = x.shape[-2], y.shape[-2]
        batch_sizes = x.shape[:-2]
        assert dim == y.shape[-1]
        assert n1 == xw.shape[-1]
        assert n2 == yw.shape[-1]
        assert batch_sizes == y.shape[:-2] == xw.shape[:-1] == yw.shape[:-1]

        xw = xw / xw.sum(-1, keepdim=True)  # (*batch_size, n1)
        yw = yw / yw.sum(-1, keepdim=True)  # (*batch_size, n2)

        cost: Tensor = self._cost_matrix(
            x, y, p=self.p
        )  # (*batch_size, n1, n2) Wasserstein cost function

        # both marginals are fixed with equal weights
        u = torch.zeros(
            *batch_sizes, n1, dtype=torch.float32, device=device
        )  # (*batch_size, n1)
        v = torch.zeros(
            *batch_sizes, n2, dtype=torch.float32, device=device
        )  # (*batch_size, n2)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        # Stopping criterion

        # Sinkhorn iterations
        for it in range(self.max_iter):
            u1 = u  # (batch_size, n1) useful to check the update
            u = (
                self.reg
                * (torch.log(xw + 1e-8) - torch.logsumexp(self.M(cost, u, v), dim=-1))
                + u
            )  # (*batch_size, n1)
            v = (
                self.reg
                * (
                    torch.log(yw + 1e-8)
                    - torch.logsumexp(self.M(cost, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )  # (*batch_size, n2)

            errs = (u - u1).abs().sum(-1)
            err = torch.quantile(errs, 0.99).item()
            if err <= self.stop_threshold:
                break

        # print(f"Stop at iter: {it}")

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        M = self.M(cost, U, V)
        plan = torch.exp(M)  # (*batch_size, n1, n2)
        # Sinkhorn distance
        distance = torch.sum(plan * cost, dim=(-2, -1))  # (*batch_size,)

        if self.reduction == "mean":
            distance = distance.mean()
        elif self.reduction == "sum":
            distance = distance.sum()
        elif self.reduction in ["none", None]:
            pass
        else:
            raise ValueError(self.reduction)

        return distance

    def M(self, cost: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Modified cost for logarithmic updates
        $M_{ij} = (-cost_{ij} + u_i + v_j) / reg$

        Args:
            cost (Tensor): (*batch_size, n1, n2)
            u (Tensor): (*batch_size, n1)
            v (Tensor): (*batch_size, n2)

        Returns:
            Tensor: (*batch_size, n1, n2)
        """
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.reg

    @staticmethod
    def _cost_matrix(x: Tensor, y: Tensor, p: float = 2.0) -> Tensor:
        """p-Norm

        Args:
            x (Tensor): (batch_size, n1, dim)
            y (Tensor): (batch_size, n2, dim)
            p (float, optional): order of norm. Defaults to 2.

        Returns:
            Tensor: (batch_size, n1, n2) p-Norm
        """
        x_col = x.unsqueeze(-2)  # (batch_size, n1, 1, dim)
        y_lin = y.unsqueeze(-3)  # (batch_size, 1, n2, dim)
        cost = torch.sum(torch.abs(x_col - y_lin) ** p, -1)  # (batch_size, n1, n2)
        return cost


class SinkhornDistanceStablized:
    def __init__(
        self,
        reg: float,
        max_iter: int,
        reduction: str = "none",
        p: float = 2.0,
        tau: float = 1e3,
        stop_threshold=1e-3,
    ):
        self.reg = reg
        self.max_iter = max_iter
        self.reduction = reduction
        self.p = p
        self.tau = tau
        self.stop_threshold = stop_threshold

    def __call__(self, x: Tensor, y: Tensor, xw: Tensor, yw: Tensor) -> Tuple:
        """_summary_

        Args:
            x (Tensor): (*batch_size, n1, dim)
            y (Tensor): (*batch_size, n2, dim)
            xw (Tensor): (*batch_size, n1)
            yw (Tensor): (*batch_size, n2)

        Returns:
            - Tuple: (distance, pi, C)
                - distance (Tensor): (*batch_size,)
        """
        device = x.device
        dim = x.shape[-1]
        n1, n2 = x.shape[-2], y.shape[-2]
        batch_sizes = x.shape[:-2]
        assert dim == y.shape[-1]
        assert n1 == xw.shape[-1]
        assert n2 == yw.shape[-1]
        assert batch_sizes == y.shape[:-2] == xw.shape[:-1] == yw.shape[:-1]

        a = xw / xw.sum(-1, keepdim=True)  # (*batch_size, n1)
        b = yw / yw.sum(-1, keepdim=True)  # (*batch_size, n2)

        cost: Tensor = self._cost_matrix(
            x, y, p=self.p
        )  # (*batch_size, n1, n2) Wasserstein cost function

        alpha = torch.zeros(*batch_sizes, n1, dtype=torch.float32, device=device)
        beta = torch.zeros(*batch_sizes, n2, dtype=torch.float32, device=device)
        u = torch.ones(*batch_sizes, n1, dtype=torch.float32, device=device) / n1
        v = torch.ones(*batch_sizes, n2, dtype=torch.float32, device=device) / n2

        def get_K(alpha, beta):
            return torch.exp(
                -(cost - alpha.unsqueeze(-1) - beta.unsqueeze(-2)) / self.reg
            )

        def get_Gamma(alpha, beta, u, v):
            return torch.exp(
                -(cost - alpha.unsqueeze(-1) - beta.unsqueeze(-2)) / self.reg
                + torch.log(u.unsqueeze(-1) + 1e-8)
                + torch.log(v.unsqueeze(-2) + 1e-8)
            )

        K = get_K(alpha, beta)  # (*batch_size, n1, n2)
        transp = K
        err = 1
        for ii in range(self.max_iter):
            uprev = u
            vprev = v

            # sinkhorn update
            v = b / torch.einsum("...ab,...a->...b", K, u)
            u = a / torch.einsum("...ab,...b->...a", K, v)

            if torch.max(torch.abs(u)) > self.tau or torch.max(torch.abs(v)) > self.tau:
                alpha = alpha + self.reg * torch.log(u + 1e-8)
                beta = beta + self.reg * torch.log(v + 1e-8)
                u = (
                    torch.ones(*batch_sizes, n1, dtype=torch.float32, device=device)
                    / n1
                )
                v = (
                    torch.ones(*batch_sizes, n2, dtype=torch.float32, device=device)
                    / n2
                )
                K = get_K(alpha, beta)

            transp = get_Gamma(alpha, beta, u, v)
            errs = torch.norm(torch.sum(transp, dim=-2) - b, -1)
            err = torch.quantile(errs, 0.99).item()
            if err <= self.stop_threshold:
                break

            if torch.isnan(u).any() or torch.isnan(v).any():
                print(f"Warning: Numerical errors at iteration {ii}")
                u = uprev
                v = vprev
                break

        else:
            print("Warning: Sinkhorn did not converge.")
            pass

        # print(f"Stop at iter: {ii}. err = {err}")

        Gamma = get_Gamma(alpha, beta, u, v)
        distance = (Gamma * cost).sum(dim=[-2, -1])
        return distance

    @staticmethod
    def _cost_matrix(x: Tensor, y: Tensor, p: float = 2.0) -> Tensor:
        """p-Norm

        Args:
            x (Tensor): (batch_size, n1, dim)
            y (Tensor): (batch_size, n2, dim)
            p (float, optional): order of norm. Defaults to 2.

        Returns:
            Tensor: (batch_size, n1, n2) p-Norm
        """
        x_col = x.unsqueeze(-2)  # (batch_size, n1, 1, dim)
        y_lin = y.unsqueeze(-3)  # (batch_size, 1, n2, dim)
        cost = torch.sum(torch.abs(x_col - y_lin) ** p, -1)  # (batch_size, n1, n2)
        return cost
