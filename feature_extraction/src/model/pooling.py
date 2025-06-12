import enum
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Union

from einops import rearrange, reduce


class PoolingFn(enum.Enum):
    """
    Enum class representing different pooling functions.

    Attributes:
        HEAD: Identity pooling.
        CLS: Class token pooling.
        GAP: Global Average Pooling (SPoC).
        MAC: Maximum Activation of Convolutions.
        GEM: Generalized Mean Pooling.
        RMAC: Regional Maximum Activation of Convolutions.
        RGEM: Regional Generalized Mean Pooling.
        LOCAL: Local feature extraction.
    """

    HEAD = enum.auto()
    CLS = enum.auto()
    GAP = enum.auto()
    MAC = enum.auto()
    GEM = enum.auto()
    RMAC = enum.auto()
    RGEM = enum.auto()
    LOCAL = enum.auto()

    def get_fn(self, **kwargs) -> Callable:
        """
        Returns the pooling function corresponding to the enum value.

        Args:
            **kwargs: Additional arguments for the pooling function.

        Returns:
            Callable: A pooling function.
        """
        return self._get_fn()(**kwargs)

    def _get_fn(self) -> Callable:
        """
        Maps enum values to their corresponding pooling functions.

        Returns:
            Callable: A pooling function class.
        """
        return {
            self.HEAD: nn.Identity,
            self.CLS: CLS,
            self.GAP: SPoC,
            self.MAC: MAC,
            self.GEM: GeM,
            self.RMAC: RMAC,
            self.RGEM: RGeM,
            self.LOCAL: Local,
        }[self]


def _assert_dims(x: torch.Tensor, dim: int = 1, ndim: int = 4) -> torch.Tensor:
    """
    Ensures the input tensor has the specified dimensions.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to transpose.
        ndim (int): Expected number of dimensions.

    Returns:
        torch.Tensor: Reshaped tensor.
    """
    if x.ndim != ndim or dim != 1:
        x = x.transpose(dim, 1).reshape(x.shape[0], x.shape[1], -1, 1)
    return x


def _spoc(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    """
    Computes Global Average Pooling (SPoC).

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to operate on.

    Returns:
        torch.Tensor: Pooled tensor.
    """
    x = _assert_dims(x, dim=dim)
    return reduce(x, "b c h w -> b c", "mean")


def _mac(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    """
    Computes Maximum Activation of Convolutions (MAC).

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to operate on.

    Returns:
        torch.Tensor: Pooled tensor.
    """
    x = _assert_dims(x, dim=dim)
    return reduce(x, "b c h w -> b c", "max")


def _gem(
    x: torch.Tensor, dim: int = 1, p: float = 3.0, eps: float = 1e-6, **kwargs
) -> torch.Tensor:
    """
    Computes Generalized Mean Pooling (GeM).

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to operate on.
        p (float): Power parameter for GeM.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Pooled tensor.
    """
    x = _assert_dims(x, dim=dim)
    m = reduce(x, "b c h w -> b c () ()", "min").clamp(max=0)
    return _spoc((x + m).clamp(min=eps).pow(p), dim).pow(1.0 / p) - m.squeeze()


def _rpool(
    x: torch.Tensor,
    L: List[int] = [1, 2, 3],
    aggr: Callable = _mac,
    p: float = 1e-6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes Regional Pooling (R-MAC or R-GeM).

    Args:
        x (torch.Tensor): Input tensor.
        L (List[int]): List of region levels.
        aggr (Callable): Aggregation function (e.g., _mac or _gem).
        p (float): Power parameter for GeM.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Pooled tensor.
    """
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.float32)  # possible regions

    W, H = x.shape[3], x.shape[2]
    w = min(W, H)
    b = (max(H, W) - w) / (steps - 1)
    idx = torch.argmin(torch.abs(((w**2 - w * b) / w**2) - ovr))

    Wd, Hd = (idx.item() + 1, 0) if H < W else (0, idx.item() + 1)

    vecs = []
    for l in L:
        if l == 1:
            vecs.append(aggr(x, dim=1, p=p, eps=eps).unsqueeze(1))
            continue
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        bW = (W - wl) / (l + Wd - 1) if l + Wd > 1 else 0
        cenW = torch.floor(wl2 + torch.arange(l + Wd) * bW).long() - wl2

        bH = (H - wl) / (l + Hd - 1) if l + Hd > 1 else 0
        cenH = torch.floor(wl2 + torch.arange(l + Hd) * bH).long() - wl2

        for i in cenH.tolist():
            for j in cenW.tolist():
                if wl == 0:
                    continue
                f = x[:, :, i : i + wl, j : j + wl]
                vecs.append(aggr(f, dim=1, p=p, eps=eps).unsqueeze(1))
    return torch.cat(vecs, dim=1).sum(1)


def _local(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Extracts local features by rearranging the tensor.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to operate on.

    Returns:
        torch.Tensor: Rearranged tensor.
    """
    x = _assert_dims(x, dim=dim)
    return rearrange(x, "b c h w -> b (h w) c")


class CLS(nn.Module):
    """
    Class token pooling layer.

    Methods:
        forward(x): Extracts the class token from the input tensor.
    """

    def __init__(self) -> None:
        super(CLS, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the class token.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Class token.
        """
        return x[:, 0]

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class MAC(nn.Module):
    """
    Maximum Activation of Convolutions (MAC) pooling layer.

    Methods:
        forward(x): Applies MAC pooling to the input tensor.
    """

    def __init__(self) -> None:
        super(MAC, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies MAC pooling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return _mac(x)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class SPoC(nn.Module):
    """
    Global Average Pooling (SPoC) layer.

    Methods:
        forward(x): Applies SPoC pooling to the input tensor.
    """

    def __init__(self) -> None:
        super(SPoC, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SPoC pooling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return _spoc(x)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Attributes:
        p (float): Power parameter for GeM.
        eps (float): Small value to avoid division by zero.

    Methods:
        forward(x): Applies GeM pooling to the input tensor.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super(GeM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GeM pooling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return _gem(x, p=self.p, eps=self.eps)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p)
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class RMAC(nn.Module):
    """
    Regional Maximum Activation of Convolutions (R-MAC) pooling layer.

    Attributes:
        L (List[int]): List of region levels.
        eps (float): Small value to avoid division by zero.

    Methods:
        forward(x): Applies R-MAC pooling to the input tensor.
    """

    def __init__(self, L: List[int] = [1, 2, 3], eps: float = 1e-6) -> None:
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies R-MAC pooling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return _rpool(x, L=self.L, aggr=_mac, eps=self.eps)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + "L=" + "{}".format(self.L) + ")"


class RGeM(nn.Module):
    """
    Regional Generalized Mean Pooling (R-GeM) layer.

    Attributes:
        L (List[int]): List of region levels.
        p (float): Power parameter for GeM.
        eps (float): Small value to avoid division by zero.

    Methods:
        forward(x): Applies R-GeM pooling to the input tensor.
    """

    def __init__(
        self, L: List[int] = [1, 2, 3], p: float = 3.0, eps: float = 1e-6
    ) -> None:
        super(RGeM, self).__init__()
        self.p = p
        self.L = L
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies R-GeM pooling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return _rpool(x, L=self.L, aggr=_gem, p=self.p, eps=self.eps)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p)
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Local(nn.Module):
    """
    Local feature extraction layer.

    Methods:
        forward(x): Extracts local features from the input tensor.
    """

    def __init__(self) -> None:
        super(Local, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts local features.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rearranged tensor with local features.
        """
        return _local(x)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"
