import math
import enum
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Union, Dict, Any

import torchvision.transforms as transforms

from ..utils import IMAGENET_MEAN, IMAGENET_STD


class InferenceTransforms(enum.Enum):
    """
    Enum for different image transformation types used in inference.
    Each transformation type is associated with a specific resizing method.
    """

    RESIZE_CENTER = enum.auto()
    RESIZE_SMALL = enum.auto()
    RESIZE_LARGE = enum.auto()
    RESIZE_SQUARE = enum.auto()
    RESIZE_PAD = enum.auto()

    def get_transforms(
        self,
        resolution: int,
        min_size: int = 0,
        mean: List[float] = IMAGENET_MEAN,
        std: List[float] = IMAGENET_STD,
        **kwargs: Any,
    ) -> transforms.Compose:
        """
        Get the appropriate transformations based on the specified type.
        Args:
            resolution (int): The target resolution for resizing.
            min_size (int): Minimum size for padding.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
            **kwargs: Additional arguments.
        Returns:
            transforms.Compose: A composition of transformations.
        """
        return self._build_transforms(resolution, min_size, mean, std)

    def _build_transforms(
        self, resolution: int, min_size: int, mean: List[float], std: List[float]
    ) -> transforms.Compose:
        """
        Build a dictionary of transformations and return the one matching the enum key.
        Args:
            resolution (int): The target resolution for resizing.
            min_size (int): Minimum size for padding.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
        Returns:
            transforms.Compose: A composition of transformations.
        Raises:
            KeyError: If the transformation type is not supported.
        """
        return {
            self.RESIZE_SMALL: transforms.Compose(
                [
                    transforms.Resize(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            ),
            self.RESIZE_LARGE: transforms.Compose(
                [
                    LargeSideResize(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    CenterPadding(min_size),
                ]
            ),
            self.RESIZE_CENTER: transforms.Compose(
                [
                    transforms.Resize(resolution),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            ),
            self.RESIZE_SQUARE: transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            ),
            self.RESIZE_PAD: transforms.Compose(
                [
                    LargeSideResize(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    CenterPadding(resolution),
                ]
            ),
        }[self]


class LargeSideResize:
    """
    Resize an image such that the larger side matches the target size while maintaining the aspect ratio.
    """

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        # Calculate the new dimensions while maintaining the aspect ratio.
        W, H = img.size
        ar = float(H) / float(W)
        H = int(self.size * ar) if ar <= 1 else self.size
        H = max(1, H)
        W = int(self.size / ar) if ar > 1 else self.size
        W = max(1, W)
        return img.resize((W, H))  # Resize the image.

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(size={self.size})"


class MultiplierPadding(nn.Module):
    """
    Add padding to an image such that its dimensions are a multiple of a given value.
    """

    def __init__(self, multipler: int) -> None:
        super().__init__()
        self.multipler = multipler

    def _get_pad(self, size: int) -> Tuple[int, int]:
        # Calculate the padding needed to make the size a multiple of the multiplier.
        new_size = math.ceil(size / self.multipler) * self.multipler
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply padding to the input tensor.
        pads = list(
            itertools.chain.from_iterable(self._get_pad(s) for s in x.shape[:1:-1])
        )
        return F.pad(x, pads)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(multipler={self.multipler})"


class CenterPadding(nn.Module):
    """
    Add padding to an image to center it within a target size.
    """

    def __init__(self, target_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def _get_pad(self, target: int, size: int) -> Tuple[int, int]:
        # Calculate the padding needed to center the image.
        if target <= size:
            return 0, 0
        pad_size = target - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pads = list(
            itertools.chain.from_iterable(
                self._get_pad(s, m) for s, m in zip(self.target_size, x.shape[::-1])
            )
        )
        return F.pad(x, pads)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(size={self.target_size})"
