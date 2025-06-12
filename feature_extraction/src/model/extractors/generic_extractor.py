import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import List
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ..pooling import PoolingFn
from ...utils import IMAGENET_MEAN, IMAGENET_STD


class GenericFeatureExtractor(nn.Module, ABC):
    """
    Generic feature extractor class for different models.
    This class serves as a base class for feature extractors.
    It provides a common interface for feature extraction and their required attributes.
    """

    def __init__(self, model: str, pooling: str, **kargs):
        """
        Initialize the GenericFeatureExtractor.

        Args:
            model (str): Name of the model.
            pooling (str): Pooling method to use.
            **kargs: Additional arguments.
        """
        super(GenericFeatureExtractor, self).__init__()

        self.name = model
        self.model = None
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

        self.pooling = pooling
        self.pooling_local = PoolingFn[pooling.upper()].get_fn()

    @abstractmethod
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to extract features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        raise NotImplementedError

    def forward(
        self, img: torch.Tensor, scales: List[float] = [1], rotate: List[float] = [0]
    ) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            img (torch.Tensor): Input image tensor.
            scales (List[float]): List of scaling factors.
            rotate (List[float]): List of rotation angles in degrees.

        Returns:
            torch.Tensor: Normalized feature tensor.
        """
        if scales == [1] and rotate == [0]:
            return self._l2_norm(self._extract_features(img))

        features = []
        for scale in scales:
            img_scaled = self._rescale_image(img, scale)
            for angle in rotate:
                img_transformed = self._rotate_image(img_scaled, angle)
                features.append(
                    self._l2_norm(self._extract_features(img_transformed).unsqueeze(1))
                )

        return self._l2_norm(torch.cat(features, dim=1).mean(dim=1))

    def _rescale_image(self, img: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scale the input image.

        Args:
            img (torch.Tensor): Input image tensor.
            scale (float): Scaling factor.

        Returns:
            torch.Tensor: Scaled image tensor.
        """
        if scale == 1:
            return img.clone()
        return F.interpolate(
            img,
            scale_factor=scale,
            mode="bicubic",
            align_corners=False,
        )

    def _rotate_image(self, img: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rotate the input image.

        Args:
            img (torch.Tensor): Input image tensor.
            angle (float): Rotation angle in degrees.

        Returns:
            torch.Tensor: Rotated image tensor.
        """
        if angle == 0:
            return img.clone()
        return TF.rotate(img, angle, InterpolationMode.BILINEAR)

    def _l2_norm(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize the extracted features.

        Args:
            features (torch.Tensor): Extracted feature tensor.

        Returns:
            torch.Tensor: Normalized feature tensor.
        """
        return F.normalize(features, p=2, dim=-1)
