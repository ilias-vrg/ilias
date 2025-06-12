import enum
from typing import Any, Callable

import torch.nn as nn


class FeatureExtractor(enum.Enum):
    """
    Enum for different feature extractors.
    Each extractor is associated with a specific model type.
    """

    TIMM = enum.auto()
    TV = enum.auto()
    HF = enum.auto()
    OCH = enum.auto()
    FB = enum.auto()
    CVN = enum.auto()
    UNC = enum.auto()
    UNICOM = enum.auto()
    TEXT = enum.auto()

    def get_model(
        self,
        model: str,
        pooling: str,
        **kwargs: Any,
    ) -> nn.Module:
        """
        Get the model for the specified feature extractor.

        Args:
            model (str): The model to use.
            pooling (str): The pooling method to use.
            **kwargs (Any): Additional arguments for the model.

        Returns:
            nn.Module: The model for the specified feature extractor.

        Raises:
            ValueError: If the feature extractor is not supported.
        """
        try:
            backbone = self._get_config()(model=model, pooling=pooling, **kwargs)
            return backbone
        except KeyError:
            raise ValueError(f"Unsupported feature extractor: {self.name}")

    def _get_config(self) -> Callable[..., Any]:
        """
        Get the configuration function for the specified feature extractor.

        Returns:
            Callable[..., Any]: The configuration function for the specified feature extractor.
        """
        return {
            self.TIMM: self._get_timm,
            self.TV: self._get_tv,
            self.HF: self._get_hf,
            self.OCH: self._get_openclip,
            self.FB: self._get_fb,
            self.CVN: self._get_cvnet,
            self.UNC: self._get_unic,
            self.UNICOM: self._get_unicom,
            self.TEXT: self._get_text,
        }[self]()

    @staticmethod
    def _get_timm() -> nn.Module:
        """
        Get the TimmModel class.

        Returns:
            nn.Module: The TimmModel class.
        """
        from .extractors.timm import TimmModel

        return TimmModel

    @staticmethod
    def _get_tv() -> nn.Module:
        """
        Get the TorchvisionModel class.

        Returns:
            nn.Module: The TorchvisionModel class.
        """
        from .extractors.torchvision import TorchvisionModel

        return TorchvisionModel

    @staticmethod
    def _get_hf() -> nn.Module:
        """
        Get the HuggingFaceModel class.

        Returns:
            nn.Module: The HuggingFaceModel class.
        """
        from .extractors.huggingface import HuggingFaceModel

        return HuggingFaceModel

    @staticmethod
    def _get_openclip() -> nn.Module:
        """
        Get the OpenClipModel class.

        Returns:
            nn.Module: The OpenClipModel class.
        """
        from .extractors.openclip import OpenClipModel

        return OpenClipModel

    @staticmethod
    def _get_fb() -> nn.Module:
        """
        Get the FBModel class.

        Returns:
            nn.Module: The FBModel class.
        """
        from .extractors.facebook import FBModel

        return FBModel

    @staticmethod
    def _get_cvnet() -> nn.Module:
        """
        Get the CVNetModel class.

        Returns:
            nn.Module: The CVNetModel class.
        """
        from .extractors.cvnet import CVNetModel

        return CVNetModel

    @staticmethod
    def _get_unic() -> nn.Module:
        """
        Get the UNICModel class.

        Returns:
            nn.Module: The UNICModel class.
        """
        from .extractors.unic import UNICModel

        return UNICModel

    @staticmethod
    def _get_unicom() -> nn.Module:
        """
        Get the UnicomModel class.

        Returns:
            nn.Module: The UnicomModel class.
        """
        from .extractors.unicom import UnicomModel

        return UnicomModel

    @staticmethod
    def _get_text() -> nn.Module:
        """
        Get the TextFeatureExtractor class.

        Returns:
            nn.Module: The TextFeatureExtractor class.
        """
        from .extractors.text import TextFeatureExtractor

        return TextFeatureExtractor
