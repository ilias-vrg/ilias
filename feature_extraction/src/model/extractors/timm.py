import timm
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .generic_extractor import GenericFeatureExtractor
from ...datasets.transforms import MultiplierPadding


class TimmModel(GenericFeatureExtractor):
    def __init__(self, **kargs):
        """
        Initialize the TimmModel.

        Args:
            **kargs: Additional arguments passed to the GenericFeatureExtractor.
        """
        super(TimmModel, self).__init__(**kargs)

        self.model = self._build_model()
        self.pad = self._initialize_padding()
        self.mean = self.model.default_cfg["mean"]
        self.std = self.model.default_cfg["std"]

    def _initialize_padding(self) -> nn.Module:
        """
        Initialize padding based on the model type.

        Returns:
            nn.Module: Padding module.
        """
        if self._is_vit():
            patch_size = 16 if "patch16" in self.name else 14
            return MultiplierPadding(patch_size)
        return nn.Identity()

    def _is_vit(self) -> bool:
        """
        Check if the model is a Vision Transformer (ViT).

        Returns:
            bool: True if the model is a ViT, False otherwise.
        """
        return any(vit in self.name for vit in ["vit", "eva", "deit"])

    def _is_clip(self) -> bool:
        """
        Check if the model is a CLIP or PE model.

        Returns:
            bool: True if the model is a CLIP or PE model, False otherwise.
        """
        return "clip" in self.name or "pe_core" in self.name

    def _build_model(self) -> nn.Module:
        """
        Build the model using the timm library.

        Returns:
            nn.Module: The constructed model.

        Raises:
            ValueError: If CLS pooling is used with non-ViT models.
        """
        if not self._is_vit():
            if self.pooling == "cls":
                raise ValueError("CLS pooling is not applicable to non-ViT models")
            return timm.create_model(
                self.name,
                pretrained=True,
                num_classes=0 if not self._is_clip() else None,
            )

        return timm.create_model(
            self.name,
            pretrained=True,
            num_classes=0 if not self._is_clip() else None,
            dynamic_img_size=True,
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        x = self.pad(x)

        if self.pooling == "head":
            return self.model(x)

        x = self.model.forward_features(x)

        if self.model.global_pool == "token":
            cls_token = x[:, 0]
            x = x[:, 1:]

        if self.pooling == "cls":
            return cls_token

        if x.ndim != 4:
            x = rearrange(x, "b l d -> b d l ()")
        return self.pooling_local(x)
