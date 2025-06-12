import math
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from functools import partial

from .generic_extractor import GenericFeatureExtractor
from ...datasets.transforms import MultiplierPadding


class FBModel(GenericFeatureExtractor):
    def __init__(self, **kargs):
        """
        Initialize the FBModel.

        Args:
            **kargs: Additional arguments passed to the GenericFeatureExtractor.
        """
        super(FBModel, self).__init__(**kargs)

        if "moco" in self.name and self.pooling != "head":
            raise NotImplementedError("Only 'head' pooling is implemented for MoCo")
        if "swav" in self.name and self.pooling != "head":
            raise NotImplementedError("Only 'head' pooling is implemented for SwAV")

        self.model = self._get_model()

    def _get_model(self) -> nn.Module:
        """
        Get the appropriate model based on the name.

        Returns:
            nn.Module: The initialized model.

        Raises:
            NotImplementedError: If the model name is not recognized.
        """
        if "dinov2" in self.name:
            return DINOv2(name=self.name)
        elif "dino" in self.name:
            return DINO(name=self.name)
        elif "swav" in self.name:
            return SwAV()
        elif "moco" in self.name:
            return MoCo(name=self.name)
        else:
            raise NotImplementedError(f"Model '{self.name}' is not implemented.")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        if "head" == self.pooling or "dino" not in self.name:
            return self.model(x)

        x, tokens = self.model(x, True)
        if self.pooling == "head":
            return x
        return self.pooling_local(tokens)


class DINO(nn.Module):
    def __init__(self, name: str):
        """
        Initialize the DINO model.

        Args:
            name (str): Name of the DINO model.
        """
        super(DINO, self).__init__()
        self.model = torch.hub.load("facebookresearch/dino:main", name)
        self.pad = MultiplierPadding(16) if "resnet50" not in name else nn.Identity()

    def forward(self, x: torch.Tensor, return_patch_token: bool = False):
        """
        Extract features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            return_patch_token (bool): Whether to return patch tokens.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Extracted features.
        """
        x = self.pad(x)

        if not return_patch_token:
            return self.model(x)

        x = self.model.get_intermediate_layers(x)[-1]
        cls_token = x[:, 0]
        x = rearrange(x[:, 1:], "b l d -> b d l ()")

        return cls_token, x


class DINOv2(nn.Module):
    def __init__(self, name: str):
        """
        Initialize the DINOv2 model.

        Args:
            name (str): Name of the DINOv2 model.
        """
        super(DINOv2, self).__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.pad = MultiplierPadding(14)

    def forward(self, x: torch.Tensor, return_patch_token: bool = False):
        """
        Extract features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            return_patch_token (bool): Whether to return patch tokens.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Extracted features.
        """
        x = self.pad(x)

        if not return_patch_token:
            return self.model(x)

        x, cls_token = self.model.get_intermediate_layers(
            x, return_class_token=True, reshape=True
        )[-1]

        return cls_token, x


class SwAV(nn.Module):
    def __init__(self):
        """
        Initialize the SwAV model.
        """
        super(SwAV, self).__init__()
        self.model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwAV model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, None]: Extracted features and None (no patch tokens).
        """
        x = self.model(x)
        return x


class MoCo(nn.Module):
    MODELS_WEIGHTS = {
        "mocov3_resnet50": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar",
        "mocov3_vitb16": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
    }

    def __init__(self, name: str):
        """
        Initialize the MoCo model.

        Args:
            name (str): Name of the MoCo model.
        """
        super(MoCo, self).__init__()
        self.model, self.pad = self._initialize_model(name)

        url = self.MODELS_WEIGHTS[name]
        weights = torch.hub.load_state_dict_from_url(url)["state_dict"]
        weights = {
            k.replace("module.base_encoder.", ""): v
            for k, v in weights.items()
            if "base_encoder" in k
        }
        self.model.load_state_dict(weights, strict=False)

    def _initialize_model(self, name) -> tuple:
        """
        Initialize the MoCo model based on its name.

        Returns:
            Tuple[nn.Module, nn.Module]: The initialized model and padding module.
        """
        if "resnet" in name:
            from torchvision.models import resnet50

            model = resnet50(zero_init_residual=True)
            model.fc = nn.Identity()
            return model, nn.Identity()
        elif "vitb16" in name:
            from timm.models.vision_transformer import VisionTransformer, _cfg

            model = VisionTransformer(
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_classes=0,
                dynamic_img_size=True,
            )
            model.default_cfg = _cfg()
            return model, MultiplierPadding(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, None]: Extracted features and None (no patch tokens).
        """
        x = self.pad(x)
        x = self.model(x)
        return x
