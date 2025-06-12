import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce, rearrange
from torchvision import models

from .generic_extractor import GenericFeatureExtractor


class TorchvisionModel(GenericFeatureExtractor):
    """
    A feature extractor class for models from the torchvision library.

    This class supports feature extraction for models like AlexNet. It initializes
    the model, applies necessary transformations, and extracts features from input images.
    """

    def __init__(self, **kwargs):
        """
        Initialize the TorchvisionModel.

        Args:
            **kwargs: Additional keyword arguments for the parent class.
        """
        super(TorchvisionModel, self).__init__(**kwargs)
        assert self.pooling != "cls", "CLS pooling is not applicable to non-ViT models"

        # Initialize the model based on the name
        self.extractor = self._get_model()

    def _get_model(self):
        """
        Retrieve the appropriate model based on the name.

        Returns:
            nn.Module: The initialized model.

        Raises:
            NotImplementedError: If the specified model name is not implemented.
        """
        if "alexnet" in self.name:
            return AlexNet()
        else:
            raise NotImplementedError(f"Model '{self.name}' is not implemented.")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input tensor using the specified model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Extracted features from the model.
        """
        x = self.extractor(x, self.pooling == "head")
        return x if self.pooling == "head" else self.pooling_local(x)


class AlexNet(nn.Module):
    """
    A wrapper class for the AlexNet model from torchvision.

    This class modifies the AlexNet model by replacing the classifier's final layer
    with an identity layer for feature extraction.
    """

    def __init__(self):
        """
        Initialize the AlexNet model with pre-trained weights.
        """
        super(AlexNet, self).__init__()
        from torchvision.models.alexnet import AlexNet_Weights

        # Load the pre-trained AlexNet model
        self.model = models.alexnet(AlexNet_Weights.IMAGENET1K_V1)
        # Replace the final classifier layer with an identity layer
        self.model.classifier[6] = nn.Identity()

    def forward(self, x: torch.Tensor, use_classifier: bool = True) -> torch.Tensor:
        """
        Forward pass for the AlexNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            use_classifier (bool): Whether to use the classifier part of the model.

        Returns:
            torch.Tensor: Output tensor after feature extraction or classification.
        """
        # Pass input through the feature extractor
        x = self.model.features(x)
        if not use_classifier:
            return x

        # Pass through the average pooling and classifier layers
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x
