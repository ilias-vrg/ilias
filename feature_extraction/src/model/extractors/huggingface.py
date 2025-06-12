import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from timm.layers.pos_embed import resample_abs_pos_embed

from .generic_extractor import GenericFeatureExtractor
from ...datasets.transforms import MultiplierPadding


class HuggingFaceModel(GenericFeatureExtractor):
    """
    A feature extractor that uses a Hugging Face model for extracting features from input data.

    This class wraps a Hugging Face model and provides functionality for feature extraction,
    including preprocessing steps like padding and normalization.
    """

    def __init__(self, **kwargs):
        """
        Initialize the HuggingFaceModel.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super(HuggingFaceModel, self).__init__(**kwargs)

        self.model = AutoModel.from_pretrained(self.name)

        self.model.vision_model.embeddings = CLIPVisionEmbeddingsWrapper(
            self.model.vision_model.embeddings, num_prefix_tokens=0
        )
        self.pad = MultiplierPadding(16)

        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def _extract_features(self, x: torch.FloatTensor) -> torch.Tensor:
        """
        Extract features from the input data using the Hugging Face model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The extracted features.
        """
        return self.model(x)


class CLIPVisionEmbeddingsWrapper(nn.Module):
    """
    A wrapper for CLIPVisionEmbeddings that allows resizing positional embeddings.

    This wrapper modifies the positional embeddings of the CLIP vision model to adapt
    to different input sizes and adds support for prefix tokens.
    """

    def __init__(self, original_model: nn.Module, num_prefix_tokens: int = 1):
        """
        Initialize the CLIPVisionEmbeddingsWrapper.

        Args:
            original_model (nn.Module): The original CLIPVisionEmbeddings model.
            num_prefix_tokens (int): The number of prefix tokens to add to the positional embeddings.
        """
        super().__init__()
        self.model = original_model
        self.model.original_position_embedding = self.model.position_embedding
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass of the CLIPVisionEmbeddingsWrapper.

        Args:
            pixel_values (torch.FloatTensor): The input pixel values.
            interpolate_pos_encoding (bool, optional): Whether to interpolate the positional embeddings.

        Returns:
            torch.Tensor: The output of the CLIPVisionEmbeddings model.
        """
        new_size = (
            pixel_values.shape[-2] // self.model.patch_size,
            pixel_values.shape[-1] // self.model.patch_size,
        )

        # Interpolate positional embeddings
        posemb = resample_abs_pos_embed(
            self.model.original_position_embedding.weight.unsqueeze(0),
            new_size=new_size,
            num_prefix_tokens=self.num_prefix_tokens,
        )
        self.model.position_embedding = nn.Embedding(
            math.prod(new_size) + self.num_prefix_tokens,
            self.model.embed_dim,
            _weight=posemb[0, ...],
        )

        # Update model parameters
        self.model.image_size = (pixel_values.shape[-2], pixel_values.shape[-1])
        self.model.num_patches = math.prod(new_size)
        self.model.num_positions = self.model.num_patches + self.num_prefix_tokens
        self.model.position_ids = (
            torch.arange(self.model.num_positions)
            .expand((1, -1))
            .to(self.model.position_ids.device)
        )

        return self.model(pixel_values)
