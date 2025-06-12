from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import (
    create_model_from_pretrained,
    get_tokenizer,
)

from ...utils import MODEL_MAPPING


class TextFeatureExtractor(nn.Module):
    """
    Feature extractor class for text models.
    """

    def __init__(self, model: str, **kargs):
        """
        Initialize the TextFeatureExtractor.

        Args:
            model (str): Name of the model.
            **kargs: Additional arguments.
        """
        super(TextFeatureExtractor, self).__init__()
        if model not in MODEL_MAPPING:
            raise ValueError(f"Backbone {model} is not supported.")

        model_name, pretrained = MODEL_MAPPING[model]
        if pretrained is None:
            self.model, _ = create_model_from_pretrained(model_name)
        else:
            self.model, _ = create_model_from_pretrained(
                model_name, pretrained=pretrained
            )
        self.tokenizer = get_tokenizer(model_name)

    def forward(self, text: List[str]) -> torch.Tensor:
        """
        Forward pass for text feature extraction.

        Args:
            text (List[str]): List of text strings to extract features from.

        Returns:
            torch.Tensor: Extracted text features.
        """
        tokens = self.tokenizer(text)
        tokens = tokens.to(next(self.model.parameters()).device)
        with torch.amp.autocast("cuda"):
            features = self.model.encode_text(tokens)
            features = F.normalize(features, dim=-1)

        return features
