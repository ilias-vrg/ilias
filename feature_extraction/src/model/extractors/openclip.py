import math
import types
import torch
import open_clip
import torch.nn.functional as F

from einops import rearrange

from .generic_extractor import GenericFeatureExtractor


class OpenClipModel(GenericFeatureExtractor):
    """
    A feature extractor class for the OpenClip model.

    This class supports the 'RN50.openai' model with 'head' pooling. It initializes
    the OpenClip model, applies necessary transformations, and extracts features
    from input images.
    """

    def __init__(self, **kargs):
        """
        Initialize the OpenClipModel.

        Args:
            **kargs: Additional keyword arguments for the parent class.
        """
        super(OpenClipModel, self).__init__(**kargs)
        if self.name != "RN50.openai":
            raise NotImplementedError(
                f"Model '{self.name}' is not supported. Only 'RN50.openai' is supported for OpenClip. "
                "Use 'timm' for other models."
            )
        if self.pooling != "head":
            raise NotImplementedError(
                f"Pooling method '{self.pooling}' is not supported. Only 'head' pooling is implemented for OpenClip."
            )

        model, dataset = self.name.split(".")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model, pretrained=dataset
        )
        self.model.visual.attnpool.forward = types.MethodType(
            refactored_forward, self.model.visual.attnpool
        )

        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the input tensor using the OpenClip model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Extracted features (class token) from the model.

        Raises:
            NotImplementedError: If a pooling method other than 'head' is used.
        """
        cls_token = self.model.encode_image(x)

        if self.pooling == "head":
            return cls_token
        raise NotImplementedError("Local pooling is not implemented for OpenClip!")


def refactored_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Refactored forward method for the attention pooling layer.

    This method reshapes the input tensor, applies positional embeddings, and
    performs multi-head attention to extract features.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Output tensor after applying attention pooling.
    """
    b, c, h, w = x.shape
    x = x.reshape(b, c, h * w).permute(2, 0, 1)  # NCHW -> (HW)NC
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

    n, _ = self.positional_embedding.shape
    pos_enc_cls, pos_enc_tkn = torch.split(self.positional_embedding, [1, n - 1], dim=0)
    pos_enc_tkn = rearrange(
        pos_enc_tkn,
        "(h w) d -> () d h w",
        h=int(math.sqrt(n - 1)),
        w=int(math.sqrt(n - 1)),
    )
    pos_enc_tkn = F.interpolate(
        pos_enc_tkn, size=(h, w), mode="bicubic", align_corners=False
    )
    pos_enc = torch.cat(
        [pos_enc_cls.unsqueeze(1), rearrange(pos_enc_tkn, "b d h w -> (h w) b d")],
        dim=0,
    )  # (HW+1)NC

    x = x + pos_enc.to(x.dtype)  # (HW+1)NC
    x, _ = F.multi_head_attention_forward(
        query=x,
        key=x,
        value=x,
        embed_dim_to_check=x.shape[-1],
        num_heads=self.num_heads,
        q_proj_weight=self.q_proj.weight,
        k_proj_weight=self.k_proj.weight,
        v_proj_weight=self.v_proj.weight,
        in_proj_weight=None,
        in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=self.c_proj.weight,
        out_proj_bias=self.c_proj.bias,
        use_separate_proj_weight=True,
        training=self.training,
        need_weights=False,
    )

    return x[0]
