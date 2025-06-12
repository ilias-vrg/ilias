import hashlib
import os
import urllib
import warnings

import torch
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from .generic_extractor import GenericFeatureExtractor
from ...datasets.transforms import MultiplierPadding


__all__ = ["load", "available_models"]

_MODELS = {
    "ViT-B/32": "https://github.com/deepglint/unicom/releases/download/b32/FP16-ViT-B-32.pt",
    "ViT-B/16": "https://github.com/deepglint/unicom/releases/download/b16/FP16-ViT-B-16.pt",
    "ViT-L/14": "https://github.com/deepglint/unicom/releases/download/l14/FP16-ViT-L-14.pt",
    "ViT-L/14@336px": "https://github.com/deepglint/unicom/releases/download/l14_336px/FP16-ViT-L-14-336px.pt",
    "ViT-B/16-sop": "feature_extraction/pretrained_models/unicom/ViT_B_16_20_test_88.847.pt",
    "ViT-B/16-gldv2": "feature_extraction/pretrained_models/unicom/ViT_B_16_GLDv2_test_0.356_val_0.327.pt",
}

_SHA256 = {
    "FP16-ViT-B-32.pt": "f9d5696a9b58dbbbefee2d31615ca59084f2895a0fdd2ca4c235e0f9b2793f7a",
    "FP16-ViT-B-16.pt": "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef",
    "FP16-ViT-L-14.pt": "ff3ab62ff782876460099e6e0ee17b73a7c01109de2fffd595f16f4129404bbd",
    "FP16-ViT-L-14-336px.pt": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
}

NAME_MAPPING = {
    "unicom_vit_base_patch16_224": "ViT-B/16",
    "unicom_vit_large_patch14_224": "ViT-L/14",
    "unicom_vit_large_patch14_336": "ViT-L/14@336px",
    "unicom_vit_base_patch16_sop": "ViT-B/16-sop",
    "unicom_vit_base_patch16_gldv2": "ViT-B/16-gldv2",
}


class UnicomModel(GenericFeatureExtractor):
    """
    A feature extractor model based on the Unicom architecture.

    This class initializes and loads a specific Unicom model based on the provided name.
    It supports different pooling mechanisms and applies padding, normalization, and
    other preprocessing steps to the input data.
    """

    def __init__(self, **kwargs):
        """
        Initializes the UnicomModel.

        Args:
            **kwargs: Additional arguments passed to the GenericFeatureExtractor.
        """
        super(UnicomModel, self).__init__(**kwargs)
        if self.pooling in "cls":
            raise NotImplementedError("CLS pooling not implemented for Unicom!")

        self.name = NAME_MAPPING[self.name]
        if "sop" in self.name or "gldv2" in self.name:
            self.model = build_model(self.name)
            weight_path = _MODELS[self.name]
            state_dict = torch.load(weight_path, "cpu")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            self.model, _ = load(self.name)

        self.pad = MultiplierPadding(14 if "14" in self.name else 16)
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)

    def _extract_features(self, x):
        """
        Extracts features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        x = self.pad(x)
        if self.pooling == "head":
            return self.model(x, head=True)
        x = self.model(x, head=False)
        return self.pooling_local(x)


def available_models() -> List[str]:
    """
    Returns the names of available Unicom models.

    Returns:
        List[str]: A list of model names.
    """
    return list(_MODELS.keys())


def rm_module_from_state_dict(state_dict: dict) -> dict:
    """
    Removes the 'module.' prefix from keys in a state dictionary.

    Args:
        state_dict (dict): The state dictionary to process.

    Returns:
        dict: A new state dictionary with 'module.' removed from keys.
    """
    result = {}
    for k, value in state_dict.items():

        if "module." in k:
            k_removed = k.split("module.")[-1]
            result[k_removed] = value
        else:
            result[k] = value
    return result


# copy from https://github.com/openai/CLIP/blob/main/clip/clip.py#L43
def _download(url: str, root: str):
    """
    Downloads a file from a URL and verifies its SHA256 checksum.

    Args:
        url (str): The URL to download the file from.
        root (str): The directory to save the downloaded file.

    Returns:
        str: The path to the downloaded file.

    Raises:
        RuntimeError: If the file exists but is not a regular file or if the checksum does not match.
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = _SHA256[filename]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


# copy from https://github.com/openai/CLIP/blob/main/clip/clip.py#L94
def load(name: str, device: str = "cpu", download_root: str = None):
    """
    Loads a Unicom model and its associated transform.

    Args:
        name (str): The name of the model to load.
        device (str): The device to load the model onto (default: "cpu").
        download_root (str): The root directory for downloading the model.

    Returns:
        Tuple[nn.Module, torchvision.transforms.Compose]: The loaded model and its transform.

    Raises:
        RuntimeError: If the model name is not found.
    """
    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root or os.path.expanduser("~/.cache/unicom")
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )
    with open(model_path, "rb") as opened_file:
        state_dict = torch.load(opened_file, map_location="cpu")

    model, transform = load_model_and_transform(name)
    state_dict_fp32 = {}
    for k, v in state_dict.items():
        state_dict_fp32[k] = v.float()

    model.load_state_dict(state_dict)
    return model, transform


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation for image feature extraction.

    This class defines the architecture of a Vision Transformer, including patch embedding,
    positional embedding, transformer blocks, and feature extraction layers.
    """

    def __init__(
        self,
        input_size=224,
        patch_size=32,
        in_channels=3,
        dim=768,
        embedding_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.0,
        using_checkpoint=True,
    ):
        """
        Initializes the VisionTransformer.

        Args:
            input_size (int): Input image size.
            patch_size (int): Size of each patch.
            in_channels (int): Number of input channels.
            dim (int): Dimension of the transformer.
            embedding_size (int): Size of the embedding.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio of hidden dimension in MLP.
            drop_path_rate (float): Drop path rate.
            using_checkpoint (bool): Whether to use gradient checkpointing.
        """
        super().__init__()
        self.dim = dim
        self.patch_embed = PatchEmbedding(
            input_size,
            patch_size,
            in_channels,
            dim,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    num_heads,
                    mlp_ratio,
                    dpr[i],
                    self.patch_embed.num_patches,
                    using_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

        self.feature = nn.Sequential(
            nn.Linear(dim * self.patch_embed.num_patches, dim, False),
            nn.BatchNorm1d(dim, eps=2e-5),
            nn.Linear(dim, embedding_size, False),
            nn.BatchNorm1d(embedding_size, eps=2e-5),
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        Extracts features from the input tensor using the transformer blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]: Extracted features and spatial dimensions.
        """
        B = x.shape[0]
        x, (H, W) = self.patch_embed(x)
        _, N, D = self.pos_embed.shape
        N_ = int(N ** (1 / 2))
        pos_emb = self.pos_embed.view(1, N_, N_, D).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb, size=(H, W), mode="bicubic")
        pos_emb = pos_emb.flatten(2).transpose(1, 2)
        x = x + pos_emb
        for func in self.blocks:
            x = func(x)
        x = self.norm(x.float())
        return x, (H, W)

    def forward(self, x, head=True):
        """
        Forward pass through the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor.
            head (bool): Whether to apply the head for feature extraction.

        Returns:
            torch.Tensor: Output tensor.
        """
        x, (H, W) = self.forward_features(x)
        if not head:
            return x.transpose(1, 2).unsqueeze(-1)
        B, _, D = x.shape
        x = x.view(B, H, W, D).permute(0, 3, 1, 2)
        N = self.patch_embed.num_patches
        N_ = int(N ** (1 / 2))
        x = F.adaptive_avg_pool2d(x, (N_, N_))
        x = x.flatten(2).transpose(1, 2)
        x = x.view(B, N * self.dim)
        return self.feature(x)


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used in the Vision Transformer.

    This block consists of two linear layers with a ReLU6 activation in between.
    """

    def __init__(self, dim, dim_hidden):
        """
        Initializes the MLP block.

        Args:
            dim (int): Input and output dimension.
            dim_hidden (int): Hidden layer dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_hidden)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        """
        Forward pass through the MLP block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """
    Self-attention mechanism used in the Vision Transformer.

    This class implements scaled dot-product attention with multiple heads.
    """

    def __init__(self, dim, num_heads):
        """
        Initializes the Attention module.

        Args:
            dim (int): Dimension of the input and output.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Forward pass through the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        B, L, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, L, 3, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        return x


class Block(nn.Module):
    """
    Transformer block consisting of self-attention and MLP layers.

    This block includes layer normalization, attention, MLP, and optional drop path.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        patch_n: int = 32,
        using_checkpoint=False,
    ):
        """
        Initializes the Transformer block.

        Args:
            dim (int): Dimension of the input and output.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio of hidden dimension in MLP.
            drop_path (float): Drop path rate.
            patch_n (int): Number of patches.
            using_checkpoint (bool): Whether to use gradient checkpointing.
        """
        super().__init__()
        self.using_checkpoint = using_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.extra_gflops = (num_heads * patch_n * (dim // num_heads) * patch_n * 2) / (
            1000**3
        )

    def forward_impl(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.using_checkpoint:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for the Vision Transformer.

    This layer splits the input image into patches and projects them into a higher-dimensional space.
    """

    def __init__(
        self, input_size=224, patch_size=32, in_channels: int = 3, dim: int = 768
    ):
        """
        Initializes the PatchEmbedding layer.

        Args:
            input_size (int): Input image size.
            patch_size (int): Size of each patch.
            in_channels (int): Number of input channels.
            dim (int): Dimension of the output embedding.
        """
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        H = input_size[0] // patch_size[0]
        W = input_size[1] // patch_size[1]
        self.num_patches = H * W
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Forward pass through the PatchEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]: Embedded patches and their spatial dimensions.
        """
        x = self.proj(x)
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)


def build_model(name="ViT-L/14@336px"):
    """
    Builds a Vision Transformer model based on the specified name.

    Args:
        name (str): Name of the model to build.

    Returns:
        VisionTransformer: The constructed Vision Transformer model.
    """
    if name == "ViT-B/32":
        model = VisionTransformer(
            input_size=224,
            patch_size=32,
            in_channels=3,
            dim=768,
            embedding_size=512,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            using_checkpoint=False,
        )
    elif name == "ViT-B/16":
        model = VisionTransformer(
            input_size=224,
            patch_size=16,
            in_channels=3,
            dim=768,
            embedding_size=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            using_checkpoint=False,
        )
    elif name == "ViT-L/14":
        model = VisionTransformer(
            input_size=224,
            patch_size=14,
            in_channels=3,
            dim=1024,
            embedding_size=768,
            depth=24,
            num_heads=16,
            drop_path_rate=0.1,
            using_checkpoint=False,
        )
    elif name == "ViT-L/14@336px":
        model = VisionTransformer(
            input_size=336,
            patch_size=14,
            in_channels=3,
            dim=1024,
            embedding_size=768,
            depth=24,
            num_heads=16,
            drop_path_rate=0.1,
            using_checkpoint=False,
        )
    elif name == "ViT-B/16-sop":
        model = VisionTransformer(
            input_size=224,
            patch_size=16,
            in_channels=3,
            dim=768,
            embedding_size=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            using_checkpoint=False,
        )
    elif name == "ViT-B/16-gldv2":
        model = VisionTransformer(
            input_size=512,
            patch_size=16,
            in_channels=3,
            dim=768,
            embedding_size=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            using_checkpoint=False,
        )
    return model


def _convert_image_to_rgb(image):
    """
    Converts an image to RGB format.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Image in RGB format.
    """
    return image.convert("RGB")


def _transform(n_px):
    """
    Creates a transformation pipeline for preprocessing images.

    Args:
        n_px (int): Target size for resizing and cropping.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def load_model_and_transform(name="ViT-L/14@336px"):
    """
    Loads a Vision Transformer model and its associated transformation pipeline.

    Args:
        name (str): Name of the model to load.

    Returns:
        Tuple[VisionTransformer, torchvision.transforms.Compose]: The model and its transform.
    """
    if name == "ViT-B/32":
        return build_model(name), _transform(224)
    elif name == "ViT-B/16":
        return build_model(name), _transform(224)
    elif name == "ViT-L/14":
        return build_model(name), _transform(224)
    elif name == "ViT-L/14@336px":
        return build_model(name), _transform(336)
    else:
        raise
