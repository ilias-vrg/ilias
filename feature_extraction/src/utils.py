import argparse
import numpy as np

from typing import Generator, Any

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Mapping from image extractor model names to their OpenCLIP equivalents (to get access to the text encoder and tokenizer)
MODEL_MAPPING = {
    "vit_large_patch16_siglip_384.webli": ("hf-hub:timm/ViT-L-16-SigLIP-384", None),
    "vit_so400m_patch14_siglip_384.webli": (
        "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        None,
    ),
    "convnext_large_mlp.clip_laion2b_ft_soup_320": (
        "convnext_large_d_320",
        "laion2b_s29b_b131k_ft_soup",
    ),
    "vit_large_patch14_clip_336.openai": ("ViT-L-14-336", "openai"),
    "eva02_large_patch14_clip_336.merged2b": ("EVA02-L-14", "merged2b_s4b_b131k"),
    "vit_large_patch14_clip_224.metaclip_2pt5b": (
        "ViT-L-14-quickgelu",
        "metaclip_fullcc",
    ),
    "RN50.openai": ("RN50", "openai"),
    "vit_base_patch16_clip_224.openai": ("ViT-B-16", "openai"),
    "vit_large_patch14_clip_224.openai": ("ViT-L-14", "openai"),
    "vit_large_patch14_clip_224.laion2b": ("ViT-L-14", "laion2b_s32b_b82k"),
    "convnext_base.clip_laion2b_augreg": (
        "convnext_base_w",
        "laion2b_s13b_b82k_augreg",
    ),
    "eva02_base_patch16_clip_224.merged2b": ("EVA02-B-16", "merged2b_s8b_b131k"),
    "vit_base_patch16_siglip_224.webli": ("hf-hub:timm/ViT-B-16-SigLIP", None),
    "vit_base_patch16_siglip_512.webli": ("hf-hub:timm/ViT-B-16-SigLIP-512", None),
    "vit_base_patch16_clip_224.metaclip_2pt5b": (
        "ViT-B-16-quickgelu",
        "metaclip_fullcc",
    ),
    "vit_base_patch16_siglip_256.webli": ("hf-hub:timm/ViT-B-16-SigLIP-256", None),
    "vit_base_patch16_siglip_384.webli": ("hf-hub:timm/ViT-B-16-SigLIP-384", None),
    "vit_large_patch16_siglip_256.webli": ("hf-hub:timm/ViT-L-16-SigLIP-256", None),
    "vit_base_patch16_siglip_384.v2_webli": ("hf-hub:timm/ViT-B-16-SigLIP2-384", None),
    "vit_base_patch16_siglip_512.v2_webli": ("hf-hub:timm/ViT-B-16-SigLIP2-512", None),
    "vit_large_patch16_siglip_384.v2_webli": ("hf-hub:timm/ViT-L-16-SigLIP2-384", None),
    "vit_large_patch16_siglip_512.v2_webli": ("hf-hub:timm/ViT-L-16-SigLIP2-512", None),
    "vit_giantopt_patch16_siglip_384.v2_webli": (
        "hf-hub:timm/ViT-gopt-16-SigLIP2-384",
        None,
    ),
    "vit_so400m_patch16_siglip_512.v2_webli": (
        "hf-hub:timm/ViT-SO400M-16-SigLIP2-512",
        None,
    ),
    "vit_pe_core_base_patch16_224.fb": ("hf-hub:timm/PE-Core-B-16", None),
    "vit_pe_core_large_patch14_336.fb": ("hf-hub:timm/PE-Core-L-14-336", None),
}


def bool_flag(s: str) -> bool:
    """
    Convert a string to a boolean value.

    Args:
        s (str): The string to convert. Valid values are "true", "false", "on", "off", "1", or "0".

    Returns:
        bool: The converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the string is not a valid boolean value.

    Example:
        >>> bool_flag("true")
        True
        >>> bool_flag("off")
        False
    """
    truthy = {"on", "true", "1"}
    falsy = {"off", "false", "0"}
    s_lower = s.lower()
    if s_lower in truthy:
        return True
    elif s_lower in falsy:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid value for a boolean flag: {s}")


def batching(tensor: Any, batch_sz: int) -> Generator[Any, None, None]:
    """
    Split a tensor into batches of size `batch_sz`. The last batch may be smaller.

    Args:
        tensor (Any): The tensor or list to split.
        batch_sz (int): The size of each batch.

    Yields:
        Any: A batch of the tensor.

    Raises:
        ValueError: If `batch_sz` is less than or equal to 0.

    Example:
        >>> list(batching([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    if batch_sz <= 0:
        raise ValueError("batch_sz must be greater than 0")

    for i in range(0, len(tensor), batch_sz):
        yield tensor[i : i + batch_sz]


def load_image_ids(file_path: str) -> set:
    """
    Load image IDs from a text file.

    Args:
        file_path (str): Path to the text file containing image IDs.

    Returns:
        set: A set of image IDs.
    """
    image_ids = np.loadtxt(file_path, dtype=str, delimiter=",")
    if image_ids.ndim > 1:
        image_ids = image_ids[:, 0]
    return image_ids.tolist()
