import os
import os.path as osp
import glob
import logging
import webdataset as wds
import numpy as np
import torch
import torch.nn.functional as F

from functools import partial
from collections import defaultdict
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List, Optional, Tuple, Union


def log_and_continue(exn: Exception) -> bool:
    """
    Handles exceptions during WebDataset processing by logging a warning and continuing.
    This ensures that errors in individual samples do not halt the entire pipeline.
    """
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def filter_no_image(sample: dict) -> bool:
    """
    Filters out samples that do not contain an image.
    Returns True if the sample contains a 'jpg' key, otherwise False.
    """
    return "jpg" in sample


def in_shortlist(shortlist: Optional[set], sample: dict) -> bool:
    """
    Checks if a sample is in the provided shortlist.
    If no shortlist is provided, all samples are included.
    """
    if shortlist is None:
        return True
    return sample["__key__"] in shortlist


def merge_id_shard(sample: dict) -> dict:
    """
    Combines the shard name and image ID into a single unique ID for the sample.
    This helps in uniquely identifying images across multiple shards.
    """
    shard = osp.basename(sample["shard"])
    image_id = sample["id"]
    sample["id"] = f"{shard}/{image_id}"
    return sample


def collate_batches(
    batch: List[Tuple[torch.Tensor, str]],
) -> Tuple[List[torch.Tensor], List[List[str]]]:
    """
    Groups images by aspect ratio and stacks them into batches.
    This ensures that images with similar aspect ratios are batched together,
    which can improve efficiency during processing.
    Returns:
        - A list of image tensors grouped by aspect ratio.
        - A list of corresponding image IDs.
    """
    aspect_ratios = defaultdict(list)
    images, image_ids = zip(*batch)
    for image, image_id in zip(images, image_ids):
        D, H, W = image.shape
        ar = H / W
        aspect_ratios[ar].append((image, image_id))
    all_images, all_image_ids = [], []
    for _, v in aspect_ratios.items():
        images, image_ids = zip(*v)
        all_images.append(torch.stack(images))
        all_image_ids.append(image_ids)
    return all_images, all_image_ids


def get_wds_dataset(
    input_shards: str,
    preprocess_img: Callable[[Image.Image], torch.Tensor],
    batch_size: int = 64,
    num_workers: int = 4,
    selected_ids: Optional[List[str]] = None,
) -> Tuple[wds.WebLoader, wds.DataPipeline]:
    """
    Creates a WebDataset pipeline and DataLoader for loading image data from shards.
    The pipeline includes filtering, decoding, renaming, preprocessing, and batching.

    Args:
        input_shards (str): Path to the input shards.
        preprocess_img (callable): Function to preprocess images.
        batch_size (int): Number of samples per batch (default: 64).
        num_workers (int): Number of worker threads for data loading (default: 4).
        selected_ids (list): Optional list of selected image IDs to filter the dataset.

    Returns:
        tuple: A DataLoader and the underlying dataset pipeline.
    """
    assert input_shards is not None, "Input shards must be specified."
    pipeline = [wds.SimpleShardList(input_shards)]

    if selected_ids is not None:
        selected_ids = set(selected_ids)

    # Define the WebDataset pipeline
    pipeline.extend(
        [
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.select(filter_no_image),
            wds.select(partial(in_shortlist, selected_ids)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg", id="__key__", shard="__url__"),
            wds.map(merge_id_shard),
            wds.map_dict(image=preprocess_img),
            wds.to_tuple("image", "id"),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    # Create a DataLoader for the dataset
    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        collate_fn=collate_batches,
    )

    return dataloader, dataset


def get_image_dataset(
    dataset_dir: str,
    preprocess_img: Callable[[Image.Image], torch.Tensor],
    batch_size: int = 64,
    num_workers: int = 4,
    partition: str = "queries",
    masking: bool = False,
    crop_resize: bool = False,
    image_ids: Optional[List[str]] = None,
) -> Tuple[DataLoader, Dataset]:
    """
    Creates a PyTorch Dataset and DataLoader for loading images from a directory.
    Supports filtering by file extensions and query-only mode.

    Args:
        dataset_dir (str): Path to the dataset directory.
        preprocess_img (callable): Function to preprocess images.
        batch_size (int): Number of samples per batch (default: 64).
        num_workers (int): Number of worker threads for data loading (default: 4).
        partition (str): The partition of the dataset to use (default: "queries").
        image_ids (list): Optional list of image IDs to filter the dataset.

    Returns:
        tuple: A DataLoader and the underlying dataset.
    """
    dataset = ImageDataset(
        dataset_dir,
        preprocess_img,
        partition=partition,
        image_ids=image_ids,
        masking=masking,
        crop_resize=crop_resize,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batches,
    )

    return dataloader, dataset


def get_intstance_descriptions(root_dir: str) -> tuple[List[str], List[str]]:
    """
    Get all instances from ILIAS dataset and their descriptions.

    Args:
        root_dir (str): Path to the ILIAS dataset directory.

    Returns:
        tuple[List[str], List[str]]: A tuple containing a list of instance IDs and their corresponding descriptions.
    """
    instances = sorted(
        [
            osp.relpath(f, root_dir)
            for f in glob.glob(f"{root_dir}/**/T*.txt", recursive=True)
        ]
    )

    descriptions = []
    for instance in instances:
        with open(os.path.join(root_dir, instance), "r") as f:
            descriptions.append(f.read())

    return instances, descriptions


class ImageDataset(Dataset):
    """
    A PyTorch Dataset for loading images from a directory.
    Supports filtering by file extensions and query-only mode.

    Args:
        root_dir (str): Path to the root directory containing images.
        transform (callable): Optional transform to apply to images.
        extensions (list): List of allowed file extensions (default: ["jpg", "jpeg", "png"]).
        partition (str): The partition of the dataset to use (default: "queries").
        image_ids (list): Optional list of image IDs to filter the dataset.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        extensions: List[str] = ["jpg", "jpeg", "png"],
        partition: str = "queries",
        image_ids: Optional[List[str]] = None,
        masking: bool = False,
        crop_resize: bool = False,
    ):
        self.root_dir = root_dir

        if image_ids is None:
            # Generate a list of image IDs based on file extensions and query mode
            extensions = set(sum([[e, e.lower(), e.upper()] for e in extensions], []))

            self.image_ids = [
                osp.relpath(img, root_dir)
                for img in glob.glob(f"{root_dir}/**/*.*", recursive=True)
                if img.split(".")[-1] in extensions
                and (
                    ("pos" in partition and "pos" in img)
                    or ("quer" in partition and "quer" in img)
                )
            ]

            self.image_ids.sort()
        else:
            self.image_ids = image_ids

        self.transform = transform
        self.masking = masking
        self.crop_resize = crop_resize

    def _masking(
        self, image: torch.Tensor, image_path: str, size: Tuple[int, int]
    ) -> torch.Tensor:
        """Applies a mask to the image based on bounding boxes defined in a text file.
        Args:
            image (torch.Tensor): The input image tensor.
            image_path (str): Path to the image file.
            size (Tuple[int, int]): Original size of the image.
        Returns:
            torch.Tensor: The masked image tensor.
        """
        mask = torch.zeros(size)
        bbox_path = image_path.replace(".jpg", "_bbox.txt")

        bboxes = np.loadtxt(bbox_path, dtype=int)
        bboxes = bboxes.reshape(-1, 4) if bboxes.ndim == 1 else bboxes

        for x, y, w, h in bboxes:
            mask[y : y + h, x : x + w] = 1

        mask = mask[None, None, ...]
        mask = F.interpolate(mask, size=image.shape[1:], mode="nearest")
        return image * mask[0]

    def _crop_resize(self, image: Image.Image, image_path: str) -> Image.Image:
        """
        Crops the image to the bounding box defined in the corresponding .txt file and returns the cropped image.

        Args:
            image (PIL.Image.Image): Input image.
            image_path (str): Path to the JPEG image. Expects a corresponding '_bbox.txt' file with bounding boxes.

        Returns:
            PIL.Image.Image: Cropped image.
        """
        img_arr = np.array(image)
        bbox_path = image_path.replace(".jpg", "_bbox.txt")

        bboxes = np.loadtxt(bbox_path, dtype=int)
        bboxes = bboxes.reshape(-1, 4) if bboxes.ndim == 1 else bboxes

        min_x = np.min(bboxes[:, 0]).clip(0, 10000)
        min_y = np.min(bboxes[:, 1]).clip(0, 10000)
        max_x = np.max(bboxes[:, 0] + bboxes[:, 2]).clip(0, 10000)
        max_y = np.max(bboxes[:, 1] + bboxes[:, 3]).clip(0, 10000)

        cropped_arr = img_arr[min_y:max_y, min_x:max_x]

        return Image.fromarray(cropped_arr)

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Loads and returns an image and its ID at the specified index.
        Applies preprocessing if a transform is provided.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: A preprocessed image tensor and its corresponding ID.
        """
        images_id = self.image_ids[idx]
        image_path = osp.join(self.root_dir, images_id)
        image_rgb = Image.open(image_path)
        image_rgb = image_rgb.convert("RGB")
        image_rgb = ImageOps.exif_transpose(image_rgb)
        org_size = image_rgb.size[::-1]

        if self.crop_resize:
            image_rgb = self._crop_resize(image_rgb, image_path)

        if self.transform is not None:
            image_rgb = self.transform(image_rgb)

        if self.masking:
            image_rgb = self._masking(image_rgb, image_path, org_size)

        return image_rgb, images_id
