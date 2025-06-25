import h5py
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from src import utils
from src.datasets.generators import (
    get_wds_dataset,
    get_image_dataset,
    get_intstance_descriptions,
)
from src.datasets.transforms import InferenceTransforms
from src.model.feature_extractor import FeatureExtractor
from src.model.pooling import PoolingFn


@torch.no_grad()
def extract_features(
    extractor: nn.Module, loader: DataLoader, args
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from images using the specified feature extractor.

    Args:
        extractor (nn.Module): The feature extractor model.
        loader (DataLoader): Dataloader providing batches of images and their IDs.
        args: Command-line arguments containing configurations such as batch size,
              multiscale, multirotate, and float precision settings.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - A NumPy array containing the extracted features.
            - A list of image IDs corresponding to each extracted feature.
    """
    pbar = tqdm(loader, desc="extracting")
    index, features = [], []

    for batches in pbar:
        # The loader returns a tuple where the first element contains images
        # and the second element contains image IDs.
        for images, image_ids in zip(*batches):
            # Process images in sub-batches defined by args.batch_size.
            for bimgs in utils.batching(images, args.batch_size):
                bimgs = bimgs.cuda()
                # Use automatic mixed precision if enabled.
                with torch.cuda.amp.autocast(enabled=args.comp_fp16):
                    scales = [0.707, 1, 1.414] if args.multiscale else [1]
                    rotations = [0, 90, 180, 270] if args.multirotate else [0]
                    feat = extractor(bimgs, scales, rotations)
                # Optionally cast features to half-precision before returning.
                if args.store_fp16:
                    feat = feat.half()
                features.append(feat.cpu().numpy())
            index.extend(image_ids)
        pbar.set_postfix(last_image=index[-1], total=len(index))
    return np.concatenate(features, axis=0), index


@torch.no_grad()
def extract_text_features(model: nn.Module, descriptions: List[str]) -> np.ndarray:
    """
    Extract features from descriptions using the specified model and tokenizer.

    Args:
        model (nn.Module): The model to use for feature extraction.
        descriptions (List[str]): A list of text descriptions to extract features from.

    Returns:
        np.ndarray: A NumPy array containing the extracted features.
    """
    features = model(descriptions)

    return features.cpu().numpy()


def store_features(
    features: np.ndarray, index: List[str], hdf5_file_name: Path, hdf5_compression: int
) -> None:
    """
    Store extracted features and their corresponding image IDs in an HDF5 file.

    Args:
        features (np.ndarray): The extracted features as a NumPy array.
        index (List[str]): A list of image IDs corresponding to the features.
        hdf5_file_name (Path): Path to the HDF5 file where features will be stored.

    Returns:
        None
    """
    print(f"\n> store features in file: {hdf5_file_name}")
    start = time.time()
    with h5py.File(hdf5_file_name, "w") as hdf5_file:
        hdf5_file.create_dataset(
            "index", data=index, dtype=h5py.string_dtype(encoding="ascii")
        )
        hdf5_file.create_dataset(
            "features",
            data=features,
            dtype=features.dtype,
            compression="gzip",
            compression_opts=hdf5_compression,
        )
    duration = time.time() - start
    print(f"feature storage completed in {duration:.2f}s (dtype: {features.dtype})")


def extract_yfcc(extractor: nn.Module, preprocessing: transforms.Compose, args) -> None:
    """
    Extract features for the YFCC100M dataset and store them in separate HDF5 files.

    Args:
        extractor (nn.Module): The feature extractor model.
        preprocessing (transforms.Compose): Preprocessing transformations applied to the images.
        args: Command-line arguments containing dataset paths, batch sizes, and other configurations.

    Returns:
        None
    """
    assert (
        args.start_tar % args.tar_batch_size == 0
    ), "'start_tar' value must be divisible by 'tar_batch_size'"

    target_dir = Path(args.hdf5_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tar_list = sorted([str(p) for p in Path(args.dataset_dir).glob("*.tar")])
    if args.total_tars is None:
        args.total_tars = len(tar_list)
    tar_list = tar_list[args.start_tar : args.start_tar + args.total_tars]
    print(f"\n> total tar files to be processed: {len(tar_list)}")

    processed_tars = []
    for i, tar_batch in enumerate(utils.batching(tar_list, args.tar_batch_size)):
        hdf5_id = i + args.start_tar // args.tar_batch_size
        hdf5_name = target_dir / f"{args.hdf5_name}_{args.partition}_{hdf5_id:05d}.hdf5"

        if hdf5_name.exists() and not args.overwrite:
            print(f"> existing feature file: {hdf5_name}")
            continue

        print(f"> extract features for {len(tar_batch)} tar files")
        loader, dataset = get_wds_dataset(
            tar_batch,
            preprocess_img=preprocessing,
            batch_size=args.webdl_batch_size,
            num_workers=args.num_workers,
            selected_ids=args.selected_ids,
        )

        features, index = extract_features(extractor, loader, args)
        store_features(features, index, hdf5_name, args.hdf5_compression)

        processed_tars.extend(tar_batch)
        print(f"> {len(processed_tars)}/{len(tar_list)} tar files processed")


def extract_ilias(
    extractor: nn.Module, preprocessing: transforms.Compose, args
) -> None:
    """
    Extract features for the ILIAS dataset and store them in an HDF5 file.

    Args:
        extractor (nn.Module): The feature extractor model.
        preprocessing(transforms.Compose): Preprocessing transformations applied to the images.
        args: Command-line arguments containing dataset paths, batch sizes, and other configurations.

    Returns:
        None
    """
    target_dir = Path(args.hdf5_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    loader, dataset = get_image_dataset(
        args.dataset_dir,
        preprocess_img=preprocessing,
        batch_size=args.webdl_batch_size,
        num_workers=args.num_workers,
        partition=args.partition,
        selected_ids=args.selected_ids,
        masking=args.masking,
        crop_resize=args.crop_resize,
    )

    print(f"\n> extract features for {len(dataset)} images")
    features, index = extract_features(extractor, loader, args)

    if args.selected_ids is None:
        idx = np.argsort(index)
    else:
        # Create an ordering based on input selected_ids.
        idx_map = {img: i for i, img in enumerate(index)}
        idx = [idx_map[img] for img in args.selected_ids]
    index = [index[i] for i in idx]
    features = features[idx]

    # Prepare the HDF5 file name.
    hdf5_file = target_dir / f"{args.hdf5_name}_{args.partition}.hdf5"

    # Encode image paths to avoid conflicts.
    index = [
        str(image_pth.encode("unicode-escape"))
        .replace("\\\\u", "#U")
        .replace("\\\\x", "#U00")[2:-1]
        for image_pth in index
    ]

    store_features(features, index, hdf5_file, args.hdf5_compression)


def extract_descriptions(extractor: nn.Module, args) -> None:
    """
    Extract features for text descriptions using the specified model and store them in an HDF5 file.

    Args:
        extractor (nn.Module): The feature extractor model.
        args: Command-line arguments containing dataset paths, batch sizes, and other configurations.

    Returns:
        None
    """
    target_dir = Path(args.hdf5_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    instances, descriptions = get_intstance_descriptions(args.dataset_dir)

    print(f"\n> extract features for {len(descriptions)} text descriptions")
    features = extract_text_features(extractor, descriptions)

    # Prepare the HDF5 file name.
    hdf5_file = target_dir / f"{args.hdf5_name}_{args.partition}.hdf5"

    store_features(features, instances, hdf5_file, args.hdf5_compression)


def main(args) -> None:
    """
    Main function to orchestrate feature extraction for the specified dataset.

    Loads the feature extractor model, applies preprocessing, and then triggers
    extraction and storage for either the YFCC100M or ILIAS dataset based on the partition.

    Args:
        args: Command-line arguments containing configurations for model, dataset, and extraction.
    """
    print("\n> loading model")
    extractor = FeatureExtractor[args.framework.upper()].get_model(**vars(args))
    extractor = extractor.cuda().eval()

    if args.partition == "text_queries":
        return extract_descriptions(extractor, args)

    preprocessing = InferenceTransforms[args.transforms.upper()].get_transforms(
        mean=extractor.mean, std=extractor.std, **vars(args)
    )
    print("\n> preprocessing transforms:")
    print(preprocessing)

    # Load image ids from file if specified.
    if args.selected_ids is not None:
        args.selected_ids = utils.load_image_ids(args.selected_ids)

    if args.partition == "distractors":
        return extract_yfcc(extractor, preprocessing, args)
    else:
        return extract_ilias(extractor, preprocessing, args)


if __name__ == "__main__":
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=90
    )
    parser = argparse.ArgumentParser(
        description="Feature extraction for ILIAS datasets",
        formatter_class=formatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the images or tar files of the dataset",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="image_queries",
        choices=["image_queries", "text_queries", "positives", "distractors", "uned"],
        help="Dataset partition to extract features from",
    )
    parser.add_argument(
        "--hdf5_dir",
        type=str,
        required=True,
        help="Directory where the HDF5 feature files will be stored",
    )
    parser.add_argument(
        "--hdf5_name",
        type=str,
        default="features",
        help="Base name for the generated HDF5 files",
    )
    parser.add_argument(
        "--hdf5_compression",
        type=int,
        default=0,
        help="Compression level for HDF5 storage (gzip options)",
    )
    parser.add_argument(
        "--overwrite",
        type=utils.bool_flag,
        default=False,
        help="Overwrite existing feature files if set",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="timm",
        choices=[x.name.lower() for x in FeatureExtractor],
        help="Feature extraction framework to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_large_patch16_siglip_384.webli",
        help="Model backbone for feature extraction",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="head",
        choices=[x.name.lower() for x in PoolingFn],
        help="Pooling function for feature extraction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size used for feature extraction",
    )
    parser.add_argument(
        "--webdl_batch_size",
        type=int,
        default=128,
        help="Batch size for the DataLoader",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default="RESIZE_LARGE",
        choices=[x.name for x in InferenceTransforms],
        help="Name of the transforms to apply for preprocessing",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resize resolution for image preprocessing",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=0,
        help="Minimum size for image resizing in preprocessing",
    )
    parser.add_argument(
        "--multiscale",
        type=utils.bool_flag,
        default=False,
        help="Use multiscale feature extraction if True",
    )
    parser.add_argument(
        "--multirotate",
        type=utils.bool_flag,
        default=False,
        help="Use multirotation feature extraction if True",
    )
    parser.add_argument(
        "--masking",
        type=utils.bool_flag,
        default=False,
        help="Use masking feature extraction if True",
    )
    parser.add_argument(
        "--crop_resize",
        type=utils.bool_flag,
        default=False,
        help="Use crop and resize feature extraction if True",
    )
    parser.add_argument(
        "--comp_fp16",
        type=utils.bool_flag,
        default=True,
        help="Use mixed precision (FP16) for feature extraction",
    )
    parser.add_argument(
        "--start_tar",
        type=int,
        default=0,
        help="Index of the first tar file to process",
    )
    parser.add_argument(
        "--total_tars",
        type=int,
        default=None,
        help="Total number of tar files to process (None means all)",
    )
    parser.add_argument(
        "--tar_batch_size",
        type=int,
        default=10,
        help="Number of tar files to process per feature file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers used for data loading",
    )
    parser.add_argument(
        "--store_fp16",
        type=utils.bool_flag,
        default=True,
        help="Store extracted features in FP16 if True",
    )
    parser.add_argument(
        "--selected_ids",
        type=str,
        default=None,
        help="File containing image IDs to extract features for",
    )
    args = parser.parse_args()

    print("\nInput arguments")
    print("---------------")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")

    main(args)
