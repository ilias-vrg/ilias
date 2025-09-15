import os
import argparse
import numpy as np
import pickle as pk
import torch

from src.uned_dataset import UnEDFeatureDataset
from src.utils import bool_flag
from src.lin_adapt import train_lin_adapt_layer


def main(args):
    """
    Main function to train the linear adaptation layer.
    """
    # Load dataset from HDF5 files.
    dataset = UnEDFeatureDataset(
        args.uned_features_path,
        args.uned_info_path,
    )

    print(f"will train on total samples: {dataset.uned_feat.shape[0]}")

    num_epochs = args.num_epochs

    print(f"will train for total num epochs: {num_epochs}")

    adaptation_layer = train_lin_adapt_layer(
        dataset,
        adaptation_layer_dim=args.adaptation_layer_dim,
        num_epochs=num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu,
    )

    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
    torch.save(adaptation_layer.state_dict(), args.save_dir)

    print(f"trained linear adaptation layer is saved to: {args.save_dir}")


if __name__ == "__main__":
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=80
    )
    parser = argparse.ArgumentParser(
        description="Code for training a linear adaptation layer on the UnED dataset",
        formatter_class=formatter,
    )
    parser.add_argument(
        "--uned_features_path",
        type=str,
        help="Path to HDF5 file containing the features of the UnED training set",
    )
    parser.add_argument(
        "--uned_info_path",
        type=str,
        help="Path to HDF5 file containing the metadata of the UnED training set",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs to train the linear adaptation layer",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training the linear adaptation layer",
    )
    parser.add_argument(
        "--adaptation_layer_dim",
        type=int,
        default=512,
        help="Output dimension of the linear adaptation layer",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to PTH file that the trained linear adaptation layer will be saved to",
    )
    parser.add_argument(
        "--use_gpu",
        type=bool_flag,
        default=True,
        help="Whether to use GPU to perform the training",
    )
    args = parser.parse_args()
    main(args)
