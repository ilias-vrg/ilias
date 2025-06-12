import os
import argparse
import numpy as np
import pickle as pk

from src.searcher import Searcher
from src.dataset import FeatureDataset
from src.utils import save_results_json, save_results_pkl, bool_flag


def main(args):
    """
    Main function to perform k-NN search and evaluate the results.

    This function:
      - Loads the dataset (queries, positives, and distractors).
      - Performs search in the dataset given the queries.
      - Saves the search results.
    """
    if args.use_gpu and args.k > 1000:
        print(
            "\nWarning: FAISS gpu does not support k values >1000. Switching to cpu.\n"
        )
        args.use_gpu = False

    # Load dataset from HDF5 files.
    dataset = FeatureDataset(
        args.query_hdf5,
        args.positive_hdf5,
        distractor_hdf5=args.distractor_hdf5,
        total_distractors=args.total_distractors,
        selected=args.selected,
    )

    # Initialize the searcher with the provided parameters.
    knn = Searcher(
        dataset,
        k=args.k,
        lin_adopt=args.lin_adopt_path,
        use_gpu=args.use_gpu,
        query_expansion=args.query_expansion,
    )

    # Retrieve query features and corresponding IDs.
    query_feat, query_ids = dataset.get_queries()
    db_ids = dataset.get_db_ids()

    # Perform the k-NN search.
    sims, ranks = knn.search(query_feat)

    # Ensure the directory for saving results exists.
    os.makedirs(args.save_dir, exist_ok=True)

    # Save primary search results (query IDs, db IDs, similarity scores, and ranks).
    if args.save_as == "json":
        save_results_json(args.save_dir, args.save_name, query_ids, db_ids, sims, ranks)
    elif args.save_as == "pickle":
        save_results_pkl(args.save_dir, args.save_name, query_ids, db_ids, sims, ranks)


if __name__ == "__main__":
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=80
    )
    parser = argparse.ArgumentParser(
        description="Code for knn search and evaluation on ILIAS datasets",
        formatter_class=formatter,
    )
    parser.add_argument(
        "--query_hdf5",
        type=str,
        help="Path to HDF5 file containing the features of the ILIAS queries",
    )
    parser.add_argument(
        "--positive_hdf5",
        type=str,
        help="Path to HDF5 file containing the features of the ILIAS positives",
    )
    parser.add_argument(
        "--distractor_hdf5",
        type=str,
        default=None,
        help="Path to HDF5 file containing the features of the ILIAS distractors",
    )
    parser.add_argument(
        "--total_distractors",
        type=int,
        default=1,
        help="Number of HDF5 files of distractors to be loaded. This depends on the settings used during feature extraction.",
    )
    parser.add_argument(
        "--selected",
        type=str,
        default=None,
        help="File of selected image IDs to filter the distractors",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Number of nearest neighbors to find for each query",
    )
    parser.add_argument(
        "--query_expansion",
        type=int,
        default=0,
        help="Number of nearest neighbors that will form the shortlist for re-ranking",
    )
    parser.add_argument(
        "--lin_adopt_path",
        type=str,
        default=None,
        help="Path to the file where the linear adaptation layer is stored",
    )
    parser.add_argument(
        "--use_gpu",
        type=bool_flag,
        default=True,
        help="Whether to use GPU to perform the search",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to store the output similarities",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="similarities",
        help="File name to store the output similarities",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="json",
        choices=["json", "pickle"],
        help="Format to save the output similarities",
    )

    args = parser.parse_args()
    main(args)
