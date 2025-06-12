import json
import pickle as pk
import argparse

from src.dataset import ILIAS


def main(args):
    """
    Evaluate mAP and Oracle metrics for the ILIAS dataset.

    This function loads similarity scores from a result file
    (supports both pickle and JSON formats), evaluates the
    results using the ILIAS dataset, and prints the mAP and Oracle scores.
    """
    # Initialize the ILIAS dataset with the specified configuration.
    print(f"> loading ILIAS dataset from: {args.dataset_dir}")
    dataset = ILIAS(args.dataset_dir, check_ids=args.check_ids)
    print(f"> evaluating results from {args.similarity_file} ...")

    # Load results depending on the file format.
    if args.similarity_file.endswith(".pkl"):
        with open(args.similarity_file, "rb") as f:
            similarities = pk.load(f)
        report = dataset.evaluate(
            **similarities, k=args.k, report=args.result_file is not None
        )
    elif args.similarity_file.endswith(".json"):
        with open(args.similarity_file, "r") as f:
            similarities = json.load(f)
        report = dataset.evaluate(
            all_similarities=similarities, k=args.k, report=args.result_file is not None
        )
    else:
        raise ValueError(
            "Unsupported file format. Use .pkl or .json for the result file."
        )

    # Print the evaluation metrics.
    print(f"\n> performance in {len(report['queries'])} queries")
    print(f"mAP: {report['map']:.2f}%")

    if args.result_file is not None:
        with open(args.result_file, "wb") as f:
            pk.dump(report, f)
        print(f"report is saved in: {args.result_file}")


if __name__ == "__main__":
    # Set up the command-line argument parser with defaults.
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=80
    )
    parser = argparse.ArgumentParser(
        description="mAP evaluation for the ILIAS dataset",
        formatter_class=formatter,
    )
    parser.add_argument(
        "--similarity_file",
        required=True,
        type=str,
        help="Path to the file with similarity scores and indices (.pkl or .json)",
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        type=str,
        help="Path to the dataset directory containing the image ID files",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        help="Path to the file where results will be stored",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Number of top results to consider for mAP calculation",
    )
    parser.add_argument(
        "--check_ids",
        action="store_true",
        help="Load and validate distractor image IDs from the dataset if specified",
    )
    args = parser.parse_args()
    main(args)
