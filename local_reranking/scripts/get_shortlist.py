import argparse
import json
import os
import numpy as np
import pickle


def write_shortlist(paths, save_dir, save_name):
    with open(os.path.join(save_dir, save_name), "wt") as fid:
        fid.write("\n".join(map(str, paths)))


def main(args):

    if args.similarity_file.endswith(".pkl"):
        with open(args.similarity_file, "rb") as data:
            results_dict = pickle.load(data)
            db_paths = results_dict["db_ids"]
    elif args.similarity_file.endswith(".json"):
        with open(args.similarity_file, "r") as f:
            similarities = json.load(f)
        db_paths = [k for v in similarities.values() for k in v]

    db_paths = np.unique(db_paths)
    mask = np.char.find(db_paths, "yfcc") >= 0
    dist = db_paths[mask]
    pos = db_paths[~mask]

    write_shortlist(pos, args.save_dir, "positive_shortlist.txt")
    dist = [p.split("/")[1] for p in dist]
    write_shortlist(dist, args.save_dir, "distractor_shortlist.txt")


if __name__ == "__main__":
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=80
    )
    parser = argparse.ArgumentParser(
        description="Re-ranking with image local features",
        formatter_class=formatter,
    )
    parser.add_argument(
        "--similarity_file",
        required=True,
        type=str,
        help="Path to the file with similarity scores and indices (.pkl or .json)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to store the positive and distractor shortlist.",
    )

    args = parser.parse_args()
    main(args)
