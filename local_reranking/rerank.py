import argparse
import json
import os
import pickle

import torch
from torch import nn
from tqdm import tqdm

from src.ames import AMES
from src.utils import save_results_pkl, save_results_json, map_index, json_to_matrix
from src.tensor_dataset import get_test_loader


@torch.inference_mode()
def rerank(
    model: nn.Module,
    query_loader,
    gallery_loader,
    global_sims: torch.Tensor,
    global_ranks: torch.Tensor,
    lamb: float = 0.0,
    temp: float = 0.5,
    device: torch.device = torch.device("cpu"),
):
    scores = []
    for q_f, i in tqdm(query_loader):
        q_score = []
        gallery_loader.batch_sampler.sampler = global_ranks[i].T.tolist()
        for db_f, j in tqdm(gallery_loader):
            current_scores = model(
                *list(map(lambda x: x.to(device, non_blocking=True), q_f)),
                *list(map(lambda x: x.to(device, non_blocking=True), db_f)),
            )
            q_score.append(current_scores.cpu().data)
        scores.append(torch.stack(q_score).T)
    raw_sim = torch.cat(scores)

    s = 1.0 / (1.0 + torch.exp(-temp * raw_sim))
    s = lamb * global_sims + (1 - lamb) * s
    sims, indices = torch.sort(s, dim=-1, descending=True)
    local_ranks = torch.gather(global_ranks, -1, indices)

    return sims, local_ranks


def main(args):
    """
    Main function to perform k-NN search and evaluate the results.

    This function:
      - Loads the local features (queries, positives, and distractors).
      - Estimates new similarity for all pairs of query-positive image of a given initial ranking.
      - Reranks and saves the new ranking.
    """

    # Load dataset from HDF5 files.
    query_loader = get_test_loader([args.query_hdf5], args.batch_size, args.num_workers)
    db_files = [args.positive_hdf5]
    if args.distractor_hdf5 is not None:
        db_files += [
            args.distractor_hdf5.format(idx=f"{i:05d}")
            for i in range(args.total_distractors)
        ]
    db_loader = get_test_loader(
        db_files, args.batch_size, args.num_workers, args.db_desc_num
    )

    # Load global retrieval results
    if args.similarity_file.endswith(".pkl"):
        with open(args.similarity_file, "rb") as data:
            results_dict = pickle.load(data)
            sims = results_dict["sims"]
            ranks = results_dict["ranks"]
            ranks = map_index(ranks, results_dict["db_ids"], db_loader.dataset.index)
    elif args.similarity_file.endswith(".json"):
        with open(args.similarity_file, "r") as f:
            similarities = json.load(f)
        sims, ranks = json_to_matrix(
            similarities, query_loader.dataset.index, db_loader.dataset.index
        )

    # Load the model
    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )

    model_dict = torch.hub.load_state_dict_from_url(
        f"https://vrg.fel.cvut.cz/ilias_data/ames/{args.model_name}.pt",
        map_location=device,
    )

    model = AMES(binarized="dist" in args.model_name).eval().to(device)
    model.load_state_dict(model_dict["state"])

    sims_k, ranks_k = torch.from_numpy(sims[:, : args.k]), torch.from_numpy(
        ranks[:, : args.k]
    )
    local_sims, local_ranks = rerank(
        model,
        query_loader,
        db_loader,
        sims_k,
        ranks_k,
        lamb=args.lamb,
        temp=args.temp,
        device=device,
    )

    sims *= args.lamb / 2.0
    ranks[:, : args.k] = local_ranks
    sims[:, : args.k] = local_sims

    # Ensure the directory for saving results exists.
    os.makedirs(args.save_dir, exist_ok=True)

    # Save reranked results (query IDs, db IDs, similarity scores, and ranks).
    if args.save_as == "json":
        save_results_json(
            args.save_dir,
            args.save_name,
            query_loader.dataset.index,
            db_loader.dataset.index,
            sims,
            ranks,
        )
    elif args.save_as == "pickle":
        save_results_pkl(
            args.save_dir,
            args.save_name,
            query_loader.dataset.index,
            db_loader.dataset.index,
            sims,
            ranks,
        )


if __name__ == "__main__":
    # Set up the command-line argument parser with defaults.
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(
        prog, max_help_position=80
    )
    parser = argparse.ArgumentParser(
        description="Re-ranking with image local features.",
        formatter_class=formatter,
    )
    parser.add_argument(
        "--similarity_file",
        required=True,
        type=str,
        help="Path to the file with similarity scores and indices (.pkl or .json)",
    )
    parser.add_argument(
        "--model_name",
        default="dinov2_ames_dist",
        type=str,
        help="AMES model name to be loaded from torch hub.",
    )
    parser.add_argument(
        "--query_hdf5",
        required=True,
        type=str,
        help="Path to HDF5 file containing the local features of the ILIAS queries",
    )
    parser.add_argument(
        "--positive_hdf5",
        required=True,
        type=str,
        help="Path to HDF5 file containing the local features of the ILIAS positives",
    )
    parser.add_argument(
        "--distractor_hdf5",
        type=str,
        default=None,
        help="Path to HDF5 file containing the local features of the ILIAS distractors",
    )
    parser.add_argument(
        "--total_distractors",
        type=int,
        default=1,
        help="Number of HDF5 files of distractors to be loaded. This depends on the settings used during feature extraction.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Number of nearest neighbors to rerank",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=0.35,
        help="Lambda parameter of the convex combination of global and local similarities",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.1,
        help="Temperature parameter scaling the local similarity distribution",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Whether to use GPU",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of process workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=300,
        help="Number of image pairs processed at once",
    )
    parser.add_argument(
        "--db_desc_num",
        type=int,
        default=100,
        help="Number of local features to use for each database image",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to store the results",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="results",
        help="File name to store the results",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="json",
        choices=["json", "pickle"],
        help="Format to save the results",
    )

    args = parser.parse_args()
    main(args)
