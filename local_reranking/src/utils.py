import json
import os
import pickle as pk
from collections import defaultdict

import numpy as np


def json_to_matrix(data, query_ids, db_ids, top_k=None):
    query_to_index = {key: i for i, key in enumerate(query_ids)}
    db_to_index = {key: i for i, key in enumerate(db_ids)}

    num_queries = len(query_ids)
    try:
        first_query_key = next(iter(data))
        if top_k is None:
            top_k = len(data[first_query_key])
            print(f"Found {num_queries} queries and inferred top_k={top_k}.")
        else:
            print(f"Found {num_queries} queries, using top_k={top_k}.")
    except StopIteration:
        print("Error: JSON file is empty. Cannot determine top_k.")
        return

    similarities = np.zeros((num_queries, top_k), dtype=np.float32)
    ranks = np.zeros((num_queries, top_k), dtype=np.int64)

    for query_key, inner_dict in data.items():
        if query_key not in query_to_index:
            print(
                f"Warning: Query key '{query_key}' from JSON not found in query_ids. Skipping."
            )
            continue

        i = query_to_index[query_key]

        # Iterate over the db_key: score pairs
        # Python 3.7+ guarantees dicts preserve insertion order
        for j, (db_key, similarity_score) in enumerate(inner_dict.items()):
            if j >= top_k:
                print(
                    f"Warning: Query '{query_key}' has more than {top_k} items. Truncating."
                )
                break

            if db_key not in db_to_index:
                print(
                    f"Warning: DB key '{db_key}' for query '{query_key}' not found in db_ids. Skipping."
                )
                continue

            db_index = db_to_index[db_key]
            similarities[i, j] = similarity_score
            ranks[i, j] = db_index

    return similarities, ranks


def map_index(I, old_idx, new_idx):
    """
    Re-indices I by mapping old index to new index.

    Args:
        I (np.ndarray): The array containing indices of old_idx.
        old_idx (list): List of strings in the original order.
        new_idx (list): List of strings in the new order.
    """
    new_lookup = {path: i for i, path in enumerate(new_idx)}
    missing_paths = [p for p in old_idx if p not in new_lookup]
    if missing_paths:
        raise ValueError(
            f"Found {len(missing_paths)} items in global shortlist not present in local feature index. "
            f"First missing: {missing_paths[0]}"
        )
    translation = np.array([new_lookup[path] for path in old_idx])
    new_I = translation[I]
    return new_I


def reindex(I, idx):
    """
    Reindex the input array `I` based on the provided index mapping `idx`.
    This function replaces the values in `I` with their corresponding indices from `idx`.
    Args:
        I (np.ndarray): Input array to be reindexed.
        idx (np.ndarray): List of indices corresponding to the unique values in `I`.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the reindexed array and the new index mapping.
    """
    unique_vals, inverse = np.unique(I, return_inverse=True)
    new_I = inverse.reshape(I.shape)
    new_idx = np.array([idx[val] for val in unique_vals])
    return new_I, new_idx


def save_results_json(save_dir, save_name, query_ids, db_ids, sims, ranks):
    """
    Save the search results in JSON format.
    Args:
        save_dir (str): Directory where the results will be saved.
        save_name (str): Name of the file to save the results.
        query_ids (np.ndarray): Array of query IDs.
        db_ids (np.ndarray): Array of database IDs.
        sims (np.ndarray): Similarity scores for each query-database pair.
        ranks (np.ndarray): Ranks of the database items for each query.
    """
    mapping = {i: img for i, img in enumerate(db_ids)}
    results = defaultdict(dict)
    for q, s, r in zip(query_ids, sims, ranks):
        for i in range(len(s)):
            db = mapping[r[i]]
            results[q][db] = s[i].item()

    save_file = os.path.join(save_dir, f"{save_name}.json")
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nsimilarities saved in file: {save_file}")


def save_results_pkl(save_dir, save_name, query_ids, db_ids, sims, ranks):
    """
    Save the search results in pickle format.
    Args:
        save_dir (str): Directory where the results will be saved.
        save_name (str): Name of the file to save the results.
        query_ids (np.ndarray): Array of query IDs.
        db_ids (np.ndarray): Array of database IDs.
        sims (np.ndarray): Similarity scores for each query-database pair.
        ranks (np.ndarray): Ranks of the database items for each query.
    """
    ranks, db_ids = reindex(ranks, db_ids)

    save_file = os.path.join(save_dir, f"{save_name}.pkl")
    with open(save_file, "wb") as f:
        results = {
            "query_ids": query_ids,
            "db_ids": db_ids,
            "sims": sims,
            "ranks": ranks,
        }
        pk.dump(results, f)
    print(f"\nsimilarities saved in file: {save_file}")
