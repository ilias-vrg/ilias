import os
import json
import h5py
import numpy as np
import pickle as pk

from collections import defaultdict


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


def load_features(file_name: str):
    """
    Load features and their corresponding IDs from an HDF5 file.

    Args:
        file_name (str): Path to the HDF5 file containing features and IDs.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the features and their corresponding IDs.

    Raises:
        ValueError: If the file is not a valid HDF5 file.
    """
    if h5py.is_hdf5(file_name):
        f = h5py.File(file_name, "r")
        return f["features"], f["index"].asstr()
    else:
        raise ValueError(f"File {file_name} is not a valid HDF5 file.")


def reindex(I, idx):
    """
    Reindex the input array `I` based on the provided index mapping `idx`.
    This function replaces the values in `I` with their corresponding indices from `idx`.
    Args:
        I (np.ndarray): Input array to be reindexed.
        idx (list): List of indices corresponding to the unique values in `I`.
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
