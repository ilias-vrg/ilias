import numpy as np
from typing import Tuple, Optional, List, Set

from torch.utils.data import Dataset
from src.utils import load_features


class FeatureDataset(Dataset):
    """
    PyTorch Dataset for feature retrieval from positive and distractor HDF5 files.
    """

    def __init__(
        self,
        query_hdf5: str,
        positive_hdf5: str,
        distractor_hdf5: Optional[str] = None,
        total_distractors: int = 0,
        selected: Optional[str] = None,
    ) -> None:
        """
        Initialize the FeatureDataset by loading features from HDF5 files.

        Args:
            query_hdf5 (str): Path to the HDF5 file containing query features.
            positive_hdf5 (str): Path to the HDF5 file containing positive example features.
            distractor_hdf5 (Optional[str]): Template path for distractor HDF5 files (must contain a placeholder '{id}').
            total_distractors (int): Total number of distractor HDF5 files to load.
            selected (Optional[str]): Path to a file containing selected image IDs for filtering.
        """
        print("> loading features from HDF5 files...")
        self.query_feat, self.query_ids = load_features(query_hdf5)
        self.positive_feat, self.positive_ids = load_features(positive_hdf5)

        self.distractor_feat = []
        self.distractor_ids = []
        if distractor_hdf5 is not None:
            for i in range(total_distractors):
                # Use the template with string formatting to replace the id placeholder.
                feats, idx = load_features(distractor_hdf5.format(idx=f"{i:05d}"))
                self.distractor_feat.append(feats)
                self.distractor_ids.append(idx[:])

        self.num_pos = len(self.positive_ids)
        self.chunk_size = len(self.distractor_ids[0]) if self.distractor_ids else 0
        self.db_ids: np.ndarray = np.concatenate(
            [self.positive_ids, *self.distractor_ids], axis=0
        )

        self.selected = None
        if selected is not None:
            # Convert positive IDs to a set for faster membership tests.
            self.pos_ids = set(self.positive_ids)
            self.selected = set(np.loadtxt(selected, dtype=str).tolist())
            # Filter the database IDs according to the selected set.
            self.db_ids, _ = self.filter(self.db_ids, self.selected)

        print(f"number of queries: {len(self.query_ids)}")
        print(f"number of database: {len(self.db_ids)}")

    def filter(
        self, idx: np.ndarray, selected: Set[str]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Filter the provided indices based on selected image IDs.

        Args:
            idx (np.ndarray): Array of image IDs.
            selected (Set[str]): Set of selected image IDs.

        Returns:
            Tuple[np.ndarray, List[int]]: A tuple containing the filtered array of IDs and
            a list of the positions (indices) that passed the filter.
        """
        filtered_positions = []
        for i, image_id in enumerate(idx):
            parts = image_id.split("/")
            if image_id in self.pos_ids or (len(parts) > 1 and parts[1] in selected):
                filtered_positions.append(i)
        filtered_idx = idx[filtered_positions]
        return filtered_idx, filtered_positions

    def get_queries(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the query features and their corresponding IDs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the query features
            and the query IDs.
        """
        return self.query_feat[:], self.query_ids[:]

    def get_positives(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the positive features and their corresponding IDs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the positive features
            and the positive IDs.
        """
        return self.positive_feat[:], self.positive_ids[:]

    def get_db_ids(self) -> np.ndarray:
        """
        Retrieve the database IDs, which include positive and distractor IDs.

        Returns:
            np.ndarray: An array containing the positive and distractor IDs.
        """
        return self.db_ids

    def __len__(self) -> int:
        """
        Return the total number of feature batches in the dataset.

        Returns:
            int: Number of batches (1 for positives plus one per distractor feature set).
        """
        # One batch for positive features and one for each distractor file.
        return 1 + len(self.distractor_ids)

    def __getitem__(self, idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve the feature array and corresponding IDs at the given index.

        If filtering is applied and no elements pass the filter, returns empty arrays.

        Args:
            index (int): Desired batch index (0 corresponds to positive features,
                subsequent indices correspond to distractor features).

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: A tuple with a feature array and its IDs,
            or None if no valid data is available.
        """
        if idx == 0:
            return self.get_positives()

        # Adjust index for distractor features (first distractor is at index 1).
        db_feat = self.distractor_feat[idx - 1][:]
        db_ids = self.distractor_ids[idx - 1][:]
        if self.selected is not None:
            _, dist_idx = self.filter(db_ids, self.selected)
            if not len(dist_idx):
                return np.array([]), np.array([])
            return db_feat[dist_idx], db_ids[dist_idx]
        return db_feat, db_ids
