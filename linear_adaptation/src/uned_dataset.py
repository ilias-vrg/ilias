from typing import defaultdict

from src.utils import load_features


class UnEDFeatureDataset:
    """
    Custom Dataset Class for features for the training set of UnED.
    """

    def __init__(
        self,
        uned_hdf5: str,
        uned_info_path: str,
    ) -> None:
        """
        Initialize the FeatureDataset by loading features from HDF5 files.

        Args:
            uned_hdf5 (str): Path to the HDF5 file containing UnED features.
        """
        print("> loading features from HDF5 files...")

        self.uned_feat, self.uned_ids = load_features(uned_hdf5)

        with open(uned_info_path, "r") as f:
            lines = f.readlines()

        self.paths, self.labels, self.domains = [], [], []
        for line in lines:
            path, label, domain = line.strip().split(",")
            self.paths.append(path)
            self.labels.append(int(label))
            self.domains.append(int(domain))

        # Create a mapping from (domain_index, class_index) to a unique (increasing) index
        self.combination_to_index = {}
        self.index_to_combination = {}

        index = 0
        for class_index, domain_index in zip(self.labels, self.domains):
            if (class_index, domain_index) not in self.combination_to_index:
                self.combination_to_index[(class_index, domain_index)] = index
                self.index_to_combination[index] = (class_index, domain_index)
                index += 1

        self.total_num_classes = index

        # Create a dictionary to store the total number of unique classes per domain
        self.total_classes_per_domain = defaultdict(set)

        for class_index, domain_index in zip(self.labels, self.domains):
            self.total_classes_per_domain[domain_index].add(class_index)

        # Convert sets to counts
        self.total_classes_per_domain = {
            k: len(v) for k, v in self.total_classes_per_domain.items()
        }

        print(
            f"Total number of classes per UnED domain: {self.total_classes_per_domain}"
        )

        # Encode function: (domain_index, class_index) -> unique index
        def encode_combination(domain_index, class_index):
            return self.combination_to_index[(domain_index, class_index)]

        # Encode the labels and domains into unique indices
        self.encoded_indices = [
            encode_combination(c, d) for c, d in zip(self.labels, self.domains)
        ]
