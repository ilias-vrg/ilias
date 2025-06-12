import faiss
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List

from src.lin_adapt import load_lin_adopt_layer


class Searcher:
    def __init__(
        self,
        db: Dataset,
        k: int = 1000,
        use_gpu: bool = True,
        lin_adopt: Optional[str] = None,
        query_expansion: int = 0,
        metric_type: int = faiss.METRIC_INNER_PRODUCT,
    ):
        """
        Initialize the Searcher to perform k-NN search using FAISS.

        Args:
            db (Dataset): Database embeddings.
            k (int): Number of nearest neighbors to retrieve.
            use_gpu (bool): If True, perform searches on GPU.
            lin_adopt (Optional[str]): Path to a linear adaptation layer model.
            query_expansion (int): Number of neighbors to use for query expansion.
            metric_type (int): FAISS metric type (e.g., faiss.METRIC_INNER_PRODUCT).
        """
        self.db = db
        self.k = k
        self.use_gpu = use_gpu
        self.lin_adopt = load_lin_adopt_layer(lin_adopt, use_gpu)
        self.query_expansion = query_expansion
        self.metric_type = metric_type

    @staticmethod
    def is_similarity_metric(metric_type: int) -> bool:
        """
        Check if the given metric type represents a similarity metric.

        Args:
            metric_type (int): Metric type to check.

        Returns:
            bool: True if the metric is a similarity metric; otherwise, False.
        """
        return metric_type in {faiss.METRIC_INNER_PRODUCT}

    def to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a torch.Tensor on the appropriate device.

        Args:
            x (np.ndarray): Input array.

        Returns:
            torch.Tensor: Converted tensor on 'cuda' if GPU is used, otherwise on CPU.
        """
        tensor = torch.from_numpy(x).float()
        return tensor.to("cuda") if self.use_gpu else tensor

    @torch.no_grad()
    def adapt(self, x: np.ndarray) -> np.ndarray:
        """
        Optionally adapt input embeddings using a linear adaptation layer.

        Args:
            x (np.ndarray): Input embeddings.

        Returns:
            np.ndarray: Adapted embeddings.
        """
        x = x.astype(np.float32)
        if self.lin_adopt is not None:
            x = self.to_tensor(x)
            x = self.lin_adopt(x)
            x = x.cpu().numpy()
        return x

    def _search(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform k-NN search over the database for the provided query embeddings.

        Args:
            queries (np.ndarray): Query embeddings of shape (num_queries, dim).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - D: Array of distances (or similarities) with shape (num_queries, k).
                - I: Array of indices of nearest neighbors with shape (num_queries, k).
        """
        print(f"\n> search for {queries.shape[0]} queries in the database...")

        loader = DataLoader(
            self.db,
            num_workers=4,
            collate_fn=lambda x: x[0],
        )

        gpu_res = faiss.StandardGpuResources() if self.use_gpu else None

        result = faiss.ResultHeap(
            queries.shape[0],
            self.k,
            keep_max=self.is_similarity_metric(self.metric_type),
        )

        offset = 0
        for chunk_feat, _ in tqdm(loader, desc="searching"):
            if chunk_feat.shape[0] == 0:
                continue
            chunk_feat = self.adapt(chunk_feat)
            if self.use_gpu and gpu_res is not None:
                D_chunk, I_chunk = faiss.knn_gpu(
                    gpu_res, queries, chunk_feat, self.k, metric=self.metric_type
                )
            else:
                D_chunk, I_chunk = faiss.knn(
                    queries, chunk_feat, self.k, metric=self.metric_type
                )
            # Adjust indices by the current offset and add results.
            result.add_result(D_chunk, I_chunk + offset)
            offset += chunk_feat.shape[0]

        result.finalize()
        return result.D, result.I

    def _query_expansion(
        self,
        queries: np.ndarray,
        shortlist: List[List[int]],
        weights: List[List[float]],
    ) -> np.ndarray:
        """
        Expand query embeddings using a weighted combination of neighbor embeddings.

        Args:
            queries (np.ndarray): Original query embeddings of shape (num_queries, dim).
            shortlist (List[List[int]]): List of neighbor indices for each query.
            weights (List[List[float]]): List of weights for each neighbor for each query.

        Returns:
            np.ndarray: Expanded and normalized query embeddings.
        """
        num_pos = self.db.num_pos
        chunk_size = self.db.chunk_size

        expanded_embeds = []
        for neigh_indices, neigh_weights in zip(shortlist, weights):
            neighbors_embeds = []
            for n, w in zip(neigh_indices, neigh_weights):
                if n >= num_pos:
                    # Compute distractor shard index.
                    shard_idx = (n - num_pos) % chunk_size
                    shard = (n - num_pos) // chunk_size
                    embed = w * self.db.distractor_feat[shard][shard_idx]
                else:
                    embed = w * self.db.positive_feat[n]
                neighbors_embeds.append(embed)
            # Combine neighbor embeddings.
            neighbors_stack = np.stack(neighbors_embeds)
            neighbors_stack = self.adapt(neighbors_stack)
            expanded_embeds.append(neighbors_stack)
        # Concatenate original queries with expanded embeddings and compute mean.
        expanded = np.concatenate(
            [queries[:, None, :], np.array(expanded_embeds)], axis=1
        )
        expanded_queries = expanded.mean(axis=1)
        norms = np.linalg.norm(expanded_queries, axis=1, keepdims=True)
        return expanded_queries / norms

    def search(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute a k-NN search and optionally re-run query expansion.

        Args:
            queries (np.ndarray): Query embeddings.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - sims: Array of distances or similarities.
                - ranks: Array of indices of nearest neighbors.
        """
        queries = self.adapt(queries)
        sims, ranks = self._search(queries)
        if self.query_expansion > 0:
            print(f"\n> apply query expansion with {self.query_expansion} neighbors...")
            expanded_queries = self._query_expansion(
                queries,
                shortlist=ranks[:, : self.query_expansion],
                weights=sims[:, : self.query_expansion],
            )
            sims, ranks = self._search(expanded_queries)
        return sims, ranks
