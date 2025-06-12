import json
import gzip
import numpy as np

from collections import defaultdict
from sklearn.metrics import average_precision_score

CLUTTER_GROUPS = [(0, 18), (18, 33), (33, 53), (53, float("inf"))]
SCALE_GROUPS = [
    (0, 0.06880213760855043),
    (0.06880213760855043, 0.13526567164179104),
    (0.13526567164179104, 0.2729876723163842),
    (0.2729876723163842, float("inf")),
]


class ILIAS:
    def __init__(self, dataset_dir, check_ids=False):
        """
        Initializes the ILIAS dataset.
        Args:
            dataset_dir (str): Path to the directory containing the dataset files.
            check_ids (bool): Whether to load distractor image IDs from the file.
        """
        self.dataset_dir = dataset_dir
        self.check_ids = check_ids

        image_query_info = np.loadtxt(
            f"{dataset_dir}/image_ids/image_query_ids.txt", dtype=str, delimiter=","
        )
        self.image_query_ids = sorted(image_query_info[:, 0].tolist())

        self.text_query_ids = sorted(
            np.loadtxt(
                f"{dataset_dir}/image_ids/text_query_ids.txt", dtype=str
            ).tolist()
        )

        positive_info = np.loadtxt(
            f"{dataset_dir}/image_ids/positive_ids.txt", dtype=str, delimiter=","
        )
        self.positive_ids = sorted(positive_info[:, 0].tolist())

        self.positive_info = {}
        for q, s, c in positive_info:
            self.positive_info[q] = (float(s), int(c))

        self.distractor_ids = []
        if self.check_ids:
            with gzip.open(f"{dataset_dir}/image_ids/distractor_ids.txt.gz", "rt") as f:
                for line in f:
                    self.distractor_ids.append(line.strip())

        self.gt = {img.split("/")[0]: set() for img in self.text_query_ids}
        for img in self.positive_ids:
            self.gt[img.split("/")[0]].add(img)

        with open(f"misc/taxonomy.json", "r") as f:
            taxonomy = json.load(f)
        self.taxonomy = {
            q: f"{c}-{sc}"
            for c, sc_dict in taxonomy.items()
            for sc, ssc_dict in sc_dict.items()
            for q_list in ssc_dict.values()
            for q in q_list
        }

    def get_image_queries(self):
        return self.image_query_ids

    def get_text_queries(self):
        return self.text_query_ids

    def get_queries(self):
        return self.get_image_queries() + self.get_text_queries()

    def get_positives(self):
        return self.positive_ids

    def get_distractors(self):
        return self.distractor_ids

    def _compute_AP(self, instance, ids, scores, ranks, k, excluded=None):
        """Computes the Average Precision (AP) for a given query.
        Args:
            q (str): Query ID.
            ids (np.ndarray): Array of database IDs.
            scores (np.ndarray): Array of similarity scores corresponding to the IDs.
            ranks (np.ndarray): Array of ranks corresponding to the IDs.
            k (int): Number of top results to consider.
            excluded (set, optional): Set of positive IDs to exclude from the evaluation.
        Returns:
            tuple: Average Precision (AP) and oracle value.
        """
        if excluded is not None:
            e = [i for i in range(len(ranks)) if ids[ranks[i]] in excluded]
            scores = np.delete(scores.copy(), e, 0)
            ranks = np.delete(ranks.copy(), e, 0)
        else:
            excluded = set()

        # sort by descending score
        order = np.argsort(-scores)
        top_ids = ids[ranks[order][:k]]

        gt = self.gt[instance] - set(excluded)
        y_true = np.isin(top_ids, list(gt))
        y_score = scores[order][:k]

        if not y_true.any():
            return 0.0, 0.0

        ap = average_precision_score(y_true, y_score)
        ap *= y_true.sum() / min(k, len(gt))
        oracle = y_true.mean() * (k / len(gt))
        return ap, oracle

    def _per_scale(self, scale, instance, db_ids, sims, ranks, k):
        """Computes the Average Precision (AP) for different scale groups.
        Args:
            scale (dict): Dictionary to store AP values for different scale groups.
            instance (str): Instance ID for which to compute AP.
            db_ids (np.ndarray): Array of database IDs.
            sims (np.ndarray): Array of similarity scores corresponding to the IDs.
            ranks (np.ndarray): Array of ranks corresponding to the IDs.
            k (int): Number of top results to consider.
        Returns:
            dict: Updated scale dictionary with AP values for each scale group.
        """
        for t1, t2 in SCALE_GROUPS:
            excluded = set(
                [
                    i
                    for i in self.gt[instance]
                    if self.positive_info[i][0] < t1 or self.positive_info[i][0] >= t2
                ]
            )
            if self.gt[instance] - excluded:
                ap, _ = self._compute_AP(instance, db_ids, sims, ranks, k, excluded)
                scale[f"{t1}-{t2}"].append(ap * 100)
        return scale

    def _per_clutter(self, clutter, instance, db_ids, sims, ranks, k):
        """Computes the Average Precision (AP) for different clutter groups.
        Args:
            clutter (dict): Dictionary to store AP values for different clutter groups.
            instance (str): Instance ID for which to compute AP.
            db_ids (np.ndarray): Array of database IDs.
            sims (np.ndarray): Array of similarity scores corresponding to the IDs.
            ranks (np.ndarray): Array of ranks corresponding to the IDs.
            k (int): Number of top results to consider.
        Returns:
            dict: Updated clutter dictionary with AP values for each clutter group.
        """
        for t1, t2 in CLUTTER_GROUPS:
            excluded = set(
                [
                    i
                    for i in self.gt[instance]
                    if self.positive_info[i][1] < t1 or self.positive_info[i][1] >= t2
                ]
            )
            if self.gt[instance] - excluded:
                ap, _ = self._compute_AP(instance, db_ids, sims, ranks, k, excluded)
                clutter[f"{t1}-{t2}"].append(ap * 100)
        return clutter

    def _evaluate_pkl(self, q_ids, db_ids, sims, ranks, k, report):
        # Check for missing or extra queries
        if self.check_ids:
            extra_q = set(q_ids) - (
                set(self.image_query_ids) | set(self.text_query_ids)
            )
            if extra_q:
                raise ValueError(f"Unknown query IDs: {extra_q}")

            # Check database IDs if loading distractors
            extra_db = set(db_ids) - (set(self.positive_ids) | set(self.distractor_ids))
            if extra_db:
                raise ValueError(f"Unknown database IDs: {extra_db}")

        if not isinstance(db_ids, np.ndarray):
            db_ids = np.asarray(db_ids)

        queries, aps = [], []
        if report:
            oracles = []
            scale, clutter = defaultdict(list), defaultdict(list)
            categories = defaultdict(lambda: defaultdict(list))
            subcategories = defaultdict(lambda: defaultdict(list))
        for q, s, r in zip(q_ids, sims, ranks):
            ins = q.split("/")[0]
            ap, oracle = self._compute_AP(ins, db_ids, s, r, k)
            queries.append(q)
            aps.append(ap * 100)

            if report:
                oracles.append(oracle * 100)
                categories[self.taxonomy[ins].split("-")[0]][ins].append(ap * 100)
                subcategories[self.taxonomy[ins]][ins].append(ap * 100)
                scale = self._per_scale(scale, ins, db_ids, s, r, k)
                clutter = self._per_clutter(clutter, ins, db_ids, s, r, k)

        if not report:
            return {
                "queries": queries,
                "map": np.mean(aps),
                "aps": np.array(aps),
            }
        return {
            "queries": queries,
            "map": np.mean(aps),
            "aps": np.array(aps),
            "oracle": np.mean(oracles),
            "scale": {k: np.mean(v) for k, v in scale.items()},
            "clutter": {k: np.mean(v) for k, v in clutter.items()},
            "categories": {
                k: np.mean([np.mean(i) for i in v.values()])
                for k, v in categories.items()
            },
            "subcategories": {
                k: np.mean([np.mean(i) for i in v.values()])
                for k, v in subcategories.items()
            },
        }

    def _evaluate_json(self, similarities, k, report):
        # Check for missing or extra queries
        if self.check_ids:
            q_ids = list(similarities.keys())
            db_ids = sum([list(i.keys()) for i in similarities.values()], [])

            extra_q = set(q_ids) - (
                set(self.image_query_ids) | set(self.text_query_ids)
            )
            if extra_q:
                raise ValueError(f"Unknown query IDs: {extra_q}")

            # Check database IDs if loading distractors
            extra_db = set(db_ids) - (set(self.positive_ids) | set(self.distractor_ids))
            if extra_db:
                raise ValueError(f"Unknown database IDs: {extra_db}")

        queries, aps = [], []
        if report:
            oracles = []
            scale, clutter = defaultdict(list), defaultdict(list)
            categories = defaultdict(lambda: defaultdict(list))
            subcategories = defaultdict(lambda: defaultdict(list))
        for q in self.get_queries():
            if q not in similarities:
                continue
            ins = q.split("/")[0]
            db_ids = np.array(list(similarities[q].keys()))
            s = np.array(list(similarities[q].values()))
            r = np.arange(len(s))
            ap, oracle = self._compute_AP(ins, db_ids, s, r, k)

            queries.append(q)
            aps.append(ap * 100)

            if report:
                oracles.append(oracle * 100)
                categories[self.taxonomy[ins].split("-")[0]][ins].append(ap * 100)
                subcategories[self.taxonomy[ins]][ins].append(ap * 100)
                scale = self._per_scale(scale, ins, db_ids, s, r, k)
                clutter = self._per_clutter(clutter, ins, db_ids, s, r, k)

        if not report:
            return {
                "queries": queries,
                "map": np.mean(aps),
                "aps": np.array(aps),
            }
        return {
            "queries": queries,
            "map": np.mean(aps),
            "aps": np.array(aps),
            "oracle": np.mean(oracles),
            "scale": {k: np.mean(v) for k, v in scale.items()},
            "clutter": {k: np.mean(v) for k, v in clutter.items()},
            "categories": {
                k: np.mean([np.mean(i) for i in v.values()])
                for k, v in categories.items()
            },
            "subcategories": {
                k: np.mean([np.mean(i) for i in v.values()])
                for k, v in subcategories.items()
            },
        }

    def evaluate(
        self,
        all_similarities=None,
        query_ids=None,
        db_ids=None,
        similarities=None,
        ranks=None,
        k=1000,
        report=False,
    ):
        """
        Evaluates the dataset using the provided similarities and ranks.
        Args:
            all_similarities (dict, optional): Dictionary of all similarities for each query.
            query_ids (list, optional): List of query IDs.
            db_ids (np.ndarray, optional): Array of database IDs.
            similarities (np.ndarray, optional): Array of similarity scores.
            ranks (np.ndarray, optional): Array of ranks corresponding to the IDs.
            k (int): Number of top results to consider for mAP calculation.
            report (bool): Whether to generate a detailed report.
        Returns:
            dict: Evaluation report containing mAP and other metrics.
        """
        if all_similarities is None:
            return self._evaluate_pkl(query_ids, db_ids, similarities, ranks, k, report)
        else:
            return self._evaluate_json(all_similarities, k, report)
