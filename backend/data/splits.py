"""Sequence-similarity-aware clustering and splitting, for leakage control.

Random train/test splits are invalid for peptide data: near-duplicate sequences land on
both sides and metrics become optimistic. The literature convention for AMPs is to
cluster at ~70% identity and split at the cluster level so a cluster never spans splits
(see research/literature_review.md section 2.4).

CD-HIT is an external binary, so an equivalent greedy incremental clustering is
implemented here to keep the pipeline reproducible without system packages. The
algorithm mirrors CD-HIT's documented strategy:

    1. sort sequences by descending length
    2. walk the list; any unassigned sequence becomes a new cluster representative
    3. assign a sequence to the first representative it matches above the threshold

This module also provides a random-split baseline whose only purpose is to measure the
optimism gap against the cluster-level split. That gap is a validity statistic, not a
performance number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.utils.peptide import sequence_identity

DEFAULT_IDENTITY_THRESHOLD = 0.70


def greedy_identity_clusters(
    sequences: list[str],
    threshold: float = DEFAULT_IDENTITY_THRESHOLD,
) -> np.ndarray:
    """Cluster ids for ``sequences`` by greedy incremental identity clustering.

    Returns an integer array aligned to the input order. Cluster ids are assigned in
    order of representative discovery, so they are deterministic given the input order
    and threshold (ties in length are broken by sequence, making it fully deterministic
    regardless of input order).
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    n = len(sequences)
    assignment = np.full(n, -1, dtype=int)
    # descending length, then lexicographic -- deterministic independent of input order
    order = sorted(range(n), key=lambda i: (-len(sequences[i]), sequences[i]))

    representatives: list[tuple[int, str]] = []  # (cluster_id, sequence)
    next_cluster = 0

    for idx in order:
        seq = sequences[idx]
        placed = False
        for cluster_id, rep in representatives:
            # length guard: identity cannot exceed shorter/longer length ratio
            shorter, longer = sorted((len(seq), len(rep)))
            if shorter / longer < threshold:
                continue
            if sequence_identity(seq, rep) >= threshold:
                assignment[idx] = cluster_id
                placed = True
                break
        if not placed:
            assignment[idx] = next_cluster
            representatives.append((next_cluster, seq))
            next_cluster += 1

    return assignment


@dataclass
class SplitResult:
    """Index arrays for one train/validation/test partition."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    n_clusters: int
    identity_threshold: float | None

    def summary(self, y: np.ndarray | None = None) -> dict:
        out = {
            "strategy": self.strategy,
            "n_clusters": self.n_clusters,
            "identity_threshold": self.identity_threshold,
            "n_train": int(len(self.train_idx)),
            "n_val": int(len(self.val_idx)),
            "n_test": int(len(self.test_idx)),
        }
        if y is not None:
            for name, idx in (
                ("train", self.train_idx),
                ("val", self.val_idx),
                ("test", self.test_idx),
            ):
                if len(idx) == 0:
                    continue
                vals = np.asarray(y)[idx]
                uniq = np.unique(vals)
                if len(uniq) <= 10 and np.all(uniq == uniq.astype(int)):
                    # treat as class labels
                    out[f"{name}_class_balance"] = {
                        int(u): int(np.sum(vals == u)) for u in uniq
                    }
                else:
                    out[f"{name}_y_mean"] = float(np.mean(vals))
                    out[f"{name}_y_std"] = float(np.std(vals))
        return out


def cluster_split(
    clusters: np.ndarray,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 0,
    identity_threshold: float | None = DEFAULT_IDENTITY_THRESHOLD,
) -> SplitResult:
    """Partition at the cluster level so no cluster spans two splits.

    Clusters are shuffled and assigned greedily to test, then validation, then train,
    until each reaches its target share of *sequences* (not clusters). Because clusters
    vary in size the realised fractions are approximate; the achieved sizes are reported
    rather than assumed.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    rng.shuffle(unique_clusters)

    sizes = {c: int(np.sum(clusters == c)) for c in unique_clusters}
    total = len(clusters)
    target_test = test_frac * total
    target_val = val_frac * total

    test_clusters: set[int] = set()
    val_clusters: set[int] = set()
    acc_test = acc_val = 0

    for c in unique_clusters:
        if acc_test < target_test:
            test_clusters.add(int(c))
            acc_test += sizes[c]
        elif acc_val < target_val:
            val_clusters.add(int(c))
            acc_val += sizes[c]

    test_mask = np.isin(clusters, list(test_clusters)) if test_clusters else np.zeros(total, bool)
    val_mask = np.isin(clusters, list(val_clusters)) if val_clusters else np.zeros(total, bool)
    train_mask = ~(test_mask | val_mask)

    return SplitResult(
        train_idx=np.flatnonzero(train_mask),
        val_idx=np.flatnonzero(val_mask),
        test_idx=np.flatnonzero(test_mask),
        strategy="cluster_level_identity",
        n_clusters=int(len(unique_clusters)),
        identity_threshold=identity_threshold,
    )


def random_split(
    n: int,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 0,
) -> SplitResult:
    """Plain random split. Used ONLY to quantify the optimism gap vs cluster splitting."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    return SplitResult(
        train_idx=idx[n_test + n_val:],
        val_idx=idx[n_test: n_test + n_val],
        test_idx=idx[:n_test],
        strategy="random_INVALID_baseline",
        n_clusters=n,
        identity_threshold=None,
    )


def redundancy_report(
    sequences: list[str],
    clusters: np.ndarray,
    threshold: float,
) -> dict:
    """Describe how redundant the dataset is, for the curation/validity record."""
    n = len(sequences)
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    return {
        "n_sequences": n,
        "n_unique_sequences": len(set(sequences)),
        "n_exact_duplicates": n - len(set(sequences)),
        "identity_threshold": threshold,
        "n_clusters": int(len(unique_clusters)),
        "largest_cluster_size": int(counts.max()) if len(counts) else 0,
        "mean_cluster_size": float(counts.mean()) if len(counts) else 0.0,
        "n_singleton_clusters": int(np.sum(counts == 1)),
        "redundancy_ratio": float(n / len(unique_clusters)) if len(unique_clusters) else 0.0,
    }


def add_clusters(
    df: pd.DataFrame,
    threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    seq_col: str = "sequence",
) -> tuple[pd.DataFrame, dict]:
    """Attach a ``cluster`` column and return the redundancy report alongside."""
    seqs = df[seq_col].tolist()
    clusters = greedy_identity_clusters(seqs, threshold)
    out = df.copy()
    out["cluster"] = clusters
    return out, redundancy_report(seqs, clusters, threshold)
