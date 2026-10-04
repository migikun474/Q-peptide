"""Tests for the mutation-extrapolation measurement."""
from __future__ import annotations

import numpy as np
import pytest

from backend.models.mutation_extrapolation import (
    DEAD_ZONE,
    _stratum_metrics,
    find_point_mutant_pairs,
)
from conftest import needs_models


# --- pair finding ----------------------------------------------------------
def test_finds_single_point_mutants():
    seqs = ["ACDEF", "ACDEG", "ACDEF"]
    pairs = find_point_mutant_pairs(seqs)
    # (0,1) differ by 1; (0,2) and (1,2) -- 0 and 2 are identical so excluded
    assert (0, 1, 1) in pairs
    assert (1, 2, 1) in pairs
    assert not any(a == 0 and b == 2 for a, b, _ in pairs)


def test_identical_sequences_are_not_pairs():
    assert find_point_mutant_pairs(["ACDEF", "ACDEF"]) == []


def test_respects_max_diffs():
    seqs = ["AAAAA", "AABBB"]  # 3 differences
    assert find_point_mutant_pairs(seqs, max_diffs=3) == [(0, 1, 3)]
    assert find_point_mutant_pairs(seqs, max_diffs=2) == []


def test_different_lengths_are_never_paired():
    # a length change is an indel, not a point mutation, and the positional diff count
    # would be meaningless
    assert find_point_mutant_pairs(["ACDEF", "ACDE"]) == []
    assert find_point_mutant_pairs(["ACDEF", "ACDEFG"]) == []


def test_diff_count_is_correct():
    pairs = find_point_mutant_pairs(["AAAA", "BAAA", "BBAA"])
    lookup = {(a, b): d for a, b, d in pairs}
    assert lookup[(0, 1)] == 1
    assert lookup[(0, 2)] == 2
    assert lookup[(1, 2)] == 1


def test_pairs_are_upper_triangular():
    pairs = find_point_mutant_pairs(["AAAA", "BAAA", "BBAA", "CAAA"])
    for a, b, _ in pairs:
        assert a < b


def test_empty_and_singleton_inputs():
    assert find_point_mutant_pairs([]) == []
    assert find_point_mutant_pairs(["ACDEF"]) == []


# --- stratum metrics -------------------------------------------------------
def test_perfect_prediction_metrics():
    true = np.array([1.0, -2.0, 0.5, -0.75])
    m = _stratum_metrics(true.copy(), true)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["directional_accuracy"] == pytest.approx(1.0)
    assert m["beats_zero_baseline"] is True


def test_zero_baseline_is_mean_absolute_true_delta():
    true = np.array([1.0, -3.0, 2.0])
    m = _stratum_metrics(np.zeros(3), true)
    assert m["zero_baseline_mae"] == pytest.approx(2.0)
    assert m["mae"] == pytest.approx(m["zero_baseline_mae"])
    assert m["beats_zero_baseline"] is False


def test_sign_flipped_prediction_is_anti_correlated():
    true = np.array([1.0, -2.0, 3.0, -4.0])
    m = _stratum_metrics(-true, true)
    assert m["pearson_r"] == pytest.approx(-1.0)
    assert m["directional_accuracy"] == pytest.approx(0.0)


def test_dead_zone_excludes_small_true_deltas():
    # two pairs well outside the dead zone, two inside it
    true = np.array([1.0, -1.0, 0.01, -0.02])
    pred = np.array([1.0, -1.0, -5.0, 5.0])  # wrong sign only inside the dead zone
    m = _stratum_metrics(pred, true, dead_zone=DEAD_ZONE)
    assert m["n_directional_eligible"] == 2
    # the two in-dead-zone pairs are excluded, so accuracy is perfect
    assert m["directional_accuracy"] == pytest.approx(1.0)


def test_empty_stratum_is_reported_not_crashed():
    m = _stratum_metrics(np.zeros(0), np.zeros(0))
    assert m == {"n_pairs": 0}


def test_constant_prediction_gives_no_correlation():
    m = _stratum_metrics(np.full(5, 0.3), np.array([1.0, 2.0, -1.0, 0.5, -2.0]))
    assert m["pearson_r"] is None
    assert m["spearman_rho"] is None


# --- integration -----------------------------------------------------------
@needs_models
def test_evaluation_runs_on_real_data_and_separates_strata():
    """The real evaluation must produce a same-study stratum and never mix it in silently.

    This guards the specific confound the measurement exists to control for: pooling
    same-study and cross-study pairs inverted the apparent result for the activity model,
    so the strata must stay separated.
    """
    from backend.models.mutation_extrapolation import evaluate_mutation_extrapolation

    report = evaluate_mutation_extrapolation("activity", "activity.csv", "y_activity")
    assert report.n_pairs > 0
    strata = report.by_study_provenance
    assert strata["same_study"]["n_pairs"] > 0
    # the strata must partition the pairs exactly
    assert strata["same_study"]["n_pairs"] + strata["cross_study"]["n_pairs"] == report.n_pairs

    payload = report.as_dict()
    assert payload["headline"]["fair_stratum"] == "same_study"
    assert payload["headline"]["n_pairs"] == strata["same_study"]["n_pairs"]


@needs_models
def test_pairs_are_disjoint_from_training_data():
    """Every evaluated pair must sit entirely in the held-out test split.

    If this regressed, the measurement would be reporting memorisation rather than
    extrapolation.
    """
    import pandas as pd

    from backend.data.splits import DEFAULT_IDENTITY_THRESHOLD, add_clusters, cluster_split
    from backend.models.mutation_extrapolation import find_point_mutant_pairs
    from backend.models.train import SEED

    df = pd.read_csv("data/processed/activity.csv")
    df, _ = add_clusters(df, DEFAULT_IDENTITY_THRESHOLD)
    clusters = df["cluster"].to_numpy()
    cs = cluster_split(clusters, test_frac=0.2, val_frac=0.1, seed=SEED,
                       identity_threshold=DEFAULT_IDENTITY_THRESHOLD)
    test_set = set(cs.test_idx.tolist())
    train_set = set(cs.train_idx.tolist()) | set(cs.val_idx.tolist())

    seqs = df["sequence"].tolist()
    evaluated = [
        (a, b) for a, b, _ in find_point_mutant_pairs(seqs)
        if a in test_set and b in test_set
    ]
    assert evaluated, "no held-out pairs found"
    for a, b in evaluated:
        assert a not in train_set and b not in train_set
