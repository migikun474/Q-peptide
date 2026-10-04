"""Candidate generation, the compatibility graph, and the Delta identities."""
from __future__ import annotations

import numpy as np
import pytest

from backend.optimization.landscape import (
    compute_landscape,
    validate_surrogate_beyond_pairs,
    verify_landscape_identities,
)
from backend.optimization.mutations import (
    FORBIDDEN_NEW_RESIDUES,
    PROTECTED_PARENT_RESIDUES,
    MutationCandidate,
    MutationSet,
    compatibility_graph,
    conflict_pairs_for,
    enumerate_admissible,
    generate_mutation_set,
)
from backend.utils.peptide import apply_mutations


# --- candidate generation --------------------------------------------------
def test_protected_parent_residues_never_mutated(parent):
    for c in enumerate_admissible(parent):
        assert c.original_aa not in PROTECTED_PARENT_RESIDUES


def test_forbidden_new_residues_never_introduced(parent):
    for c in enumerate_admissible(parent):
        assert c.new_aa not in FORBIDDEN_NEW_RESIDUES


def test_no_identity_substitutions(parent):
    for c in enumerate_admissible(parent):
        assert c.new_aa != c.original_aa


def test_original_aa_matches_parent(parent):
    for c in enumerate_admissible(parent):
        assert parent[c.position] == c.original_aa


def test_candidates_are_unique(parent):
    cands = enumerate_admissible(parent)
    assert len({(c.position, c.new_aa) for c in cands}) == len(cands)


def test_candidate_label_is_one_based():
    c = MutationCandidate(3, "E", "K")
    assert c.label == "K4E"
    assert c.as_dict()["position_1based"] == 4


def test_generation_respects_target_n(synthetic_scorer, parent):
    for n in (4, 8, 12, 16):
        mset = generate_mutation_set(parent, synthetic_scorer, target_n=n,
                                     max_per_position=2)
        assert mset.n <= n


def test_max_per_position_respected(synthetic_scorer, parent):
    for cap in (1, 2, 3):
        mset = generate_mutation_set(parent, synthetic_scorer, target_n=16,
                                     max_per_position=cap)
        counts: dict[int, int] = {}
        for c in mset.candidates:
            counts[c.position] = counts.get(c.position, 0) + 1
        assert max(counts.values()) <= cap


def test_max_per_position_one_gives_no_conflicts(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=10,
                                 max_per_position=1)
    assert mset.conflict_pairs == []


def test_variable_indexing_is_deterministic(synthetic_scorer, parent):
    a = generate_mutation_set(parent, synthetic_scorer, target_n=10, max_per_position=2)
    b = generate_mutation_set(parent, synthetic_scorer, target_n=10, max_per_position=2)
    assert [c.label for c in a.candidates] == [c.label for c in b.candidates]


def test_candidates_sorted_by_position_then_residue(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=12,
                                 max_per_position=2)
    keys = [(c.position, c.new_aa) for c in mset.candidates]
    assert keys == sorted(keys)


def test_duplicate_candidates_rejected(parent):
    dup = [MutationCandidate(0, "A", parent[0]), MutationCandidate(0, "A", parent[0])]
    with pytest.raises(ValueError, match="duplicate"):
        MutationSet(parent, dup, [], {})


def test_unknown_screen_rejected(synthetic_scorer, parent):
    with pytest.raises(ValueError, match="screen"):
        generate_mutation_set(parent, synthetic_scorer, target_n=4, screen="bogus")


def test_index_of_roundtrip(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=8)
    for i, c in enumerate(mset.candidates):
        assert mset.index_of(c.position, c.new_aa) == i
    with pytest.raises(KeyError):
        mset.index_of(999, "A")


def test_apply_produces_expected_residues(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=8,
                                 max_per_position=1)
    chosen = [0, 2]
    seq = mset.apply(chosen)
    assert len(seq) == len(parent)
    for i in chosen:
        c = mset.candidates[i]
        assert seq[c.position] == c.new_aa


def test_apply_empty_selection_returns_parent(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=6)
    assert mset.apply([]) == parent


# --- compatibility graph ---------------------------------------------------
def test_compatibility_graph_structure(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=10,
                                 max_per_position=2)
    g = compatibility_graph(mset)
    assert len(g["nodes"]) == mset.n
    assert len(g["edges"]) == mset.n * (mset.n - 1) // 2
    assert g["n_conflict_edges"] == len(mset.conflict_pairs)
    assert g["n_conflict_edges"] + g["n_compatible_edges"] == len(g["edges"])

    for e in g["edges"]:
        same_pos = (
            mset.candidates[e["source"]].position
            == mset.candidates[e["target"]].position
        )
        assert (e["relation"] == "conflict") == same_pos


def test_are_compatible_agrees_with_conflict_pairs(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=10,
                                 max_per_position=2)
    conflicts = set(mset.conflict_pairs)
    for i in range(mset.n):
        assert not mset.are_compatible(i, i)
        for j in range(i + 1, mset.n):
            assert mset.are_compatible(i, j) == ((i, j) not in conflicts)


def test_conflict_pairs_sorted_and_upper_triangular(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=12,
                                 max_per_position=3)
    assert mset.conflict_pairs == sorted(mset.conflict_pairs)
    for i, j in mset.conflict_pairs:
        assert i < j


# --- Delta identities ------------------------------------------------------
def test_delta_i_equals_score_difference(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=8,
                                 max_per_position=2)
    ls = compute_landscape(mset, synthetic_scorer)
    s_parent = synthetic_scorer.score_one(parent)
    assert ls.parent_score == pytest.approx(s_parent)

    for i, c in enumerate(mset.candidates):
        mutant = apply_mutations(parent, [(c.position, c.new_aa)])
        expected = synthetic_scorer.score_one(mutant) - s_parent
        assert ls.delta[i] == pytest.approx(expected, abs=1e-10)


def test_delta_ij_equals_mixed_second_difference(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=8,
                                 max_per_position=2)
    ls = compute_landscape(mset, synthetic_scorer)
    s_p = synthetic_scorer.score_one(parent)

    for i, j in ls.compatible_pairs():
        ci, cj = mset.candidates[i], mset.candidates[j]
        s_i = synthetic_scorer.score_one(apply_mutations(parent, [(ci.position, ci.new_aa)]))
        s_j = synthetic_scorer.score_one(apply_mutations(parent, [(cj.position, cj.new_aa)]))
        s_ij = synthetic_scorer.score_one(
            apply_mutations(parent, [(ci.position, ci.new_aa), (cj.position, cj.new_aa)])
        )
        expected = s_ij - s_i - s_j + s_p
        assert ls.delta_pair[i, j] == pytest.approx(expected, abs=1e-10)


def test_delta_pair_symmetric_and_nan_on_conflicts(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=10,
                                 max_per_position=2)
    ls = compute_landscape(mset, synthetic_scorer)
    for i in range(ls.n):
        assert np.isnan(ls.delta_pair[i, i])
        for j in range(i + 1, ls.n):
            a, b = ls.delta_pair[i, j], ls.delta_pair[j, i]
            if np.isnan(a):
                assert np.isnan(b)
            else:
                assert a == pytest.approx(b)
    for i, j in mset.conflict_pairs:
        assert np.isnan(ls.delta_pair[i, j])


def test_surrogate_exact_for_zero_one_and_two_mutations(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=8,
                                 max_per_position=2)
    ls = compute_landscape(mset, synthetic_scorer)
    report = verify_landscape_identities(ls, synthetic_scorer, tol=1e-9)
    assert report["passed"], report["failures"]
    assert report["n_cases_checked"] == 1 + ls.n + len(ls.compatible_pairs())


def test_empty_selection_gives_zero_surrogate(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=6)
    ls = compute_landscape(mset, synthetic_scorer)
    assert ls.surrogate_delta(np.zeros(ls.n)) == pytest.approx(0.0)


def test_model_evaluation_count_is_as_derived(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=8,
                                 max_per_position=2)
    ls = compute_landscape(mset, synthetic_scorer)
    assert ls.n_model_evaluations == 1 + ls.n + len(ls.compatible_pairs())


def test_interactions_are_present_for_a_nonadditive_score(synthetic_scorer, parent):
    """The synthetic score is non-additive, so some Delta_ij must be non-zero.

    If this ever passes trivially the pairwise machinery would be untested.
    """
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=10,
                                 max_per_position=2)
    ls = compute_landscape(mset, synthetic_scorer)
    pairs = [ls.delta_pair[i, j] for i, j in ls.compatible_pairs()]
    assert pairs, "no compatible pairs generated"
    assert max(abs(v) for v in pairs) > 1e-6


# --- surrogate validation beyond pairs -------------------------------------
def test_surrogate_validation_beyond_pairs_runs(synthetic_scorer, parent):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=10,
                                 max_per_position=2)
    ls = compute_landscape(mset, synthetic_scorer)
    report = validate_surrogate_beyond_pairs(
        ls, synthetic_scorer, min_mutations=3, max_mutations=5, n_samples=60, seed=0
    )
    assert report["status"] == "evaluated"
    assert report["n_samples"] > 0
    assert report["mae"] >= 0.0
    assert report["rmse"] >= report["mae"] - 1e-12
    assert report["mutations_per_sample"]["min"] >= 3
    assert set(report["error_by_mutation_count"])


def test_surrogate_validation_reports_not_evaluated_when_impossible(
    synthetic_scorer, parent
):
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=2,
                                 max_per_position=1)
    ls = compute_landscape(mset, synthetic_scorer)
    report = validate_surrogate_beyond_pairs(ls, synthetic_scorer, min_mutations=3)
    assert report["status"] == "not evaluated"
    assert "reason" in report


def test_surrogate_samples_are_feasible(synthetic_scorer, parent):
    """Sampled multi-mutation selections must never contain a same-position conflict."""
    mset = generate_mutation_set(parent, synthetic_scorer, target_n=12,
                                 max_per_position=3)
    ls = compute_landscape(mset, synthetic_scorer)
    # exercised indirectly: if an infeasible selection were sampled, apply_mutations
    # would raise inside validate_surrogate_beyond_pairs
    report = validate_surrogate_beyond_pairs(
        ls, synthetic_scorer, min_mutations=3, max_mutations=6, n_samples=120, seed=5
    )
    assert report["status"] == "evaluated"


def test_landscape_serialisable(synthetic_scorer, parent):
    import json

    mset = generate_mutation_set(parent, synthetic_scorer, target_n=8)
    ls = compute_landscape(mset, synthetic_scorer)
    payload = json.loads(json.dumps(ls.as_dict()))
    assert payload["n_variables"] == ls.n
    assert len(payload["individual_effects"]) == ls.n
    for e in payload["pairwise_effects"]:
        assert e["interaction"] in {"synergistic", "antagonistic", "additive"}
