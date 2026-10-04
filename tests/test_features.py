"""Sequence validation, physicochemical descriptors and feature extraction."""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.features.featurize import (
    DEFAULT_CONFIG,
    FeatureConfig,
    DIPEPTIDES,
    feature_names,
    featurize,
    featurize_many,
    n_features,
)
from backend.utils.peptide import (
    AA_ALPHABET,
    InvalidSequenceError,
    aliphatic_index,
    apply_mutations,
    aromaticity,
    clean_sequence,
    hydrophobic_moment,
    isoelectric_point,
    molecular_weight,
    net_charge,
    sequence_identity,
    validate_sequence,
)


# --- validation ------------------------------------------------------------
def test_valid_sequence_roundtrip():
    assert validate_sequence("  gigkflhsa  ") == "GIGKFLHSA"


def test_empty_sequence_rejected():
    with pytest.raises(InvalidSequenceError, match="empty"):
        validate_sequence("")
    with pytest.raises(InvalidSequenceError):
        validate_sequence("   ")


def test_non_canonical_residues_rejected():
    for bad in ("GIGKBX", "ACDEU", "SEQ-WITH-DASH", "ACD1EF"):
        with pytest.raises(InvalidSequenceError):
            validate_sequence(bad)


def test_non_string_rejected():
    with pytest.raises(InvalidSequenceError):
        validate_sequence(None)  # type: ignore[arg-type]


def test_length_bounds_enforced():
    with pytest.raises(InvalidSequenceError, match="minimum"):
        validate_sequence("AC", min_len=5)
    with pytest.raises(InvalidSequenceError, match="maximum"):
        validate_sequence("ACDEFGHIK", max_len=5)


def test_clean_sequence_returns_none_instead_of_raising():
    assert clean_sequence("ACDX") is None
    assert clean_sequence("") is None
    assert clean_sequence(None) is None
    assert clean_sequence("acd") == "ACD"


# --- physicochemistry ------------------------------------------------------
def test_molecular_weight_matches_manual_sum():
    # glycine dipeptide: 2 * 57.0519 + 18.0153
    assert molecular_weight("GG") == pytest.approx(2 * 57.0519 + 18.0153, abs=1e-6)


def test_net_charge_signs():
    # poly-lysine is strongly cationic, poly-glutamate strongly anionic at pH 7.4
    assert net_charge("KKKKK", 7.4) > 3.5
    assert net_charge("EEEEE", 7.4) < -3.5


def test_net_charge_monotonic_in_ph():
    seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    charges = [net_charge(seq, ph) for ph in (2.0, 5.0, 7.4, 10.0, 12.0)]
    assert all(a > b for a, b in zip(charges, charges[1:]))


def test_isoelectric_point_zeroes_net_charge():
    for seq in ("KKKKK", "EEEEE", "GIGKFLHSAKKFGKAFVGEIMNS", "ACDEFGHIK"):
        pi = isoelectric_point(seq)
        assert 0.0 <= pi <= 14.0
        assert abs(net_charge(seq, pi)) < 1e-2


def test_isoelectric_point_ordering():
    assert isoelectric_point("KKKKK") > isoelectric_point("EEEEE")


def test_hydrophobic_moment_detects_amphipathicity():
    # a perfectly alternating pattern at ~100 deg per residue segregates faces;
    # a uniform sequence cannot have a large moment
    uniform = hydrophobic_moment("AAAAAAAAAAA")
    designed = hydrophobic_moment("LKLLKKLLKLL")
    assert designed > uniform


def test_hydrophobic_moment_non_negative_and_bounded():
    for seq in ("A", "GIGKFLHSAKKFGKAFVGEIMNS", "KKKK"):
        mu = hydrophobic_moment(seq)
        assert mu >= 0.0
        assert mu < 5.0


def test_aromaticity_and_aliphatic_index():
    assert aromaticity("FWY") == pytest.approx(1.0)
    assert aromaticity("AAAA") == pytest.approx(0.0)
    # all-alanine: AI = 100
    assert aliphatic_index("AAAA") == pytest.approx(100.0)
    # isoleucine weighs 3.9
    assert aliphatic_index("IIII") == pytest.approx(3.9 * 100.0)


# --- mutations -------------------------------------------------------------
def test_apply_mutations_basic():
    assert apply_mutations("ACDEF", [(0, "G")]) == "GCDEF"
    assert apply_mutations("ACDEF", [(0, "G"), (4, "W")]) == "GCDEW"


def test_apply_mutations_rejects_same_position_conflict():
    with pytest.raises(ValueError, match="same position|two mutations"):
        apply_mutations("ACDEF", [(1, "G"), (1, "W")])


def test_apply_mutations_rejects_bad_inputs():
    with pytest.raises(IndexError):
        apply_mutations("ACDEF", [(99, "G")])
    with pytest.raises(InvalidSequenceError):
        apply_mutations("ACDEF", [(0, "X")])


def test_apply_mutations_is_order_independent():
    a = apply_mutations("ACDEFGHIK", [(0, "W"), (5, "Y")])
    b = apply_mutations("ACDEFGHIK", [(5, "Y"), (0, "W")])
    assert a == b


# --- identity --------------------------------------------------------------
def test_sequence_identity_edge_cases():
    assert sequence_identity("ACDEF", "ACDEF") == 1.0
    assert sequence_identity("", "ACDEF") == 0.0
    assert sequence_identity("AAAAA", "CCCCC") == 0.0
    assert sequence_identity("ACDEF", "ACDEG") == pytest.approx(0.8)


def test_sequence_identity_symmetric():
    a, b = "ACDEFGHIK", "ACDEFGH"
    assert sequence_identity(a, b) == pytest.approx(sequence_identity(b, a))


# --- featurisation ---------------------------------------------------------
def test_feature_names_match_vector_length():
    for cfg in (
        DEFAULT_CONFIG,
        FeatureConfig(use_dipeptide=False),
        FeatureConfig(use_composition=False, use_dipeptide=False),
        FeatureConfig(use_composition=True, use_dipeptide=False, use_physchem=False),
    ):
        vec = featurize("GIGKFLHSAKKFGKAFVGEIMNS", cfg)
        assert len(vec) == n_features(cfg) == len(feature_names(cfg))


def test_feature_names_are_unique():
    names = feature_names(DEFAULT_CONFIG)
    assert len(names) == len(set(names))


def test_composition_sums_to_one():
    cfg = FeatureConfig(use_composition=True, use_dipeptide=False, use_physchem=False)
    names = feature_names(cfg)
    vec = featurize("GIGKFLHSAKKFGKAFVGEIMNS", cfg)
    comp_idx = [i for i, nm in enumerate(names) if nm.startswith("comp_")]
    assert vec[comp_idx].sum() == pytest.approx(1.0)


def test_dipeptide_composition_sums_to_one():
    cfg = FeatureConfig(use_composition=False, use_dipeptide=True, use_physchem=False)
    names = feature_names(cfg)
    vec = featurize("GIGKFLHSAKKFGKAFVGEIMNS", cfg)
    idx = [i for i, nm in enumerate(names) if nm.startswith("dip_")]
    assert vec[idx].sum() == pytest.approx(1.0)
    assert len(DIPEPTIDES) == 400


def test_dipeptide_counts_correct():
    cfg = FeatureConfig(use_composition=False, use_dipeptide=True, use_physchem=False)
    names = feature_names(cfg)
    vec = featurize("AAAB".replace("B", "C"), cfg)  # "AAAC": AA, AA, AC
    lookup = {nm: v for nm, v in zip(names, vec)}
    assert lookup["dip_AA"] == pytest.approx(2 / 3)
    assert lookup["dip_AC"] == pytest.approx(1 / 3)


def test_single_residue_sequence_has_no_dipeptides():
    # the global length block is always emitted, so only the dipeptide slice is summed
    cfg = FeatureConfig(use_composition=False, use_dipeptide=True, use_physchem=False)
    names = feature_names(cfg)
    vec = featurize("A", cfg)
    idx = [i for i, nm in enumerate(names) if nm.startswith("dip_")]
    assert vec[idx].sum() == pytest.approx(0.0)


def test_featurize_is_deterministic():
    a = featurize("GIGKFLHSAKKFGKAFVGEIMNS")
    b = featurize("GIGKFLHSAKKFGKAFVGEIMNS")
    assert np.array_equal(a, b)


def test_featurize_rejects_invalid_sequence():
    with pytest.raises(InvalidSequenceError):
        featurize("GIGKX")
    with pytest.raises(InvalidSequenceError):
        featurize("")


def test_featurize_many_shapes():
    seqs = ["ACDEF", "GIGKFLHSA", "KKKKKKK"]
    X = featurize_many(seqs)
    assert X.shape == (3, n_features(DEFAULT_CONFIG))
    assert featurize_many([]).shape == (0, n_features(DEFAULT_CONFIG))


def test_features_are_finite():
    for seq in ("A", "ACDEF", "GIGKFLHSAKKFGKAFVGEIMNS", AA_ALPHABET):
        vec = featurize(seq)
        assert np.all(np.isfinite(vec)), seq


def test_mutation_changes_features():
    base = featurize("GIGKFLHSAKKFGKAFVGEIMNS")
    mutant = featurize(apply_mutations("GIGKFLHSAKKFGKAFVGEIMNS", [(3, "E")]))
    assert not np.array_equal(base, mutant)
