"""Sequence -> fixed-length feature vector.

The feature set is a *contract*: `feature_names()` returns names in exactly the order
`featurize()` produces values, and that order is frozen by AA_ALPHABET. Models persist
the feature configuration they were trained with and refuse to score vectors built under
a different one.

Blocks:
  global         length and a log length
  composition    20 amino-acid fractions
  dipeptide      400 dipeptide fractions (optional; see note below)
  physchem       charge, hydrophobicity, moment, polarity, aromaticity, aliphatic, pI

Note on dipeptide composition: it is required by the specification and it carries real
signal, but at 400 dimensions on ~2.3k samples it is also the main vector for the k-mer
leakage documented in research/literature_review.md section 2.4. It is therefore
switchable, and the cluster-level split is what keeps its evaluation honest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from backend.utils.peptide import (
    AA_ALPHABET,
    aliphatic_index,
    aromaticity,
    fraction_charged,
    hydrophobic_moment,
    isoelectric_point,
    mean_hydrophobicity,
    mean_polarity,
    molecular_weight,
    net_charge,
    validate_sequence,
)

DIPEPTIDES: tuple[str, ...] = tuple(a + b for a in AA_ALPHABET for b in AA_ALPHABET)

PHYSCHEM_NAMES: tuple[str, ...] = (
    "net_charge_ph74",
    "net_charge_ph70",
    "mean_hydrophobicity_kd",
    "hydrophobic_moment_helix",
    "mean_polarity",
    "aromaticity",
    "aliphatic_index",
    "isoelectric_point",
    "frac_positive_residues",
    "frac_negative_residues",
    "molecular_weight",
    "charge_per_residue",
    "hydrophobic_moment_x_charge",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Which feature blocks are active. Persisted with every trained model."""
    use_composition: bool = True
    use_dipeptide: bool = True
    use_physchem: bool = True
    moment_window: int = 11
    moment_angle_deg: float = 100.0

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FeatureConfig":
        known = FeatureConfig.__dataclass_fields__.keys()
        return FeatureConfig(**{k: v for k, v in d.items() if k in known})


DEFAULT_CONFIG = FeatureConfig()


def feature_names(config: FeatureConfig = DEFAULT_CONFIG) -> list[str]:
    """Feature names in the exact order `featurize` emits them."""
    names = ["length", "log_length"]
    if config.use_composition:
        names += [f"comp_{a}" for a in AA_ALPHABET]
    if config.use_dipeptide:
        names += [f"dip_{d}" for d in DIPEPTIDES]
    if config.use_physchem:
        names += list(PHYSCHEM_NAMES)
    return names


def n_features(config: FeatureConfig = DEFAULT_CONFIG) -> int:
    return len(feature_names(config))


def _composition(seq: str) -> np.ndarray:
    n = len(seq)
    counts = np.zeros(len(AA_ALPHABET), dtype=np.float64)
    for a in seq:
        counts[AA_ALPHABET.index(a)] += 1.0
    return counts / n


def _dipeptide_composition(seq: str) -> np.ndarray:
    vec = np.zeros(len(DIPEPTIDES), dtype=np.float64)
    if len(seq) < 2:
        return vec
    index = {d: i for i, d in enumerate(DIPEPTIDES)}
    total = len(seq) - 1
    for i in range(total):
        vec[index[seq[i:i + 2]]] += 1.0
    return vec / total


def _physchem(seq: str, config: FeatureConfig) -> np.ndarray:
    q74 = net_charge(seq, 7.4)
    moment = hydrophobic_moment(
        seq, angle_deg=config.moment_angle_deg, window=config.moment_window
    )
    pos, neg = fraction_charged(seq)
    return np.array(
        [
            q74,
            net_charge(seq, 7.0),
            mean_hydrophobicity(seq),
            moment,
            mean_polarity(seq),
            aromaticity(seq),
            aliphatic_index(seq),
            isoelectric_point(seq),
            pos,
            neg,
            molecular_weight(seq),
            q74 / len(seq),
            # explicit interaction: amphipathicity x cationicity is the pairing the
            # magainin charge/hydrophobicity literature identifies as driving the
            # activity/hemolysis trade-off (literature_review.md section 3.1)
            moment * q74,
        ],
        dtype=np.float64,
    )


def featurize(seq: str, config: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Feature vector for one sequence. Raises InvalidSequenceError on bad input."""
    s = validate_sequence(seq, min_len=1)
    parts = [np.array([float(len(s)), float(np.log1p(len(s)))])]
    if config.use_composition:
        parts.append(_composition(s))
    if config.use_dipeptide:
        parts.append(_dipeptide_composition(s))
    if config.use_physchem:
        parts.append(_physchem(s, config))
    return np.concatenate(parts)


def featurize_many(
    sequences: list[str], config: FeatureConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Feature matrix of shape (len(sequences), n_features(config))."""
    if not sequences:
        return np.zeros((0, n_features(config)))
    return np.vstack([featurize(s, config) for s in sequences])
