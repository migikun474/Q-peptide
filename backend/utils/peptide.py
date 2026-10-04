"""Peptide sequence primitives: validation, mass, charge, hydrophobicity scales.

All constants carry their literature source in a comment. Nothing here depends on the
rest of the package, so this module is safe to import anywhere.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

# The canonical 20 proteinogenic amino acids, in a fixed order. The order is part of the
# feature contract: composition vectors are indexed by it, so it must never change.
AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA_ALPHABET)
AA_INDEX = {a: i for i, a in enumerate(AA_ALPHABET)}

# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------
# Average residue masses in Da (monomer mass minus water), IUPAC average masses.
RESIDUE_MASS: dict[str, float] = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519, "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167, "Q": 128.1307, "R": 156.1875,
    "S": 87.0782, "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
}
WATER_MASS = 18.0153

# ---------------------------------------------------------------------------
# Hydrophobicity scales
# ---------------------------------------------------------------------------
# Kyte & Doolittle (1982) J Mol Biol 157:105-132 -- hydropathy index.
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
# Eisenberg et al. (1984) consensus normalised scale. Used for the hydrophobic moment
# because the moment was originally defined on this kind of normalised scale.
EISENBERG: dict[str, float] = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}
# Grantham (1974) polarity.
POLARITY: dict[str, float] = {
    "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2,
    "G": 9.0, "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9,
    "M": 5.7, "N": 11.6, "P": 8.0, "Q": 10.5, "R": 10.5,
    "S": 9.2, "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2,
}

AROMATIC = frozenset("FWY")
ALIPHATIC = frozenset("AVIL")
CHARGED_POS = frozenset("KRH")
CHARGED_NEG = frozenset("DE")

# ---------------------------------------------------------------------------
# pKa values for charge / pI. Side chains plus termini.
# EMBOSS values, as used by standard pI calculators.
# ---------------------------------------------------------------------------
PKA_SIDECHAIN: dict[str, float] = {
    "D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1, "H": 6.5, "K": 10.8, "R": 12.5,
}
PKA_N_TERM = 8.6
PKA_C_TERM = 3.6
POSITIVE_RESIDUES = frozenset("HKR")
NEGATIVE_RESIDUES = frozenset("DECY")


class InvalidSequenceError(ValueError):
    """Raised when a sequence is not a usable peptide over the canonical alphabet."""


def clean_sequence(seq: str | None) -> str | None:
    """Normalise a sequence, or return None if it is not usable.

    Strips whitespace and upper-cases. Returns None (rather than raising) for empty
    input or any residue outside the canonical 20, so this can be mapped over a messy
    column. Use :func:`validate_sequence` when an exception is wanted instead.
    """
    if not isinstance(seq, str):
        return None
    s = re.sub(r"\s+", "", seq).upper()
    if not s:
        return None
    if any(c not in AA_SET for c in s):
        return None
    return s


def validate_sequence(seq: str, min_len: int = 1, max_len: int | None = None) -> str:
    """Return the cleaned sequence or raise :class:`InvalidSequenceError`."""
    if not isinstance(seq, str):
        raise InvalidSequenceError(f"sequence must be a string, got {type(seq).__name__}")
    s = re.sub(r"\s+", "", seq).upper()
    if not s:
        raise InvalidSequenceError("sequence is empty")
    bad = sorted({c for c in s if c not in AA_SET})
    if bad:
        raise InvalidSequenceError(
            f"sequence contains non-canonical residues {bad}; "
            f"allowed alphabet is {AA_ALPHABET}"
        )
    if len(s) < min_len:
        raise InvalidSequenceError(f"sequence length {len(s)} < minimum {min_len}")
    if max_len is not None and len(s) > max_len:
        raise InvalidSequenceError(f"sequence length {len(s)} > maximum {max_len}")
    return s


def molecular_weight(seq: str) -> float:
    """Average molecular weight in Da for a linear, unmodified peptide."""
    return sum(RESIDUE_MASS[a] for a in seq) + WATER_MASS


def net_charge(seq: str, ph: float = 7.4) -> float:
    """Net charge at a given pH via Henderson-Hasselbalch.

    Includes both termini. Default pH 7.4 (physiological), which is the relevant pH for
    membrane-activity arguments about cationic AMPs.
    """
    charge = 0.0
    # N-terminus behaves as a base: protonated fraction carries +1
    charge += 1.0 / (1.0 + 10 ** (ph - PKA_N_TERM))
    # C-terminus behaves as an acid: deprotonated fraction carries -1
    charge -= 1.0 / (1.0 + 10 ** (PKA_C_TERM - ph))
    for aa in seq:
        pka = PKA_SIDECHAIN.get(aa)
        if pka is None:
            continue
        if aa in POSITIVE_RESIDUES:
            charge += 1.0 / (1.0 + 10 ** (ph - pka))
        else:
            charge -= 1.0 / (1.0 + 10 ** (pka - ph))
    return charge


def isoelectric_point(seq: str, tol: float = 1e-4) -> float:
    """pH at which net charge is zero, found by bisection on [0, 14]."""
    lo, hi = 0.0, 14.0
    # net_charge is monotonically decreasing in pH
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        q = net_charge(seq, mid)
        if abs(q) < tol:
            return mid
        if q > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_hydrophobicity(seq: str, scale: dict[str, float] | None = None) -> float:
    """Mean residue hydrophobicity (Kyte-Doolittle by default)."""
    s = scale or KYTE_DOOLITTLE
    return sum(s[a] for a in seq) / len(seq)


def hydrophobic_moment(
    seq: str,
    angle_deg: float = 100.0,
    window: int | None = 11,
    scale: dict[str, float] | None = None,
) -> float:
    """Maximum normalised hydrophobic moment <muH> (Eisenberg et al. 1982).

    The moment treats residue hydrophobicities as vectors spaced ``angle_deg`` apart
    around a helical axis:

        muH = |sum_k H_k * exp(i * k * delta)| / n

    ``angle_deg=100`` corresponds to an alpha-helix (3.6 residues/turn). The value is
    computed over every window of ``window`` residues and the maximum is returned, which
    is the standard convention for locating the most amphipathic segment. If the sequence
    is shorter than the window, the whole sequence is used.

    A high moment means hydrophobic and polar residues segregate onto opposite faces --
    the amphipathic signature of membrane-active AMPs.
    """
    s = scale or EISENBERG
    delta = math.radians(angle_deg)
    n = len(seq)
    w = n if (window is None or window > n) else window

    best = 0.0
    for start in range(0, n - w + 1):
        sub = seq[start:start + w]
        re_sum = sum(s[a] * math.cos(k * delta) for k, a in enumerate(sub))
        im_sum = sum(s[a] * math.sin(k * delta) for k, a in enumerate(sub))
        moment = math.hypot(re_sum, im_sum) / len(sub)
        best = max(best, moment)
    return best


def aromaticity(seq: str) -> float:
    """Fraction of residues that are F, W or Y (Lobry & Gautier 1994 definition)."""
    return sum(1 for a in seq if a in AROMATIC) / len(seq)


def aliphatic_index(seq: str) -> float:
    """Ikai (1980) aliphatic index: relative volume of aliphatic side chains.

        AI = X(A) + 2.9 * X(V) + 3.9 * (X(I) + X(L))

    where X is mole percent. Correlates with thermostability.
    """
    n = len(seq)
    pct = {a: 100.0 * seq.count(a) / n for a in "AVIL"}
    return pct["A"] + 2.9 * pct["V"] + 3.9 * (pct["I"] + pct["L"])


def mean_polarity(seq: str) -> float:
    """Mean Grantham polarity."""
    return sum(POLARITY[a] for a in seq) / len(seq)


def fraction_charged(seq: str) -> tuple[float, float]:
    """(positive fraction, negative fraction) by residue identity, not pH."""
    n = len(seq)
    pos = sum(1 for a in seq if a in CHARGED_POS) / n
    neg = sum(1 for a in seq if a in CHARGED_NEG) / n
    return pos, neg


def apply_mutations(seq: str, mutations: Iterable[tuple[int, str]]) -> str:
    """Apply ``(zero_based_position, new_residue)`` substitutions to a sequence.

    Raises if two mutations target the same position, since that is a conflict the
    compatibility graph is supposed to have excluded -- failing loudly here catches
    constraint-encoding bugs rather than silently producing a wrong mutant.
    """
    chars = list(seq)
    seen: set[int] = set()
    for pos, new_aa in mutations:
        if not 0 <= pos < len(chars):
            raise IndexError(f"position {pos} out of range for length {len(chars)}")
        if new_aa not in AA_SET:
            raise InvalidSequenceError(f"invalid residue {new_aa!r}")
        if pos in seen:
            raise ValueError(
                f"two mutations target position {pos}; same-position conflicts must be "
                "excluded before applying mutations"
            )
        seen.add(pos)
        chars[pos] = new_aa
    return "".join(chars)


def sequence_identity(a: str, b: str) -> float:
    """Fractional identity between two sequences.

    Equal lengths: ungapped positional identity (fast path).
    Unequal lengths: Smith-Waterman-style local match count via a simple DP on matches
    only, normalised by the shorter length. This is a deliberate, documented
    approximation to CD-HIT's identity rather than a full alignment with affine gaps --
    it is used only to group near-duplicates for leakage control, where the exact value
    matters far less than the grouping.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)

    # longest common subsequence length, normalised by the shorter sequence
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[len(b)] / len(a)
