"""Candidate mutation generation and the compatibility graph.

Pipeline (spec section 4):

    parent sequence
        -> sequence validation
        -> physicochemical filtering
        -> biological plausibility filtering
        -> ML pre-screening
        -> candidate mutation set M

The binary-variable <-> mutation mapping is the contract the whole optimizer rests on. It
is deterministic (candidates are sorted by a total order before indices are assigned) and
serialisable, so a QUBO solution can always be decoded back to named substitutions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations

import numpy as np

from backend.utils.peptide import (
    AA_ALPHABET,
    CHARGED_NEG,
    CHARGED_POS,
    KYTE_DOOLITTLE,
    apply_mutations,
    validate_sequence,
)

# Residues whose substitution is structurally risky in peptides, and which are therefore
# not mutated away from by default:
#   C -- may participate in a disulfide bond; breaking one can unfold the peptide
#   G -- backbone flexibility, often required at turns
#   P -- helix breaker / turn former, usually positionally essential
PROTECTED_PARENT_RESIDUES = frozenset("CGP")

# Residues not introduced by default:
#   C -- an unpaired cysteine invites mis-pairing and oxidation
#   P -- introducing a helix breaker into an amphipathic helix is usually destructive
#   M, W -- oxidation-prone; avoided in peptide therapeutic design
FORBIDDEN_NEW_RESIDUES = frozenset("CPMW")


@dataclass(frozen=True, order=True)
class MutationCandidate:
    """One candidate substitution. ``position`` is 0-based.

    The field order makes the dataclass orderable by (position, new_aa), which is what
    gives variable indices a deterministic assignment.
    """
    position: int
    new_aa: str
    original_aa: str

    @property
    def label(self) -> str:
        """Human-readable 1-based label, e.g. ``K12A``."""
        return f"{self.original_aa}{self.position + 1}{self.new_aa}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        d["position_1based"] = self.position + 1
        return d


@dataclass
class MutationSet:
    """An indexed, serialisable candidate mutation set with its conflict structure."""
    parent: str
    candidates: list[MutationCandidate]
    conflict_pairs: list[tuple[int, int]]
    generation_report: dict

    def __post_init__(self) -> None:
        # the index <-> mutation map must be a bijection
        if len({(c.position, c.new_aa) for c in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate mutation candidates in set")

    @property
    def n(self) -> int:
        return len(self.candidates)

    def index_of(self, position: int, new_aa: str) -> int:
        for i, c in enumerate(self.candidates):
            if c.position == position and c.new_aa == new_aa:
                return i
        raise KeyError(f"no candidate for position {position} -> {new_aa}")

    def are_compatible(self, i: int, j: int) -> bool:
        """False iff the two mutations conflict (same position)."""
        if i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        return (a, b) not in set(self.conflict_pairs)

    def apply(self, selection: list[int] | np.ndarray) -> str:
        """Build the mutant sequence for a set of selected variable indices."""
        idx = (
            np.flatnonzero(np.asarray(selection))
            if len(np.shape(selection)) and np.asarray(selection).dtype == bool
            else list(selection)
        )
        muts = [(self.candidates[i].position, self.candidates[i].new_aa) for i in idx]
        return apply_mutations(self.parent, muts)

    def apply_bitstring(self, x: np.ndarray) -> str:
        """Build the mutant for a binary vector over the first ``n`` variables."""
        x = np.asarray(x)
        chosen = [i for i in range(self.n) if x[i]]
        return self.apply(chosen)

    def variable_map(self) -> list[dict]:
        """The serialisable binary-variable <-> mutation mapping."""
        return [
            {"variable_index": i, **c.as_dict()} for i, c in enumerate(self.candidates)
        ]

    def as_dict(self) -> dict:
        return {
            "parent": self.parent,
            "n_variables": self.n,
            "variable_map": self.variable_map(),
            "conflict_pairs": [list(p) for p in self.conflict_pairs],
            "generation_report": self.generation_report,
        }


def _physicochemical_admissible(
    original: str, new: str, min_hydropathy_delta: float, require_property_change: bool
) -> bool:
    """Keep only substitutions that actually move a design-relevant property.

    A substitution that changes neither charge nor hydrophobicity appreciably is a wasted
    binary variable: it cannot move the objective, so it only enlarges the search space.
    The charge/hydrophobicity axes are the ones the magainin literature identifies as
    controlling the activity/hemolysis trade-off.
    """
    if new == original:
        return False
    if not require_property_change:
        return True

    def charge_class(a: str) -> int:
        if a in CHARGED_POS:
            return 1
        if a in CHARGED_NEG:
            return -1
        return 0

    charge_changes = charge_class(new) != charge_class(original)
    hydro_change = abs(KYTE_DOOLITTLE[new] - KYTE_DOOLITTLE[original])
    return charge_changes or hydro_change >= min_hydropathy_delta


def enumerate_admissible(
    parent: str,
    protect_parent: frozenset[str] = PROTECTED_PARENT_RESIDUES,
    forbid_new: frozenset[str] = FORBIDDEN_NEW_RESIDUES,
    min_hydropathy_delta: float = 1.5,
    require_property_change: bool = True,
    positions: list[int] | None = None,
) -> list[MutationCandidate]:
    """All substitutions surviving the structural and physicochemical filters."""
    seq = validate_sequence(parent, min_len=2)
    out: list[MutationCandidate] = []
    allowed_positions = range(len(seq)) if positions is None else positions
    for pos in allowed_positions:
        orig = seq[pos]
        if orig in protect_parent:
            continue
        for new in AA_ALPHABET:
            if new in forbid_new:
                continue
            if not _physicochemical_admissible(
                orig, new, min_hydropathy_delta, require_property_change
            ):
                continue
            out.append(MutationCandidate(pos, new, orig))
    return sorted(out)


def conflict_pairs_for(candidates: list[MutationCandidate]) -> list[tuple[int, int]]:
    """Index pairs that cannot both be selected, i.e. same residue position.

    Returned as sorted ``(i, j)`` with ``i < j``. This is the data the QUBO builder
    consumes; it never re-derives conflicts from strings.
    """
    pairs = [
        (i, j)
        for i, j in combinations(range(len(candidates)), 2)
        if candidates[i].position == candidates[j].position
    ]
    return sorted(pairs)


def generate_mutation_set(
    parent: str,
    scorer,
    target_n: int = 16,
    max_per_position: int = 2,
    min_hydropathy_delta: float = 1.5,
    protect_parent: frozenset[str] = PROTECTED_PARENT_RESIDUES,
    forbid_new: frozenset[str] = FORBIDDEN_NEW_RESIDUES,
    screen: str = "abs_delta",
) -> MutationSet:
    """Build the candidate set of about ``target_n`` mutations.

    ML pre-screening: every admissible single mutant is scored with the real biological
    models and ranked. ``screen`` chooses the ranking:

      ``abs_delta``   by |S(P_i) - S(P)|, keeping the mutations with the largest effect
                      in either direction. This is the default because the QUBO needs
                      informative variables, and a strongly deleterious mutation is as
                      informative about the landscape as a beneficial one -- it also
                      keeps the problem from being trivially "select everything".
      ``improving``   by S(P_i) - S(P) descending, keeping only the most beneficial.
                      Produces an easier instance whose optimum tends toward selecting
                      the budget; recorded as an option, not the default.

    ``max_per_position`` caps how many substitutions are kept at one residue. Keeping a
    few per position is what creates the same-position conflicts that make the constraint
    structure non-trivial; keeping many would waste variables on one site.
    """
    seq = validate_sequence(parent, min_len=2)
    admissible = enumerate_admissible(
        seq,
        protect_parent=protect_parent,
        forbid_new=forbid_new,
        min_hydropathy_delta=min_hydropathy_delta,
    )
    if not admissible:
        raise ValueError(
            "no admissible mutations for this parent under the current filters"
        )

    parent_score = scorer.score_one(seq)
    mutants = [apply_mutations(seq, [(c.position, c.new_aa)]) for c in admissible]
    scores = scorer.score(mutants)
    deltas = np.asarray(scores) - parent_score

    if screen == "improving":
        rank_key = -deltas
    elif screen == "abs_delta":
        rank_key = -np.abs(deltas)
    else:
        raise ValueError(f"unknown screen {screen!r}")

    order = np.argsort(rank_key, kind="stable")

    kept: list[MutationCandidate] = []
    per_position: dict[int, int] = {}
    for idx in order:
        cand = admissible[idx]
        if per_position.get(cand.position, 0) >= max_per_position:
            continue
        kept.append(cand)
        per_position[cand.position] = per_position.get(cand.position, 0) + 1
        if len(kept) >= target_n:
            break

    kept = sorted(kept)  # deterministic variable indexing
    conflicts = conflict_pairs_for(kept)

    kept_deltas = {}
    for c in kept:
        k = admissible.index(c)
        kept_deltas[c.label] = float(deltas[k])

    report = {
        "parent": seq,
        "parent_length": len(seq),
        "parent_score": float(parent_score),
        "n_admissible_after_filters": len(admissible),
        "n_selected": len(kept),
        "target_n": target_n,
        "max_per_position": max_per_position,
        "n_conflict_pairs": len(conflicts),
        "n_positions_used": len(per_position),
        "filters": {
            "protected_parent_residues": "".join(sorted(protect_parent)),
            "forbidden_new_residues": "".join(sorted(forbid_new)),
            "min_hydropathy_delta": min_hydropathy_delta,
            "rationale": (
                "protected parent residues C/G/P are structurally load-bearing "
                "(disulfide, turn flexibility, helix breaking); forbidden new residues "
                "C/P/M/W avoid unpaired cysteine, helix breaking and oxidation-prone "
                "sites; the hydropathy threshold removes substitutions that move no "
                "design-relevant property"
            ),
        },
        "screening": {
            "method": screen,
            "description": (
                "every admissible single mutant was scored with the actual ML models; "
                "no heuristic coefficients were used"
            ),
            "n_scored": len(admissible),
            "single_mutation_deltas": kept_deltas,
        },
        "size_justification": (
            "N is kept near 12-20 so that exact enumeration of 2^N remains feasible, "
            "which is what makes an optimality gap measurable; coefficient cost is "
            "1 + N + N(N-1)/2 real model evaluations"
        ),
    }
    return MutationSet(seq, kept, conflicts, report)


def compatibility_graph(mset: MutationSet) -> dict:
    """Nodes and edges for the mutation-landscape view.

    Edges are labelled ``conflict`` (mutually exclusive) or ``compatible``. Only conflict
    edges constrain the QUBO; compatible edges carry the pairwise interaction once the
    landscape has been computed.
    """
    conflict = set(mset.conflict_pairs)
    nodes = [
        {
            "id": i,
            "label": c.label,
            "position": c.position,
            "position_1based": c.position + 1,
            "original_aa": c.original_aa,
            "new_aa": c.new_aa,
        }
        for i, c in enumerate(mset.candidates)
    ]
    edges = []
    for i, j in combinations(range(mset.n), 2):
        edges.append(
            {
                "source": i,
                "target": j,
                "relation": "conflict" if (i, j) in conflict else "compatible",
            }
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "n_conflict_edges": len(conflict),
        "n_compatible_edges": len(edges) - len(conflict),
    }
