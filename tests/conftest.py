"""Shared fixtures.

Most tests use a deterministic *synthetic* scorer rather than the trained XGBoost models,
so the mathematical core can be tested without model artefacts and without depending on
training having been run. The synthetic scorer is a closed-form nonlinear function of
physicochemical descriptors: it is cheap, reproducible, and genuinely non-additive over
mutations, which is what the pairwise-interaction machinery needs to be exercised.

Tests that require the real models are marked ``needs_models`` and skip cleanly when the
artefacts are absent.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from backend.utils.peptide import (
    hydrophobic_moment,
    mean_hydrophobicity,
    net_charge,
)

PARENT = "GIGKFLHSAKKFGKAFVGEIMNS"  # magainin 2
MODEL_DIR = Path("models")


class SyntheticScorer:
    """A deterministic stand-in for BiologicalScorer with the same interface.

    The score deliberately contains a product term and a saturating term so that
    mutations interact (Delta_ij != 0); a purely linear surrogate would make the
    pairwise tests vacuous.
    """

    def __init__(self) -> None:
        self.calls = 0

    def score(self, sequences: list[str]) -> np.ndarray:
        self.calls += len(sequences)
        out = np.empty(len(sequences), dtype=float)
        for k, s in enumerate(sequences):
            q = net_charge(s, 7.4)
            h = mean_hydrophobicity(s)
            mu = hydrophobic_moment(s, window=min(11, len(s)))
            # saturating activity in charge, penalty growing with hydrophobicity*charge
            activity = math.tanh(q / 4.0) + 0.3 * mu
            hemolysis = 0.15 * max(h, 0.0) * max(q, 0.0) + 0.1 * mu * mu
            out[k] = activity - hemolysis
        return out

    def score_one(self, sequence: str) -> float:
        return float(self.score([sequence])[0])


@pytest.fixture
def synthetic_scorer() -> SyntheticScorer:
    return SyntheticScorer()


@pytest.fixture
def parent() -> str:
    return PARENT


@pytest.fixture
def small_problem(synthetic_scorer, parent):
    """A small QUBO (few variables) built from the synthetic scorer.

    Kept small enough that every test can enumerate all assignments.
    """
    from backend.optimization.landscape import compute_landscape
    from backend.optimization.mutations import generate_mutation_set
    from backend.optimization.qubo import BudgetMode, build_qubo

    mset = generate_mutation_set(parent, synthetic_scorer, target_n=6, max_per_position=2)
    landscape = compute_landscape(mset, synthetic_scorer)
    problem = build_qubo(landscape, budget_k=2, budget_mode=BudgetMode.AT_MOST_K)
    return problem


@pytest.fixture
def exactly_k_problem(synthetic_scorer, parent):
    from backend.optimization.landscape import compute_landscape
    from backend.optimization.mutations import generate_mutation_set
    from backend.optimization.qubo import BudgetMode, build_qubo

    mset = generate_mutation_set(parent, synthetic_scorer, target_n=6, max_per_position=2)
    landscape = compute_landscape(mset, synthetic_scorer)
    return build_qubo(landscape, budget_k=2, budget_mode=BudgetMode.EXACTLY_K)


def models_available() -> bool:
    return (MODEL_DIR / "activity.meta.json").exists() and (
        MODEL_DIR / "hemolysis.meta.json"
    ).exists()


needs_models = pytest.mark.skipif(
    not models_available(),
    reason="trained models not found; run `python -m backend.models.train`",
)
