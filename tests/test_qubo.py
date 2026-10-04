"""QUBO construction, constraint encoding, penalties and the matrix convention."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from backend.optimization.landscape import compute_landscape
from backend.optimization.mutations import conflict_pairs_for, generate_mutation_set
from backend.optimization.qubo import (
    BudgetMode,
    QuboPolynomial,
    build_qubo,
    evaluate_matrix,
    objective_bound,
    objective_range,
    representable_slack_values,
    slack_weights,
    verify_matrix_matches_polynomial,
    verify_penalty_sufficiency,
    verify_slack_range,
)


# --- the matrix convention -------------------------------------------------
def test_polynomial_matrix_equivalence_hand_built():
    """E(x) = 3 x0 - 2 x1 + 5 x0 x1 + 7, checked against the matrix form."""
    poly = QuboPolynomial(n_vars=2)
    poly.add_linear(0, 3.0)
    poly.add_linear(1, -2.0)
    poly.add_quadratic(0, 1, 5.0)
    poly.add_constant(7.0)

    Q, c = poly.to_matrix()
    # the off-diagonal must be HALF the polynomial coefficient
    assert Q[0, 1] == pytest.approx(2.5)
    assert Q[1, 0] == pytest.approx(2.5)
    assert Q[0, 0] == pytest.approx(3.0)
    assert Q[1, 1] == pytest.approx(-2.0)

    for x in itertools.product([0, 1], repeat=2):
        xv = np.array(x)
        expected = 3.0 * x[0] - 2.0 * x[1] + 5.0 * x[0] * x[1] + 7.0
        assert poly.evaluate(xv) == pytest.approx(expected)
        assert evaluate_matrix(Q, c, xv) == pytest.approx(expected)


def test_matrix_is_symmetric(small_problem):
    assert np.allclose(small_problem.Q, small_problem.Q.T)


def test_matrix_matches_polynomial_on_random_bitstrings(small_problem):
    report = verify_matrix_matches_polynomial(small_problem, n_samples=3000, seed=1)
    assert report["passed"], report["failures"]
    assert report["max_abs_deviation"] < 1e-9


def test_matrix_matches_polynomial_exhaustively(small_problem):
    n = small_problem.n_vars
    assert n <= 16, "fixture should stay small enough to enumerate"
    for code in range(1 << n):
        x = np.array([(code >> b) & 1 for b in range(n)])
        assert small_problem.energy(x) == pytest.approx(
            small_problem.energy_polynomial(x), abs=1e-9
        )


def test_squared_variable_folds_into_linear():
    poly = QuboPolynomial(n_vars=1)
    poly.add_quadratic(0, 0, 4.0)  # x0^2 == x0
    assert poly.linear[0] == pytest.approx(4.0)
    assert poly.quadratic == {}


def test_polynomial_rejects_out_of_range_indices():
    poly = QuboPolynomial(n_vars=2)
    with pytest.raises(IndexError):
        poly.add_linear(5, 1.0)
    with pytest.raises(IndexError):
        poly.add_quadratic(0, 9, 1.0)


def test_polynomial_evaluate_rejects_wrong_length():
    poly = QuboPolynomial(n_vars=3)
    with pytest.raises(ValueError):
        poly.evaluate(np.array([1, 0]))


def test_quadratic_key_is_order_independent():
    poly = QuboPolynomial(n_vars=3)
    poly.add_quadratic(2, 0, 1.5)
    poly.add_quadratic(0, 2, 2.5)
    assert poly.quadratic == {(0, 2): pytest.approx(4.0)}


# --- slack encoding --------------------------------------------------------
@pytest.mark.parametrize("k", list(range(0, 18)))
def test_slack_weights_represent_exactly_zero_to_k(k):
    report = verify_slack_range(k)
    assert report["passed"], report
    assert representable_slack_values(slack_weights(k)) == set(range(k + 1))


@pytest.mark.parametrize("k", [1, 2, 3, 4, 7, 8, 15, 16])
def test_slack_weights_sum_to_k(k):
    assert sum(slack_weights(k)) == k


def test_slack_bit_count_is_logarithmic():
    assert len(slack_weights(0)) == 0
    assert len(slack_weights(1)) == 1
    assert len(slack_weights(3)) == 2
    assert len(slack_weights(7)) == 3
    assert len(slack_weights(15)) == 4


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        slack_weights(-1)


# --- budget modes are distinct --------------------------------------------
def test_at_most_k_allows_fewer_mutations(small_problem):
    assert small_problem.budget_mode is BudgetMode.AT_MOST_K
    n_mut = small_problem.n_mutation_vars
    x = np.zeros(small_problem.n_vars, dtype=int)
    # zero mutations is within an "at most K" budget
    assert small_problem.is_mutation_feasible(x)


def test_exactly_k_forbids_fewer_mutations(exactly_k_problem):
    assert exactly_k_problem.budget_mode is BudgetMode.EXACTLY_K
    x = np.zeros(exactly_k_problem.n_vars, dtype=int)
    assert not exactly_k_problem.is_mutation_feasible(x)
    # and has no slack variables
    assert exactly_k_problem.n_slack_vars == 0


def test_exactly_k_accepts_exactly_k(exactly_k_problem):
    k = exactly_k_problem.budget_k
    mset = exactly_k_problem.landscape.mutation_set
    # pick k mutually compatible mutations
    conflicts = set(mset.conflict_pairs)
    chosen: list[int] = []
    for i in range(mset.n):
        if all((min(i, j), max(i, j)) not in conflicts for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    assert len(chosen) == k
    x = np.zeros(exactly_k_problem.n_vars, dtype=int)
    x[chosen] = 1
    assert exactly_k_problem.is_mutation_feasible(x)


def test_at_most_k_and_exactly_k_differ_in_feasible_count(
    small_problem, exactly_k_problem
):
    from backend.optimization.solvers import solve_exact

    a = solve_exact(small_problem)
    b = solve_exact(exactly_k_problem)
    # equality is strictly more restrictive on the mutation part
    assert a.n_feasible != b.n_feasible


def test_budget_mode_descriptions_are_explicit():
    assert "<=" in BudgetMode.AT_MOST_K.description
    assert "==" in BudgetMode.EXACTLY_K.description


# --- conflicts -------------------------------------------------------------
def test_same_position_pair_is_penalised(small_problem):
    mset = small_problem.landscape.mutation_set
    if not mset.conflict_pairs:
        pytest.skip("fixture produced no same-position conflicts")
    i, j = mset.conflict_pairs[0]
    assert mset.candidates[i].position == mset.candidates[j].position

    base = np.zeros(small_problem.n_vars, dtype=int)
    only_i = base.copy()
    only_i[i] = 1
    both = only_i.copy()
    both[j] = 1
    # selecting both must raise the energy by at least the conflict penalty
    assert small_problem.energy(both) > small_problem.energy(only_i)
    assert not small_problem.is_mutation_feasible(both)


def test_conflict_detection_matches_positions():
    from backend.optimization.mutations import MutationCandidate

    cands = [
        MutationCandidate(0, "A", "K"),
        MutationCandidate(0, "E", "K"),
        MutationCandidate(5, "W", "G"),
    ]
    assert conflict_pairs_for(cands) == [(0, 1)]


def test_feasibility_distinguishes_mutation_and_full_constraints(small_problem):
    """A valid selection with wrong slack bits is mutation-feasible but not fully so."""
    if small_problem.n_slack_vars == 0:
        pytest.skip("no slack variables in this problem")
    x = np.zeros(small_problem.n_vars, dtype=int)
    # zero mutations: needs slack == K to satisfy the budget equation
    assert small_problem.is_mutation_feasible(x)
    assert not small_problem.is_feasible(x)  # slack is 0, not K


def test_vectorised_feasibility_matches_scalar(small_problem):
    from backend.optimization.solvers import (
        _all_bitstrings,
        feasibility_mask_vectorised,
    )

    bits = _all_bitstrings(small_problem.n_vars)
    fast = feasibility_mask_vectorised(small_problem, bits)
    slow = np.array(
        [small_problem.is_feasible(bits[k]) for k in range(bits.shape[0])], dtype=bool
    )
    assert np.array_equal(fast, slow)


# --- penalties -------------------------------------------------------------
def test_penalty_sufficiency_verified_by_enumeration(small_problem):
    report = verify_penalty_sufficiency(small_problem)
    assert report["status"] == "evaluated"
    assert report["passed"], report
    assert report["margin"] > 0


def test_optimum_is_feasible_when_penalties_sufficient(small_problem):
    from backend.optimization.solvers import solve_exact

    res = solve_exact(small_problem)
    assert res.optimal_energy == pytest.approx(res.best_feasible_energy)
    for x in res.optimal_solutions:
        assert small_problem.is_feasible(x)


def test_objective_span_is_tighter_than_triangle_bound(small_problem):
    landscape = small_problem.landscape
    span = objective_range(landscape)["span"]
    bound = objective_bound(landscape)
    assert span <= bound + 1e-9


def test_objective_span_matches_brute_force(small_problem):
    landscape = small_problem.landscape
    n = landscape.n
    vals = []
    for code in range(1 << n):
        x = np.array([(code >> b) & 1 for b in range(n)])
        vals.append(landscape.surrogate_delta(x))
    info = objective_range(landscape)
    assert info["min"] == pytest.approx(min(vals), abs=1e-9)
    assert info["max"] == pytest.approx(max(vals), abs=1e-9)


def _all_beneficial_landscape(parent: str, n: int = 6):
    """A landscape where every mutation strictly improves the score, additively.

    The unconstrained optimum is therefore "select all n", which violates any budget
    K < n. This is the adversarial case a penalty actually has to work for -- in a
    generic instance the unconstrained optimum may already happen to be feasible, in
    which case no penalty is needed and a tiny one would pass vacuously.
    """
    from backend.optimization.landscape import MutationLandscape
    from backend.optimization.mutations import (
        MutationSet,
        conflict_pairs_for,
        enumerate_admissible,
    )

    # one candidate per distinct position, so no same-position conflict interferes;
    # deduplicate BEFORE slicing, otherwise the first n candidates all share a position
    seen: set[int] = set()
    unique = []
    for c in enumerate_admissible(parent):
        if c.position in seen:
            continue
        seen.add(c.position)
        unique.append(c)
        if len(unique) == n:
            break
    assert len(unique) == n, f"only found {len(unique)} distinct positions"
    unique = sorted(unique)
    mset = MutationSet(parent, unique, conflict_pairs_for(unique), {"synthetic": True})

    delta = np.full(mset.n, 1.0)
    delta_pair = np.zeros((mset.n, mset.n))
    np.fill_diagonal(delta_pair, np.nan)
    return MutationLandscape(mset, 0.0, delta, delta_pair, 0, {"synthetic": True})


def test_insufficient_penalty_is_detected(parent):
    """A deliberately tiny penalty must FAIL the sufficiency check.

    This guards the checker itself: a verifier that always passes is worthless.
    """
    landscape = _all_beneficial_landscape(parent, n=6)
    bad = build_qubo(
        landscape,
        budget_k=2,
        budget_mode=BudgetMode.AT_MOST_K,
        penalty_budget=1e-6,
        penalty_conflict=1e-6,
    )
    report = verify_penalty_sufficiency(bad)
    assert report["status"] == "evaluated"
    assert not report["passed"], (
        "a 1e-6 penalty cannot hold a budget against a uniformly beneficial objective"
    )
    # and the unconstrained optimum really does over-select
    assert report["lowest_energy_infeasible_assignment"] is not None


def test_derived_penalty_holds_in_the_adversarial_case(parent):
    """With the derived penalty, the same adversarial instance stays feasible."""
    from backend.optimization.solvers import solve_exact

    landscape = _all_beneficial_landscape(parent, n=6)
    good = build_qubo(landscape, budget_k=2, budget_mode=BudgetMode.AT_MOST_K)
    report = verify_penalty_sufficiency(good)
    assert report["passed"], report

    res = solve_exact(good)
    for x in res.optimal_solutions:
        assert good.is_feasible(x)
        # exactly K selected, since every mutation is beneficial
        assert int(np.sum(good.mutation_part(x))) == good.budget_k


def test_unknown_penalty_mode_rejected(small_problem):
    with pytest.raises(ValueError, match="penalty_mode"):
        build_qubo(small_problem.landscape, budget_k=2, penalty_mode="nonsense")


def test_budget_exceeding_candidates_rejected(small_problem):
    n = small_problem.n_mutation_vars
    with pytest.raises(ValueError, match="exceeds"):
        build_qubo(small_problem.landscape, budget_k=n + 1)


# --- variable mapping ------------------------------------------------------
def test_variable_map_is_a_bijection(small_problem):
    vmap = small_problem.landscape.mutation_set.variable_map()
    assert len(vmap) == small_problem.n_mutation_vars
    indices = [e["variable_index"] for e in vmap]
    assert indices == list(range(small_problem.n_mutation_vars))
    keys = {(e["position"], e["new_aa"]) for e in vmap}
    assert len(keys) == len(vmap)


def test_variable_map_serialisable(small_problem):
    import json

    payload = small_problem.as_dict(include_matrix=True)
    text = json.dumps(payload)
    assert "variable_map" in json.loads(text)


def test_decode_roundtrip(small_problem):
    mset = small_problem.landscape.mutation_set
    conflicts = set(mset.conflict_pairs)
    chosen: list[int] = []
    for i in range(mset.n):
        if all((min(i, j), max(i, j)) not in conflicts for j in chosen):
            chosen.append(i)
        if len(chosen) == small_problem.budget_k:
            break

    x = np.zeros(small_problem.n_vars, dtype=int)
    x[chosen] = 1
    decoded = small_problem.decode(x)
    assert decoded["feasible"]
    assert decoded["n_mutations"] == len(chosen)
    assert decoded["sequence"] is not None
    assert decoded["sequence"] != mset.parent
    assert set(decoded["mutations"]) == {mset.candidates[i].label for i in chosen}

    # every reported mutation is actually present in the sequence
    for i in chosen:
        c = mset.candidates[i]
        assert decoded["sequence"][c.position] == c.new_aa


def test_decode_reports_infeasible_without_sequence(small_problem):
    mset = small_problem.landscape.mutation_set
    if not mset.conflict_pairs:
        pytest.skip("no conflicts available")
    i, j = mset.conflict_pairs[0]
    x = np.zeros(small_problem.n_vars, dtype=int)
    x[i] = x[j] = 1
    decoded = small_problem.decode(x)
    assert not decoded["feasible"]
    assert decoded["sequence"] is None
    assert "conflict" in decoded["infeasible_reason"]


def test_constraint_residuals_report_slack(small_problem):
    x = np.zeros(small_problem.n_vars, dtype=int)
    res = small_problem.constraint_residuals(x)
    assert res["n_mutations_selected"] == 0
    assert res["budget_k"] == small_problem.budget_k
    if small_problem.n_slack_vars:
        assert res["budget_equation_residual"] == -small_problem.budget_k
