"""QUBO construction from the mutation landscape, with explicit constraint encoding.

Derivation (spec sections 16-21). The biological objective is a maximisation:

    max_x  DeltaS_hat(x) = sum_i Delta_i x_i + sum_{i<j} Delta_ij x_i x_j

Minimisation form, then penalties:

    E(x) = -DeltaS_hat(x) + E_constraints(x)

MATRIX CONVENTION, used consistently and tested against the polynomial form:

    E(x) = x^T Q x + c        with Q SYMMETRIC

Since x_i^2 = x_i for binary x, this expands to

    E(x) = sum_i Q_ii x_i + 2 sum_{i<j} Q_ij x_i x_j + c

so a desired polynomial coefficient c_ij on x_i x_j requires Q_ij = Q_ji = c_ij / 2, and
a desired linear coefficient c_i requires Q_ii = c_i. The factor of two is the classic
QUBO bug; `verify_matrix_matches_polynomial` exists to catch it.

VARIABLE LAYOUT
    indices [0, N)              mutation selection variables
    indices [N, N + S)          slack bits, only for the AT_MOST_K budget mode

Both budget modes are implemented and never conflated:
    EXACTLY_K   penalise (sum_i x_i - K)^2                 -- enforces equality
    AT_MOST_K   introduce slack s in [0, K] with
                sum_i x_i + s = K, penalise (sum_i x_i + s - K)^2
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations

import numpy as np

from backend.optimization.landscape import MutationLandscape


class BudgetMode(str, Enum):
    """Which mutation-count constraint is meant. Never inferred; always explicit."""
    AT_MOST_K = "at_most_k"
    EXACTLY_K = "exactly_k"

    @property
    def description(self) -> str:
        return {
            BudgetMode.AT_MOST_K: "sum_i x_i <= K  (slack-variable encoding)",
            BudgetMode.EXACTLY_K: "sum_i x_i == K  (equality penalty, no slack)",
        }[self]


# ---------------------------------------------------------------------------
# Polynomial accumulator
# ---------------------------------------------------------------------------
@dataclass
class QuboPolynomial:
    """A QUBO as an explicit polynomial: linear dict, upper-triangular quadratic dict.

    This is the source of truth. The matrix form is derived from it, and a test asserts
    the two agree on random bitstrings.
    """
    n_vars: int
    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)
    constant: float = 0.0

    def add_linear(self, i: int, coeff: float) -> None:
        if not 0 <= i < self.n_vars:
            raise IndexError(f"variable {i} out of range for n_vars={self.n_vars}")
        if coeff:
            self.linear[i] = self.linear.get(i, 0.0) + coeff

    def add_quadratic(self, i: int, j: int, coeff: float) -> None:
        if i == j:
            # x_i^2 == x_i for binary variables
            self.add_linear(i, coeff)
            return
        a, b = (i, j) if i < j else (j, i)
        if not (0 <= a and b < self.n_vars):
            raise IndexError(f"pair ({i},{j}) out of range for n_vars={self.n_vars}")
        if coeff:
            self.quadratic[(a, b)] = self.quadratic.get((a, b), 0.0) + coeff

    def add_constant(self, c: float) -> None:
        self.constant += c

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate directly from the polynomial, independent of any matrix."""
        x = np.asarray(x)
        if len(x) != self.n_vars:
            raise ValueError(f"expected {self.n_vars} variables, got {len(x)}")
        total = self.constant
        for i, c in self.linear.items():
            if x[i]:
                total += c
        for (i, j), c in self.quadratic.items():
            if x[i] and x[j]:
                total += c
        return float(total)

    def to_matrix(self) -> tuple[np.ndarray, float]:
        """Symmetric ``Q`` with ``Q_ij = c_ij / 2`` off-diagonal, plus the constant."""
        Q = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        for i, c in self.linear.items():
            Q[i, i] = c
        for (i, j), c in self.quadratic.items():
            Q[i, j] = c / 2.0
            Q[j, i] = c / 2.0
        return Q, self.constant

    def max_abs_coefficient(self) -> float:
        vals = list(map(abs, self.linear.values())) + list(
            map(abs, self.quadratic.values())
        )
        return max(vals) if vals else 0.0


def evaluate_matrix(Q: np.ndarray, constant: float, x: np.ndarray) -> float:
    """``x^T Q x + c``. The matrix-side evaluator used in the equivalence test."""
    x = np.asarray(x, dtype=np.float64)
    return float(x @ Q @ x + constant)


# ---------------------------------------------------------------------------
# Slack encoding
# ---------------------------------------------------------------------------
def slack_weights(k: int) -> list[int]:
    """Binary weights representing every integer in ``[0, K]`` exactly once-coverable.

    Standard powers of two, with the top weight clamped so the representable maximum is
    exactly ``K`` rather than ``2^m - 1``:

        w = [1, 2, 4, ..., 2^(m-2), K - 2^(m-1) + 1]

    with ``m = ceil(log2(K+1))``. The weights sum to ``K``, and because the clamped top
    weight never exceeds ``2^(m-1)`` the representable set is the contiguous range
    ``{0, ..., K}``. ``K = 0`` needs no slack at all.
    """
    if k < 0:
        raise ValueError("K must be non-negative")
    if k == 0:
        return []
    m = math.ceil(math.log2(k + 1))
    if m == 0:
        return []
    weights = [1 << t for t in range(m - 1)]
    top = k - ((1 << (m - 1)) - 1)
    if top > 0:
        weights.append(top)
    return weights


def representable_slack_values(weights: list[int]) -> set[int]:
    """Every value the slack bits can represent. Used by the slack-range test."""
    values = {0}
    for w in weights:
        values |= {v + w for v in values}
    return values


# ---------------------------------------------------------------------------
# QUBO assembly
# ---------------------------------------------------------------------------
@dataclass
class QuboProblem:
    """A fully specified QUBO plus everything needed to interpret and reproduce it."""
    polynomial: QuboPolynomial
    Q: np.ndarray
    constant: float
    n_mutation_vars: int
    n_slack_vars: int
    budget_mode: BudgetMode
    budget_k: int
    penalty_budget: float
    penalty_conflict: float
    objective_bound: float
    landscape: MutationLandscape
    build_report: dict

    @property
    def n_vars(self) -> int:
        return self.polynomial.n_vars

    def energy(self, x: np.ndarray) -> float:
        """QUBO energy from the matrix form (the canonical path)."""
        return evaluate_matrix(self.Q, self.constant, x)

    def energy_polynomial(self, x: np.ndarray) -> float:
        """QUBO energy from the polynomial form (independent implementation)."""
        return self.polynomial.evaluate(x)

    def mutation_part(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[: self.n_mutation_vars]

    def slack_value(self, x: np.ndarray) -> int:
        """Integer value encoded by the slack bits (0 when there are none)."""
        if self.n_slack_vars == 0:
            return 0
        bits = np.asarray(x)[self.n_mutation_vars: self.n_mutation_vars + self.n_slack_vars]
        weights = self.build_report["slack_weights"]
        return int(sum(int(b) * int(w) for b, w in zip(bits, weights)))

    def is_mutation_feasible(self, x: np.ndarray) -> bool:
        """Biological feasibility: no same-position conflict, and the budget respected.

        This ignores the slack bits, because they are auxiliary bookkeeping with no
        biological meaning. Use this when decoding a bitstring into a candidate.
        """
        xm = self.mutation_part(x)
        for i, j in self.landscape.mutation_set.conflict_pairs:
            if xm[i] and xm[j]:
                return False
        count = int(np.sum(xm))
        if self.budget_mode is BudgetMode.EXACTLY_K:
            return count == self.budget_k
        return count <= self.budget_k

    def is_feasible(self, x: np.ndarray) -> bool:
        """Feasibility of the full constrained problem, i.e. ALL penalty terms vanish.

        For the slack encoding this additionally requires the slack bits to balance the
        budget equation, ``sum_i x_i + s == K``. That distinction matters: a selection can
        respect the mutation budget while the slack bits are set wrongly, in which case
        the assignment still carries a large budget penalty. Counting such states as
        feasible would overstate the feasible probability and understate the difficulty
        of the QUBO, so they are excluded here.

        Use :meth:`is_mutation_feasible` for the biological notion.
        """
        if not self.is_mutation_feasible(x):
            return False
        if self.budget_mode is BudgetMode.AT_MOST_K and self.n_slack_vars > 0:
            count = int(np.sum(self.mutation_part(x)))
            return count + self.slack_value(x) == self.budget_k
        return True

    def decode(self, x: np.ndarray) -> dict:
        """Decode a bitstring into a named candidate, without scoring it."""
        xm = self.mutation_part(x)
        chosen = [i for i in range(self.n_mutation_vars) if xm[i]]
        mset = self.landscape.mutation_set
        # The sequence is buildable whenever the MUTATION part is feasible; the slack
        # bits carry no biological meaning. Both notions are reported.
        feasible = self.is_mutation_feasible(x)
        out = {
            "selected_indices": chosen,
            "mutations": [mset.candidates[i].label for i in chosen],
            "n_mutations": len(chosen),
            "feasible": feasible,
            "fully_constraint_satisfying": self.is_feasible(x),
            "slack_value": self.slack_value(x),
            "qubo_energy": self.energy(x),
            "surrogate_delta_score": self.landscape.surrogate_delta(xm),
        }
        if feasible:
            out["sequence"] = mset.apply(chosen)
        else:
            out["sequence"] = None
            out["infeasible_reason"] = self._infeasible_reason(xm)
        return out

    def _infeasible_reason(self, xm: np.ndarray) -> str:
        reasons = []
        for i, j in self.landscape.mutation_set.conflict_pairs:
            if xm[i] and xm[j]:
                a = self.landscape.mutation_set.candidates[i].label
                b = self.landscape.mutation_set.candidates[j].label
                reasons.append(f"same-position conflict {a}/{b}")
        count = int(np.sum(xm))
        if self.budget_mode is BudgetMode.EXACTLY_K and count != self.budget_k:
            reasons.append(f"selected {count} mutations, require exactly {self.budget_k}")
        elif self.budget_mode is BudgetMode.AT_MOST_K and count > self.budget_k:
            reasons.append(f"selected {count} mutations, budget is at most {self.budget_k}")
        return "; ".join(reasons) if reasons else "unknown"

    def constraint_residuals(self, x: np.ndarray) -> dict:
        """Per-constraint violation detail, including the slack balance."""
        xm = self.mutation_part(x)
        count = int(np.sum(xm))
        conflicts = [
            f"{self.landscape.mutation_set.candidates[i].label}/"
            f"{self.landscape.mutation_set.candidates[j].label}"
            for i, j in self.landscape.mutation_set.conflict_pairs
            if xm[i] and xm[j]
        ]
        out = {
            "n_mutations_selected": count,
            "budget_k": self.budget_k,
            "budget_mode": self.budget_mode.value,
            "conflict_violations": conflicts,
            "mutation_feasible": self.is_mutation_feasible(x),
            "fully_constraint_satisfying": self.is_feasible(x),
        }
        if self.budget_mode is BudgetMode.AT_MOST_K and self.n_slack_vars > 0:
            s = self.slack_value(x)
            out["slack_value"] = s
            out["budget_equation_residual"] = count + s - self.budget_k
        return out

    def as_dict(self, include_matrix: bool = True) -> dict:
        d = {
            "n_vars": self.n_vars,
            "n_mutation_vars": self.n_mutation_vars,
            "n_slack_vars": self.n_slack_vars,
            "budget_mode": self.budget_mode.value,
            "budget_mode_meaning": self.budget_mode.description,
            "budget_k": self.budget_k,
            "penalty_budget": self.penalty_budget,
            "penalty_conflict": self.penalty_conflict,
            "objective_bound_B": self.objective_bound,
            "constant": self.constant,
            "matrix_convention": (
                "E(x) = x^T Q x + c with Q symmetric; off-diagonal Q_ij = c_ij/2"
            ),
            "linear_terms": {str(k): v for k, v in sorted(self.polynomial.linear.items())},
            "quadratic_terms": {
                f"{i},{j}": v for (i, j), v in sorted(self.polynomial.quadratic.items())
            },
            "variable_map": self.landscape.mutation_set.variable_map(),
            "slack_variable_indices": list(
                range(self.n_mutation_vars, self.n_mutation_vars + self.n_slack_vars)
            ),
            "build_report": self.build_report,
        }
        if include_matrix:
            d["Q"] = self.Q.tolist()
        return d


def objective_range(landscape: MutationLandscape, max_enumerate: int = 22) -> dict:
    """Exact span of the unconstrained objective over all mutation assignments.

    ``B = sum_i |Delta_i| + sum_{i<j} |Delta_ij|`` (see :func:`objective_bound`) is a
    valid bound but a very loose one: it assumes every term can be made to contribute its
    full magnitude with the same sign, which no single assignment achieves. The resulting
    penalty can exceed the real objective scale by two or three orders of magnitude, and
    that coefficient spread is precisely what makes a penalised QUBO hard for QAOA.

    Since ``N`` is small enough to enumerate anyway for the exact baseline, the true span

        span = max_x DeltaS_hat(x) - min_x DeltaS_hat(x)

    can be computed directly. Any penalty strictly greater than ``span`` is sufficient:
    satisfying the constraints costs at most ``span`` in objective, so a violation can
    never pay for itself. This yields a far tighter, still rigorous penalty.
    """
    n = landscape.n
    if n > max_enumerate:
        B = objective_bound(landscape)
        return {
            "mode": "bound_only",
            "reason": f"N={n} exceeds enumeration limit {max_enumerate}",
            "span": B,
            "min": -B,
            "max": B,
        }
    codes = np.arange(1 << n, dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(
        np.float64
    )
    values = bits @ landscape.delta
    for i, j in landscape.compatible_pairs():
        values += float(landscape.delta_pair[i, j]) * bits[:, i] * bits[:, j]
    return {
        "mode": "exact_enumeration",
        "n_assignments": int(1 << n),
        "min": float(values.min()),
        "max": float(values.max()),
        "span": float(values.max() - values.min()),
    }


def objective_bound(landscape: MutationLandscape) -> float:
    """Bound ``B`` on the magnitude of the unconstrained objective.

        B = sum_i |Delta_i| + sum_{i<j} |Delta_ij|

    By the triangle inequality no assignment can make |DeltaS_hat(x)| exceed this, so a
    penalty above ``B`` cannot be out-earned by violating a constraint.
    """
    total = float(np.sum(np.abs(landscape.delta)))
    for i, j in landscape.compatible_pairs():
        total += abs(float(landscape.delta_pair[i, j]))
    return total


def build_qubo(
    landscape: MutationLandscape,
    budget_k: int,
    budget_mode: BudgetMode = BudgetMode.AT_MOST_K,
    penalty_factor: float = 2.0,
    penalty_epsilon: float = 1e-3,
    penalty_budget: float | None = None,
    penalty_conflict: float | None = None,
    penalty_mode: str = "objective_span",
) -> QuboProblem:
    """Assemble the penalised QUBO.

    ``penalty_mode`` selects the scale the penalty is derived from:

      ``objective_span``  (default) ``P = penalty_factor * (span + epsilon)`` where
                          ``span`` is the exact range of the unconstrained objective from
                          :func:`objective_range`. Tightest rigorous choice, and the one
                          that keeps the coefficient spread small enough for QAOA to have
                          a chance.
      ``global_bound``    ``P = penalty_factor * (B + epsilon)`` with ``B`` the triangle-
                          inequality bound. Valid but loose; retained so the
                          penalty-sensitivity study can show what the looser choice costs.

    Either way the choice is then VERIFIED by enumeration
    (:func:`verify_penalty_sufficiency`) rather than trusted.
    """
    if budget_k < 0:
        raise ValueError("budget_k must be non-negative")
    mset = landscape.mutation_set
    n_mut = mset.n
    if budget_k > n_mut:
        raise ValueError(
            f"budget_k={budget_k} exceeds the number of candidate mutations ({n_mut})"
        )

    weights = slack_weights(budget_k) if budget_mode is BudgetMode.AT_MOST_K else []
    n_slack = len(weights)
    n_vars = n_mut + n_slack

    B = objective_bound(landscape)
    span_info = objective_range(landscape)
    if penalty_mode == "objective_span":
        scale_basis = span_info["span"]
    elif penalty_mode == "global_bound":
        scale_basis = B
    else:
        raise ValueError(
            f"unknown penalty_mode {penalty_mode!r}; expected 'objective_span' or "
            "'global_bound'"
        )
    derived = penalty_factor * (scale_basis + penalty_epsilon)
    P = derived if penalty_budget is None else penalty_budget
    P_pos = derived if penalty_conflict is None else penalty_conflict

    poly = QuboPolynomial(n_vars=n_vars)

    # --- objective: minimise -DeltaS_hat(x) --------------------------------
    for i in range(n_mut):
        poly.add_linear(i, -float(landscape.delta[i]))
    for i, j in landscape.compatible_pairs():
        poly.add_quadratic(i, j, -float(landscape.delta_pair[i, j]))

    # --- same-position exclusivity: P_pos * x_i x_j ------------------------
    for i, j in mset.conflict_pairs:
        poly.add_quadratic(i, j, P_pos)

    # --- mutation budget ----------------------------------------------------
    # Both modes expand P * (a . z - K)^2 with a_p = 1 for mutations and a_p = w_t for
    # slack bits. Using z_p^2 = z_p:
    #     linear_p    = P * (a_p^2 - 2 K a_p)
    #     quadratic_pq= P * 2 a_p a_q
    #     constant    = P * K^2
    coeffs: list[tuple[int, float]] = [(i, 1.0) for i in range(n_mut)]
    coeffs += [(n_mut + t, float(w)) for t, w in enumerate(weights)]

    for p, a_p in coeffs:
        poly.add_linear(p, P * (a_p * a_p - 2.0 * budget_k * a_p))
    for (p, a_p), (q, a_q) in combinations(coeffs, 2):
        poly.add_quadratic(p, q, P * 2.0 * a_p * a_q)
    poly.add_constant(P * budget_k * budget_k)

    Q, constant = poly.to_matrix()

    report = {
        "objective": "E(x) = -DeltaS_hat(x) + penalty_budget_term + penalty_conflict_term",
        "sign_convention": (
            "S is maximised biologically; the QUBO minimises its negation, so lower "
            "energy means a better predicted candidate"
        ),
        "objective_bound_B": B,
        "bound_definition": "B = sum_i |Delta_i| + sum_{i<j} |Delta_ij|",
        "objective_span": span_info,
        "penalty_mode": penalty_mode,
        "penalty_strategy": (
            f"P = penalty_factor * ({penalty_mode} + epsilon) = {penalty_factor} * "
            f"({scale_basis:.6f} + {penalty_epsilon}) = {derived:.6f}; any penalty "
            "strictly greater than the objective span is sufficient because satisfying "
            "the constraints costs at most the span (Lucas 2014 principle), and this is "
            "then verified by enumeration rather than assumed"
        ),
        "penalty_factor": penalty_factor,
        "penalty_epsilon": penalty_epsilon,
        "coefficient_spread": (
            float(P / max(abs(scale_basis), 1e-12)) if scale_basis else None
        ),
        "budget_mode": budget_mode.value,
        "budget_mode_meaning": budget_mode.description,
        "slack_weights": weights,
        "slack_representable_values": (
            sorted(representable_slack_values(weights)) if weights else []
        ),
        "n_conflict_pairs_penalised": len(mset.conflict_pairs),
        "n_objective_quadratic_terms": len(landscape.compatible_pairs()),
        "max_abs_coefficient": poly.max_abs_coefficient(),
    }

    return QuboProblem(
        polynomial=poly,
        Q=Q,
        constant=constant,
        n_mutation_vars=n_mut,
        n_slack_vars=n_slack,
        budget_mode=budget_mode,
        budget_k=budget_k,
        penalty_budget=P,
        penalty_conflict=P_pos,
        objective_bound=B,
        landscape=landscape,
        build_report=report,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_matrix_matches_polynomial(
    problem: QuboProblem, n_samples: int = 2000, seed: int = 0, tol: float = 1e-9
) -> dict:
    """Assert ``x^T Q x + c`` equals the explicit polynomial on random bitstrings.

    This is the test that catches the off-diagonal factor-of-two error (spec section 22).
    """
    rng = np.random.default_rng(seed)
    n = problem.n_vars
    max_dev = 0.0
    failures = []

    # always include the structured corner cases
    specials = [np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    for i in range(min(n, 64)):
        e = np.zeros(n, dtype=int)
        e[i] = 1
        specials.append(e)

    samples = specials + [
        rng.integers(0, 2, size=n) for _ in range(max(0, n_samples - len(specials)))
    ]
    for x in samples:
        em = problem.energy(x)
        ep = problem.energy_polynomial(x)
        dev = abs(em - ep)
        max_dev = max(max_dev, dev)
        if dev > tol and len(failures) < 10:
            failures.append(
                {"x": np.asarray(x).tolist(), "matrix": em, "polynomial": ep}
            )

    return {
        "n_samples": len(samples),
        "tolerance": tol,
        "max_abs_deviation": max_dev,
        "passed": max_dev <= tol,
        "failures": failures,
        "convention_checked": "E(x) = x^T Q x + c, Q symmetric, Q_ij = c_ij/2",
    }


def verify_penalty_sufficiency(problem: QuboProblem, max_enumerate_vars: int = 22) -> dict:
    """Check by enumeration that no infeasible assignment beats the feasible optimum.

    Returns ``status: "not evaluated"`` when the problem is too large to enumerate,
    rather than silently claiming success.
    """
    n = problem.n_vars
    if n > max_enumerate_vars:
        return {
            "status": "not evaluated",
            "reason": f"n_vars={n} exceeds enumeration limit {max_enumerate_vars}",
        }

    # Vectorised: a Python loop over 2^n states is unusable past roughly 18 variables,
    # which would make this check silently unavailable exactly where it matters most.
    from backend.optimization.solvers import (
        _all_bitstrings,
        feasibility_mask_vectorised,
    )

    bits = _all_bitstrings(n)
    X = bits.astype(np.float64)
    energies = np.einsum("ni,ni->n", X @ problem.Q, X) + problem.constant
    feasible = feasibility_mask_vectorised(problem, bits)

    n_feasible = int(feasible.sum())
    best_feasible = float(energies[feasible].min()) if n_feasible else math.inf
    worst_violator: dict | None = None
    if n_feasible < energies.size:
        inf_idx = np.flatnonzero(~feasible)
        k = int(inf_idx[np.argmin(energies[inf_idx])])
        best_infeasible = float(energies[k])
        xv = bits[k].astype(int)
        worst_violator = {
            "x": xv.tolist(),
            "energy": best_infeasible,
            "reason": problem._infeasible_reason(problem.mutation_part(xv)),
        }
    else:
        best_infeasible = math.inf

    margin = (
        best_infeasible - best_feasible
        if math.isfinite(best_infeasible) and math.isfinite(best_feasible)
        else None
    )
    return {
        "status": "evaluated",
        "n_assignments": 1 << n,
        "n_feasible": n_feasible,
        "n_infeasible": (1 << n) - n_feasible,
        "best_feasible_energy": None if not math.isfinite(best_feasible) else best_feasible,
        "best_infeasible_energy": (
            None if not math.isfinite(best_infeasible) else best_infeasible
        ),
        "margin": margin,
        "passed": bool(margin is None or margin > 0),
        "lowest_energy_infeasible_assignment": worst_violator,
        "interpretation": (
            "passed means every infeasible assignment has strictly higher energy than the "
            "best feasible one, so the penalties cannot be out-earned"
        ),
    }


def verify_slack_range(budget_k: int) -> dict:
    """Assert the slack bits represent exactly ``{0, ..., K}``."""
    w = slack_weights(budget_k)
    values = representable_slack_values(w)
    expected = set(range(budget_k + 1))
    return {
        "budget_k": budget_k,
        "weights": w,
        "n_slack_bits": len(w),
        "representable": sorted(values),
        "expected": sorted(expected),
        "passed": values == expected,
    }
