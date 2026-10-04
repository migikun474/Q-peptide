"""Exact and classical solvers. All operate on the identical QuboProblem object.

The classical solvers receive only the QUBO -- never the biological models, the parent
sequence, or the mutation semantics. That restriction is deliberate: giving a classical
baseline information QAOA does not have would make the comparison meaningless
(spec section 25).

Solvers:
    solve_exact           brute-force enumeration; ground truth for benchmarking
    solve_simulated_annealing
    solve_greedy
    solve_local_search
    solve_random_sampling  the baseline that matched QAOA in Boulebnane et al. 2022
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from backend.optimization.qubo import QuboProblem


@dataclass
class SolverResult:
    """Uniform result record, so every method reports the same fields."""
    method: str
    best_energy: float
    best_x: np.ndarray
    n_evaluations: int
    runtime_seconds: float
    feasible: bool
    # populated by stochastic solvers run over multiple seeds
    per_seed_energies: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def mean_energy(self) -> float | None:
        return float(np.mean(self.per_seed_energies)) if self.per_seed_energies else None

    @property
    def std_energy(self) -> float | None:
        return float(np.std(self.per_seed_energies)) if self.per_seed_energies else None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "best_energy": self.best_energy,
            "best_x": np.asarray(self.best_x).tolist(),
            "n_evaluations": self.n_evaluations,
            "runtime_seconds": self.runtime_seconds,
            "feasible": self.feasible,
            "mean_energy": self.mean_energy,
            "std_energy": self.std_energy,
            "n_seeds": len(self.per_seed_energies) or None,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class ExactResult:
    """Full enumeration outcome, including the degenerate optimum set."""
    optimal_energy: float
    optimal_solutions: list[np.ndarray]
    optimal_feasible_solutions: list[np.ndarray]
    best_feasible_energy: float
    n_assignments: int
    n_feasible: int
    runtime_seconds: float
    energy_percentiles: dict
    feasible_energy_sorted: np.ndarray

    def as_dict(self, max_solutions: int = 20) -> dict:
        return {
            "optimal_energy": self.optimal_energy,
            "n_optimal_solutions": len(self.optimal_solutions),
            "optimal_solutions": [s.tolist() for s in self.optimal_solutions[:max_solutions]],
            "best_feasible_energy": self.best_feasible_energy,
            "n_optimal_feasible_solutions": len(self.optimal_feasible_solutions),
            "optimal_feasible_solutions": [
                s.tolist() for s in self.optimal_feasible_solutions[:max_solutions]
            ],
            "n_assignments": self.n_assignments,
            "n_feasible": self.n_feasible,
            "runtime_seconds": self.runtime_seconds,
            "energy_percentiles": self.energy_percentiles,
        }


def feasibility_mask_vectorised(problem: QuboProblem, bits: np.ndarray) -> np.ndarray:
    """Vectorised full-constraint feasibility over a (M, n) bit matrix.

    A per-state Python loop costs minutes at N=20 (2^20 states), which would make the
    scaling benchmark impractical. This computes the same predicate as
    ``QuboProblem.is_feasible`` with array operations; a test asserts the two agree.
    """
    from backend.optimization.qubo import BudgetMode

    n_mut = problem.n_mutation_vars
    counts = bits[:, :n_mut].sum(axis=1)
    ok = np.ones(bits.shape[0], dtype=bool)
    for i, j in problem.landscape.mutation_set.conflict_pairs:
        ok &= ~((bits[:, i] == 1) & (bits[:, j] == 1))
    if problem.budget_mode is BudgetMode.EXACTLY_K:
        ok &= counts == problem.budget_k
    else:
        ok &= counts <= problem.budget_k
        if problem.n_slack_vars > 0:
            weights = np.array(problem.build_report["slack_weights"], dtype=np.int64)
            slack = (
                bits[:, n_mut: n_mut + problem.n_slack_vars].astype(np.int64) @ weights
            )
            ok &= (counts + slack) == problem.budget_k
    return ok


def _all_bitstrings(n: int) -> np.ndarray:
    """All 2^n bitstrings as a (2^n, n) uint8 array, little-endian bit order."""
    codes = np.arange(1 << n, dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(np.uint8)
    return bits


def solve_exact(problem: QuboProblem, max_vars: int = 24) -> ExactResult:
    """Enumerate all 2^n assignments and return the ground truth.

    Energies are computed in one vectorised pass: for binary x,
    ``x^T Q x = sum_i Q_ii x_i + 2 sum_{i<j} Q_ij x_i x_j``, which equals
    ``einsum('ni,ij,nj->n', X, Q, X)``.

    ``optimal_solutions`` are the global minima of the penalised energy over ALL
    assignments; ``optimal_feasible_solutions`` are the minima restricted to feasible
    assignments. With sufficient penalties these coincide, and the penalty-sufficiency
    check asserts exactly that.
    """
    n = problem.n_vars
    if n > max_vars:
        raise ValueError(
            f"exact enumeration refused: n_vars={n} > max_vars={max_vars} "
            f"({1 << n} assignments). Raise max_vars deliberately if intended."
        )
    t0 = time.perf_counter()
    bits = _all_bitstrings(n)
    X = bits.astype(np.float64)
    # x^T Q x for every row, without materialising an (M, n, n) tensor
    energies = np.einsum("ni,ni->n", X @ problem.Q, X) + problem.constant
    feasible_mask = feasibility_mask_vectorised(problem, bits)

    e_min = float(energies.min())
    opt_idx = np.flatnonzero(np.isclose(energies, e_min, rtol=0.0, atol=1e-9))

    if feasible_mask.any():
        feas_energies = energies[feasible_mask]
        e_feas_min = float(feas_energies.min())
        feas_idx = np.flatnonzero(feasible_mask)
        opt_feas_idx = feas_idx[
            np.isclose(energies[feas_idx], e_feas_min, rtol=0.0, atol=1e-9)
        ]
        feas_sorted = np.sort(feas_energies)
    else:
        e_feas_min = math.inf
        opt_feas_idx = np.array([], dtype=int)
        feas_sorted = np.array([])

    runtime = time.perf_counter() - t0
    return ExactResult(
        optimal_energy=e_min,
        optimal_solutions=[X[k].astype(int) for k in opt_idx],
        optimal_feasible_solutions=[X[k].astype(int) for k in opt_feas_idx],
        best_feasible_energy=e_feas_min,
        n_assignments=int(X.shape[0]),
        n_feasible=int(feasible_mask.sum()),
        runtime_seconds=runtime,
        energy_percentiles={
            "min": float(energies.min()),
            "p1": float(np.percentile(energies, 1)),
            "p25": float(np.percentile(energies, 25)),
            "median": float(np.median(energies)),
            "p75": float(np.percentile(energies, 75)),
            "max": float(energies.max()),
            "mean": float(energies.mean()),
        },
        feasible_energy_sorted=feas_sorted,
    )


# ---------------------------------------------------------------------------
# Classical heuristics
# ---------------------------------------------------------------------------
def _delta_energy(Q: np.ndarray, x: np.ndarray, i: int) -> float:
    """Energy change from flipping bit ``i``, in O(n).

    For ``E = x^T Q x`` with symmetric Q, flipping x_i by ``s = 1 - 2 x_i`` gives
    ``dE = s * (Q_ii + 2 * sum_{j != i} Q_ij x_j)``.
    """
    s = 1.0 - 2.0 * x[i]
    interaction = float(Q[i] @ x) - Q[i, i] * x[i]
    return s * (Q[i, i] + 2.0 * interaction)


def solve_simulated_annealing(
    problem: QuboProblem,
    n_seeds: int = 10,
    n_sweeps: int = 2000,
    t_start: float | None = None,
    t_end: float = 1e-3,
    seed: int = 0,
) -> SolverResult:
    """Simulated annealing with a geometric temperature schedule.

    ``t_start`` defaults to the largest absolute coefficient magnitude, which scales the
    schedule to the problem rather than to an arbitrary constant. Each seed is an
    independent run; the best is returned and the spread reported.
    """
    Q = problem.Q
    n = problem.n_vars
    if t_start is None:
        t_start = max(problem.polynomial.max_abs_coefficient(), 1.0)

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_overall = math.inf
    best_x = np.zeros(n, dtype=int)
    per_seed: list[float] = []
    evals = 0

    cooling = (t_end / t_start) ** (1.0 / max(n_sweeps - 1, 1))

    for s in range(n_seeds):
        x = rng.integers(0, 2, size=n).astype(np.float64)
        e = problem.energy(x)
        evals += 1
        best_e, best_local = e, x.copy()
        T = t_start
        for _ in range(n_sweeps):
            for i in rng.permutation(n):
                dE = _delta_energy(Q, x, int(i))
                evals += 1
                if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
                    x[i] = 1.0 - x[i]
                    e += dE
                    if e < best_e:
                        best_e, best_local = e, x.copy()
            T *= cooling
        per_seed.append(float(best_e))
        if best_e < best_overall:
            best_overall, best_x = float(best_e), best_local.astype(int)

    runtime = time.perf_counter() - t0
    return SolverResult(
        method="simulated_annealing",
        best_energy=best_overall,
        best_x=best_x,
        n_evaluations=evals,
        runtime_seconds=runtime,
        feasible=problem.is_feasible(best_x),
        per_seed_energies=per_seed,
        extra={
            "n_seeds": n_seeds,
            "n_sweeps": n_sweeps,
            "t_start": t_start,
            "t_end": t_end,
            "schedule": "geometric",
            "base_seed": seed,
        },
    )


def solve_greedy(problem: QuboProblem) -> SolverResult:
    """Deterministic greedy descent from the all-zero state.

    Repeatedly flips the single bit giving the largest energy decrease until none does.
    """
    n = problem.n_vars
    t0 = time.perf_counter()
    x = np.zeros(n, dtype=np.float64)
    e = problem.energy(x)
    evals = 1
    improved = True
    while improved:
        improved = False
        deltas = np.array([_delta_energy(problem.Q, x, i) for i in range(n)])
        evals += n
        i = int(np.argmin(deltas))
        if deltas[i] < -1e-12:
            x[i] = 1.0 - x[i]
            e += float(deltas[i])
            improved = True
    runtime = time.perf_counter() - t0
    xi = x.astype(int)
    return SolverResult(
        method="greedy",
        best_energy=float(e),
        best_x=xi,
        n_evaluations=evals,
        runtime_seconds=runtime,
        feasible=problem.is_feasible(xi),
        extra={"start": "all_zeros", "deterministic": True},
    )


def solve_local_search(
    problem: QuboProblem, n_seeds: int = 20, seed: int = 0
) -> SolverResult:
    """Multi-start steepest-descent local search from random states."""
    n = problem.n_vars
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_overall, best_x = math.inf, np.zeros(n, dtype=int)
    per_seed: list[float] = []
    evals = 0

    for _ in range(n_seeds):
        x = rng.integers(0, 2, size=n).astype(np.float64)
        e = problem.energy(x)
        evals += 1
        while True:
            deltas = np.array([_delta_energy(problem.Q, x, i) for i in range(n)])
            evals += n
            i = int(np.argmin(deltas))
            if deltas[i] >= -1e-12:
                break
            x[i] = 1.0 - x[i]
            e += float(deltas[i])
        per_seed.append(float(e))
        if e < best_overall:
            best_overall, best_x = float(e), x.astype(int)

    runtime = time.perf_counter() - t0
    return SolverResult(
        method="local_search",
        best_energy=best_overall,
        best_x=best_x,
        n_evaluations=evals,
        runtime_seconds=runtime,
        feasible=problem.is_feasible(best_x),
        per_seed_energies=per_seed,
        extra={"n_restarts": n_seeds, "base_seed": seed},
    )


def solve_random_sampling(
    problem: QuboProblem, n_samples: int = 10000, n_seeds: int = 5, seed: int = 0
) -> SolverResult:
    """Uniform random bitstring sampling.

    This is a mandatory baseline, not a filler: Boulebnane et al. (2022) found QAOA's
    performance on a peptide problem "can be matched by random sampling up to a small
    overhead". Any QAOA result here is reported against this.
    """
    n = problem.n_vars
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_overall, best_x = math.inf, np.zeros(n, dtype=int)
    per_seed: list[float] = []

    for _ in range(n_seeds):
        X = rng.integers(0, 2, size=(n_samples, n)).astype(np.float64)
        energies = np.einsum("ni,ni->n", X @ problem.Q, X) + problem.constant
        k = int(np.argmin(energies))
        per_seed.append(float(energies[k]))
        if energies[k] < best_overall:
            best_overall, best_x = float(energies[k]), X[k].astype(int)

    runtime = time.perf_counter() - t0
    return SolverResult(
        method="random_sampling",
        best_energy=best_overall,
        best_x=best_x,
        n_evaluations=n_samples * n_seeds,
        runtime_seconds=runtime,
        feasible=problem.is_feasible(best_x),
        per_seed_energies=per_seed,
        extra={"n_samples_per_seed": n_samples, "n_seeds": n_seeds, "base_seed": seed},
    )


def run_all_classical(
    problem: QuboProblem, seed: int = 0, sa_seeds: int = 10
) -> dict[str, SolverResult]:
    """Every classical baseline on the same QUBO."""
    return {
        "simulated_annealing": solve_simulated_annealing(
            problem, n_seeds=sa_seeds, seed=seed
        ),
        "greedy": solve_greedy(problem),
        "local_search": solve_local_search(problem, seed=seed),
        "random_sampling": solve_random_sampling(problem, seed=seed),
    }
