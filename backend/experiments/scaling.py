"""Scaling benchmark: exact vs classical vs QAOA across problem sizes.

Spec section 24: benchmarking a single instance says very little, so the same pipeline is
run at several values of N and the methods compared at each size against the exact
optimum.

This measures how the methods behave as the instance grows on this hardware and at these
sizes. It is far too small to support any claim about asymptotic scaling, and certainly
not about quantum advantage -- the literature review records why (Boulebnane et al. 2022
found QAOA matched by random sampling on a comparable peptide problem).

Run:  python -m backend.experiments.scaling
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from backend.models.property_models import BiologicalScorer, ScoreWeights
from backend.optimization.ising import qubo_to_ising, verify_ising_mapping
from backend.optimization.landscape import compute_landscape
from backend.optimization.mutations import generate_mutation_set
from backend.optimization.pipeline import (
    DEFAULT_PARENT,
    DEFAULT_SEED,
    benchmark_metrics,
    environment_record,
    _json_default,
)
from backend.optimization.qaoa import run_qaoa_ideal, verify_simulator_agreement
from backend.optimization.qubo import (
    BudgetMode,
    build_qubo,
    verify_matrix_matches_polynomial,
    verify_penalty_sufficiency,
)
from backend.optimization.solvers import run_all_classical, solve_exact

RESULTS_DIR = Path("results/experiments")
# N=18 gives 20 qubits (about 1e6 assignments), the largest size where exact enumeration,
# the penalty-sufficiency check and the exhaustive Ising verification all stay comfortable
# in memory. Beyond that the exact baseline -- the entire point of this benchmark -- is no
# longer available, so there is nothing to compare against.
SIZES = (6, 8, 10, 12, 14, 16, 18)
BUDGET_K = 3
QAOA_DEPTHS = (1, 2, 3)


def run_size(
    n_mutations: int,
    scorer: BiologicalScorer,
    budget_k: int = BUDGET_K,
    depths: tuple[int, ...] = QAOA_DEPTHS,
    seed: int = DEFAULT_SEED,
    shots: int = 8192,
    max_qubits_exact: int = 24,
    verbose: bool = True,
) -> dict:
    """One point of the scaling curve."""
    t_total = time.perf_counter()
    mset = generate_mutation_set(
        DEFAULT_PARENT, scorer, target_n=n_mutations, max_per_position=2
    )
    landscape = compute_landscape(mset, scorer)
    problem = build_qubo(landscape, budget_k=budget_k, budget_mode=BudgetMode.AT_MOST_K)

    record: dict = {
        "target_n_mutations": n_mutations,
        "n_mutation_vars": problem.n_mutation_vars,
        "n_slack_vars": problem.n_slack_vars,
        "n_qubits": problem.n_vars,
        "budget_k": budget_k,
        "n_conflict_pairs": len(mset.conflict_pairs),
        "n_objective_quadratic_terms": len(landscape.compatible_pairs()),
        "interaction_strength_ratio": landscape.report["interaction_strength_ratio"],
        "penalty": problem.penalty_budget,
        "objective_span": problem.build_report["objective_span"]["span"],
        "n_model_evaluations_for_coefficients": landscape.n_model_evaluations,
    }

    # verification at every size, not just the headline instance
    hamiltonian, _ = qubo_to_ising(problem)
    record["verification"] = {
        "matrix_vs_polynomial": verify_matrix_matches_polynomial(problem, n_samples=500)[
            "passed"
        ],
        "penalty_sufficiency": verify_penalty_sufficiency(problem).get("passed"),
        "ising_mapping": verify_ising_mapping(
            problem, hamiltonian, max_enumerate_vars=18, n_random=4000
        )["passed"],
        "simulator_agreement": verify_simulator_agreement(problem, p=2)["passed"],
    }

    if problem.n_vars > max_qubits_exact:
        record["exact"] = {
            "status": "not evaluated",
            "reason": f"n_qubits={problem.n_vars} exceeds enumeration limit",
        }
        record["runtime_seconds"] = time.perf_counter() - t_total
        return record

    exact = solve_exact(problem, max_vars=max_qubits_exact)
    uniform = len(exact.optimal_solutions) / exact.n_assignments
    record["exact"] = {
        "optimal_energy": exact.optimal_energy,
        "n_optimal_solutions": len(exact.optimal_solutions),
        "n_assignments": exact.n_assignments,
        "n_feasible": exact.n_feasible,
        "runtime_seconds": exact.runtime_seconds,
        "uniform_random_success_probability": uniform,
    }

    classical = run_all_classical(problem, seed=seed)
    record["classical"] = {
        name: {
            "best_energy": res.best_energy,
            "mean_energy": res.mean_energy,
            "std_energy": res.std_energy,
            "n_evaluations": res.n_evaluations,
            "runtime_seconds": res.runtime_seconds,
            "feasible": res.feasible,
            **benchmark_metrics(res.best_energy, exact),
        }
        for name, res in classical.items()
    }

    qaoa: dict = {}
    for p in depths:
        res = run_qaoa_ideal(
            problem,
            p=p,
            optimizer="COBYLA",
            maxiter=300,
            seed=seed % (2**31),
            optimal_solutions=exact.optimal_solutions,
            n_restarts=3,
            shots=shots,
        )
        cm = res.circuit_metrics
        qaoa[f"p{p}"] = {
            "p": p,
            "final_expectation": res.final_expectation,
            "success_probability": res.success_probability,
            "success_probability_vs_uniform": (
                res.success_probability / uniform if uniform > 0 else None
            ),
            "feasible_probability": res.feasible_probability,
            "n_function_evaluations": res.n_function_evaluations,
            "runtime_seconds": res.runtime_seconds,
            "logical_depth": cm["logical_depth"],
            "logical_two_qubit_gates": cm["logical_two_qubit_gates"],
            "transpiled_depth": cm.get("transpiled", {}).get("depth"),
            "transpiled_two_qubit_gates": cm.get("transpiled", {}).get("two_qubit_gates"),
            **benchmark_metrics(res.best_sampled_energy, exact),
        }
    record["qaoa"] = qaoa
    record["runtime_seconds"] = time.perf_counter() - t_total

    if verbose:
        q3 = qaoa.get(f"p{max(depths)}", {})
        print(
            f"  N={problem.n_mutation_vars:2d} qubits={problem.n_vars:2d} "
            f"E*={exact.optimal_energy:9.5f} "
            f"exact={exact.runtime_seconds:6.3f}s "
            f"SA_gap={record['classical']['simulated_annealing']['absolute_gap']:.2e} "
            f"rand_gap={record['classical']['random_sampling']['absolute_gap']:.2e} "
            f"QAOA_p{max(depths)}_succ={q3.get('success_probability', 0):.5f} "
            f"({q3.get('success_probability_vs_uniform') or 0:.1f}x)",
            flush=True,
        )
    return record


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scorer = BiologicalScorer.load(weights=ScoreWeights(1.0, 1.0))

    print("scaling benchmark: exact vs classical vs QAOA")
    print(f"parent={DEFAULT_PARENT}  budget_k={BUDGET_K}  depths={QAOA_DEPTHS}")
    points = []
    for n in SIZES:
        points.append(run_size(n, scorer, verbose=True))

    summary = {
        "experiment": "scaling_benchmark",
        "parent": DEFAULT_PARENT,
        "budget_k": BUDGET_K,
        "qaoa_depths": list(QAOA_DEPTHS),
        "sizes_requested": list(SIZES),
        "seed": DEFAULT_SEED,
        "environment": environment_record(),
        "points": points,
        "caveats": [
            "These sizes are far too small to support any claim about asymptotic "
            "scaling, and no claim of quantum advantage is made.",
            "QAOA runs on an ideal statevector simulator with an exact objective; it is "
            "not competing with the classical solvers on runtime, and its runtime is "
            "simulation cost rather than quantum execution cost.",
            "Uniform random sampling is included because it is the baseline that matched "
            "QAOA on a comparable peptide problem in Boulebnane et al. (2022).",
            "Classical solvers see exactly the same QUBO and no biological information.",
        ],
    }
    path = RESULTS_DIR / "scaling_benchmark.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"\nreport -> {path}")

    # compact table
    print("\n N  qubits      E*      exact_s   SA_gap  LS_gap  greedy_gap  rand_gap  "
          "p1_succ  p2_succ  p3_succ  p3_depth  p3_2q")
    for pt in points:
        if pt.get("exact", {}).get("status") == "not evaluated":
            print(f"{pt['n_mutation_vars']:3d} {pt['n_qubits']:6d}   not evaluated")
            continue
        c = pt["classical"]
        q = pt["qaoa"]
        print(
            f"{pt['n_mutation_vars']:3d} {pt['n_qubits']:6d} "
            f"{pt['exact']['optimal_energy']:9.5f} {pt['exact']['runtime_seconds']:8.3f} "
            f"{c['simulated_annealing']['absolute_gap']:8.1e} "
            f"{c['local_search']['absolute_gap']:7.1e} "
            f"{c['greedy']['absolute_gap']:10.1e} "
            f"{c['random_sampling']['absolute_gap']:9.1e} "
            f"{q['p1']['success_probability']:8.5f} "
            f"{q['p2']['success_probability']:8.5f} "
            f"{q['p3']['success_probability']:8.5f} "
            f"{q['p3']['logical_depth']:9d} {q['p3']['logical_two_qubit_gates']:6d}"
        )


if __name__ == "__main__":
    main()
