"""The end-to-end optimization pipeline, shared by the API and the experiment scripts.

Stages (spec section 57):

    parent -> candidate mutations -> ML mutation landscape -> QUBO
           -> exact -> classical -> QAOA (p sweep) -> noise -> optional QPU
           -> independent ML re-scoring -> Pareto analysis

Everything a run needs to be reproduced is recorded in the manifest: seeds, model
configuration, normalization parameters, candidate set, QUBO coefficients, penalties,
depths, optimizer, shots, noise model and backend (spec section 54).

Nothing here fabricates a result. A stage that did not execute is recorded as
``{"status": "not evaluated", "reason": ...}``.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.models.property_models import BiologicalScorer, ScoreWeights
from backend.optimization.analysis import (
    Candidate,
    attach_trustworthiness,
    pareto_analysis,
    rescore_candidates,
    surrogate_error_summary,
)
from backend.optimization.ising import qubo_to_ising, verify_ising_mapping, x_to_bitstring
from backend.optimization.landscape import (
    MutationLandscape,
    compute_landscape,
    surrogate_fidelity_exhaustive,
    validate_surrogate_beyond_pairs,
    verify_landscape_identities,
)
from backend.optimization.mutations import compatibility_graph, generate_mutation_set
from backend.optimization.qaoa import (
    bitstring_to_x,
    run_qaoa_ideal,
    sample_distribution,
    verify_simulator_agreement,
)
from backend.optimization.qubo import (
    BudgetMode,
    QuboProblem,
    build_qubo,
    verify_matrix_matches_polynomial,
    verify_penalty_sufficiency,
    verify_slack_range,
)
from backend.optimization.solvers import (
    ExactResult,
    run_all_classical,
    solve_exact,
)

RESULTS_DIR = Path("results/experiments")
DEFAULT_SEED = 20261004

# Magainin 2. A classic, extensively characterised alpha-helical AMP and the subject of
# the charge/hydrophobicity trade-off literature that motivates the two-term objective
# (research/literature_review.md section 3.1). Configurable; this is only the default.
DEFAULT_PARENT = "GIGKFLHSAKKFGKAFVGEIMNS"


def environment_record() -> dict:
    """Versions and platform, for the reproducibility manifest."""
    import importlib

    mods = [
        "numpy", "pandas", "scipy", "sklearn", "xgboost", "qiskit", "qiskit_aer",
        "qiskit_ibm_runtime",
    ]
    versions = {}
    for m in mods:
        try:
            versions[m] = getattr(importlib.import_module(m), "__version__", "unknown")
        except Exception:
            versions[m] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


@dataclass
class OptimizationConfig:
    """Everything that defines a run."""
    parent: str = DEFAULT_PARENT
    budget_k: int = 3
    budget_mode: BudgetMode = BudgetMode.AT_MOST_K
    alpha: float = 1.0
    beta: float = 1.0
    target_n: int = 14
    max_per_position: int = 2
    penalty_factor: float = 2.0
    penalty_mode: str = "objective_span"
    qaoa_depths: tuple[int, ...] = (1, 2, 3)
    qaoa_optimizers: tuple[str, ...] = ("COBYLA", "Powell")
    qaoa_maxiter: int = 300
    qaoa_restarts: int = 3
    shots: int = 8192
    seed: int = DEFAULT_SEED
    screen: str = "abs_delta"

    def as_dict(self) -> dict:
        d = {
            "parent": self.parent,
            "budget_k": self.budget_k,
            "budget_mode": self.budget_mode.value,
            "budget_mode_meaning": self.budget_mode.description,
            "alpha": self.alpha,
            "beta": self.beta,
            "target_n": self.target_n,
            "max_per_position": self.max_per_position,
            "penalty_factor": self.penalty_factor,
            "penalty_mode": self.penalty_mode,
            "qaoa_depths": list(self.qaoa_depths),
            "qaoa_optimizers": list(self.qaoa_optimizers),
            "qaoa_maxiter": self.qaoa_maxiter,
            "qaoa_restarts": self.qaoa_restarts,
            "shots": self.shots,
            "seed": self.seed,
            "screen": self.screen,
        }
        return d


@dataclass
class PipelineStageResults:
    """Accumulated stage outputs, serialised as the run record."""
    config: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    parent_analysis: dict = field(default_factory=dict)
    mutation_set: dict = field(default_factory=dict)
    compatibility_graph: dict = field(default_factory=dict)
    landscape: dict = field(default_factory=dict)
    qubo: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    exact: dict = field(default_factory=dict)
    classical: dict = field(default_factory=dict)
    qaoa: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    surrogate_validation: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    trustworthiness: dict = field(default_factory=dict)
    pareto: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "environment": self.environment,
            "parent_analysis": self.parent_analysis,
            "mutation_set": self.mutation_set,
            "compatibility_graph": self.compatibility_graph,
            "landscape": self.landscape,
            "qubo": self.qubo,
            "verification": self.verification,
            "exact": self.exact,
            "classical": self.classical,
            "qaoa": self.qaoa,
            "noise": self.noise,
            "hardware": self.hardware,
            "surrogate_validation": self.surrogate_validation,
            "candidates": self.candidates,
            "trustworthiness": self.trustworthiness,
            "pareto": self.pareto,
            "timings": self.timings,
        }


def analyse_parent(parent: str, scorer: BiologicalScorer) -> dict:
    """Descriptors and model predictions for the parent peptide."""
    from backend.utils.peptide import (
        aliphatic_index,
        aromaticity,
        hydrophobic_moment,
        isoelectric_point,
        mean_hydrophobicity,
        mean_polarity,
        molecular_weight,
        net_charge,
        validate_sequence,
    )

    seq = validate_sequence(parent, min_len=2)
    b = scorer.breakdown([seq])
    return {
        "sequence": seq,
        "length": len(seq),
        "molecular_weight": molecular_weight(seq),
        "net_charge_ph74": net_charge(seq, 7.4),
        "isoelectric_point": isoelectric_point(seq),
        "mean_hydrophobicity_kd": mean_hydrophobicity(seq),
        "hydrophobic_moment": hydrophobic_moment(seq),
        "mean_polarity": mean_polarity(seq),
        "aromaticity": aromaticity(seq),
        "aliphatic_index": aliphatic_index(seq),
        "model_predicted_activity": float(b.activity_raw[0]),
        "model_predicted_hemolysis": float(b.hemolysis_raw[0]),
        "activity_normalized": float(b.activity_norm[0]),
        "hemolysis_normalized": float(b.hemolysis_norm[0]),
        "score": float(b.score[0]),
        "units": {
            "model_predicted_activity": "pMIC-scale, 6 - log10(MIC[uM]); higher = more active",
            "model_predicted_hemolysis": (
                "p-dose scale, 6 - log10(D50[uM]); higher = MORE hemolytic"
            ),
        },
        "caveat": (
            "model-predicted values only; no experimental measurement is claimed for "
            "this peptide by this tool"
        ),
    }


def build_problem(
    config: OptimizationConfig, scorer: BiologicalScorer
) -> tuple[QuboProblem, MutationLandscape, dict]:
    """Stages 1-3: candidates, landscape, QUBO."""
    timings = {}
    t0 = time.perf_counter()
    mset = generate_mutation_set(
        config.parent,
        scorer,
        target_n=config.target_n,
        max_per_position=config.max_per_position,
        screen=config.screen,
    )
    timings["mutation_generation_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    landscape = compute_landscape(mset, scorer)
    timings["landscape_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    problem = build_qubo(
        landscape,
        budget_k=config.budget_k,
        budget_mode=config.budget_mode,
        penalty_factor=config.penalty_factor,
        penalty_mode=config.penalty_mode,
    )
    timings["qubo_build_seconds"] = time.perf_counter() - t0
    return problem, landscape, timings


def verify_qubo_optimum_is_true_best(
    problem: QuboProblem, exact: ExactResult, scorer, limit: int = 60000
) -> dict:
    """Check whether the QUBO optimum really is the best feasible candidate under the ML
    models.

    This is the end-to-end claim that matters, and it is checkable directly: enumerate
    every feasible mutation selection, score each one by RUNNING THE MODELS, and compare
    the true best against what minimising the QUBO returns.

    It must pass whenever the budget keeps the surrogate in its exact regime (K <= 2) and
    may legitimately fail for larger budgets, where the second-order model omits
    higher-order interactions. Either outcome is recorded; neither is assumed.
    """
    from itertools import combinations

    mset = problem.landscape.mutation_set
    n = mset.n
    conflict = set(mset.conflict_pairs)
    k_max = problem.budget_k
    exactly = problem.budget_mode is BudgetMode.EXACTLY_K

    selections: list[tuple[int, ...]] = []
    sizes = [k_max] if exactly else range(0, k_max + 1)
    for k in sizes:
        for sel in combinations(range(n), k):
            if any((a, b) in conflict for a, b in combinations(sel, 2)):
                continue
            selections.append(sel)
            if len(selections) > limit:
                return {
                    "status": "not evaluated",
                    "reason": f"more than {limit} feasible selections to score",
                }
    if not selections:
        return {"status": "not evaluated", "reason": "no feasible selections"}

    sequences = [mset.apply(list(s)) for s in selections]
    parent_score = scorer.score_one(mset.parent)
    true_delta = np.asarray(scorer.score(sequences), dtype=float) - parent_score
    best_true = int(np.argmax(true_delta))

    # what the QUBO's optimum decodes to
    if not exact.optimal_feasible_solutions:
        return {"status": "not evaluated", "reason": "exact solver found no feasible optimum"}
    decoded = problem.decode(np.asarray(exact.optimal_feasible_solutions[0]))
    qubo_choice = tuple(sorted(decoded["selected_indices"]))
    try:
        qubo_idx = selections.index(qubo_choice)
    except ValueError:
        return {
            "status": "not evaluated",
            "reason": "the QUBO optimum is not among the enumerated feasible selections",
        }

    order = np.argsort(-true_delta)
    rank = int(np.where(order == qubo_idx)[0][0]) + 1
    matches = bool(abs(true_delta[qubo_idx] - true_delta[best_true]) <= 1e-9)

    return {
        "status": "evaluated",
        "n_feasible_selections_scored": len(selections),
        "budget_k": k_max,
        "budget_mode": problem.budget_mode.value,
        "qubo_optimum": {
            "mutations": decoded["mutations"],
            "sequence": decoded["sequence"],
            "true_delta_score": float(true_delta[qubo_idx]),
            "true_rank": rank,
            "out_of": len(selections),
        },
        "true_best": {
            "mutations": [mset.candidates[i].label for i in selections[best_true]],
            "sequence": sequences[best_true],
            "true_delta_score": float(true_delta[best_true]),
        },
        "qubo_optimum_is_true_best": matches,
        "expected_to_pass": bool(k_max <= 2),
        "interpretation": (
            "with K <= 2 the surrogate is exact, so this must pass; a failure there would "
            "indicate an implementation bug. For K >= 3 a failure is a genuine property "
            "of the second-order approximation, not a bug."
        ),
    }


def run_verification(problem: QuboProblem, landscape: MutationLandscape, scorer) -> dict:
    """All mathematical checks, with their results recorded rather than assumed."""
    hamiltonian, ising_report = qubo_to_ising(problem)
    out = {
        "landscape_identities": verify_landscape_identities(landscape, scorer),
        "qubo_matrix_vs_polynomial": verify_matrix_matches_polynomial(problem, seed=0),
        "penalty_sufficiency": verify_penalty_sufficiency(problem),
        "slack_range": verify_slack_range(problem.budget_k),
        "ising_mapping": verify_ising_mapping(problem, hamiltonian),
        "ising_report": ising_report,
        "simulator_agreement": verify_simulator_agreement(problem, p=2, seed=0),
    }
    checks = [
        out["landscape_identities"]["passed"],
        out["qubo_matrix_vs_polynomial"]["passed"],
        out["ising_mapping"]["passed"],
        out["simulator_agreement"]["passed"],
    ]
    if out["penalty_sufficiency"].get("status") == "evaluated":
        checks.append(out["penalty_sufficiency"]["passed"])
    if problem.budget_mode is BudgetMode.AT_MOST_K:
        checks.append(out["slack_range"]["passed"])
    out["all_passed"] = all(checks)
    return out


def benchmark_metrics(found_energy: float, exact: ExactResult) -> dict:
    """Energy gaps, using the absolute gap as the primary metric.

    The ratio ``E_found / E*`` is deliberately NOT reported: QUBO energies are
    sign-indefinite, so that ratio is meaningless (spec section 32). A normalised gap is
    only reported when a well-defined reference (the mean energy over all assignments)
    gives a non-degenerate denominator.
    """
    gap = found_energy - exact.optimal_energy
    out = {
        "found_energy": found_energy,
        "optimal_energy": exact.optimal_energy,
        "absolute_gap": gap,
        "reached_optimum": bool(abs(gap) <= 1e-8),
    }
    reference = exact.energy_percentiles["mean"]
    denom = abs(reference - exact.optimal_energy)
    if denom > 1e-9:
        out["normalised_gap"] = float(gap / denom)
        out["normalised_gap_definition"] = (
            "(E_found - E*) / |E_mean_over_all_assignments - E*|"
        )
    else:
        out["normalised_gap"] = None
        out["normalised_gap_definition"] = (
            "not reported: the reference denominator is degenerate"
        )
    return out


def run_pipeline(
    config: OptimizationConfig,
    scorer: BiologicalScorer | None = None,
    include_noise: bool = True,
    include_hardware: bool = False,
    include_surrogate_validation: bool = True,
    verbose: bool = True,
) -> PipelineStageResults:
    """Execute the full pipeline and return every stage result."""
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    results = PipelineStageResults()
    results.config = config.as_dict()
    results.environment = environment_record()

    if scorer is None:
        scorer = BiologicalScorer.load(weights=ScoreWeights(config.alpha, config.beta))
    else:
        scorer = scorer.with_weights(ScoreWeights(config.alpha, config.beta))

    results.config["normalization"] = {
        "activity": scorer.activity.normalization.as_dict(),
        "hemolysis": scorer.hemolysis.normalization.as_dict(),
    }
    results.config["model_metadata"] = {
        "activity": scorer.activity.metadata,
        "hemolysis": scorer.hemolysis.metadata,
    }
    results.config["feature_config"] = scorer.activity.config.as_dict()
    results.config["score_definition"] = (
        "S(P) = alpha * A_tilde(P) - beta * H_tilde(P); LARGER IS BETTER"
    )

    log("parent analysis")
    results.parent_analysis = analyse_parent(config.parent, scorer)

    log("candidate mutations + landscape + QUBO")
    problem, landscape, timings = build_problem(config, scorer)
    results.timings.update(timings)
    results.mutation_set = problem.landscape.mutation_set.as_dict()
    results.compatibility_graph = compatibility_graph(problem.landscape.mutation_set)
    results.landscape = landscape.as_dict()
    results.qubo = problem.as_dict(include_matrix=True)
    log(f"  N={problem.n_mutation_vars} mutation vars + "
        f"{problem.n_slack_vars} slack = {problem.n_vars} qubits")

    log("verification")
    t0 = time.perf_counter()
    results.verification = run_verification(problem, landscape, scorer)
    results.timings["verification_seconds"] = time.perf_counter() - t0
    log(f"  all checks passed: {results.verification['all_passed']}")

    log("exact enumeration")
    exact = solve_exact(problem)
    results.exact = exact.as_dict()
    log(f"  E* = {exact.optimal_energy:.6f} with "
        f"{len(exact.optimal_solutions)} optimal solution(s); "
        f"{exact.n_feasible}/{exact.n_assignments} feasible")

    log("end-to-end check: is the QUBO optimum the true best candidate?")
    results.verification["qubo_optimum_vs_true_best"] = verify_qubo_optimum_is_true_best(
        problem, exact, scorer
    )
    qvt = results.verification["qubo_optimum_vs_true_best"]
    if qvt.get("status") == "evaluated":
        log(f"  QUBO optimum ranks {qvt['qubo_optimum']['true_rank']}/"
            f"{qvt['qubo_optimum']['out_of']} by true ML score "
            f"(is true best: {qvt['qubo_optimum_is_true_best']}, "
            f"expected to pass: {qvt['expected_to_pass']})")

    uniform_success = len(exact.optimal_solutions) / exact.n_assignments
    results.exact["uniform_random_success_probability"] = uniform_success
    results.exact["uniform_baseline_note"] = (
        "probability that a single uniformly random bitstring is optimal; QAOA success "
        "probabilities are reported as a multiple of this"
    )

    log("classical baselines")
    classical = run_all_classical(problem, seed=config.seed)
    results.classical = {
        name: {**res.as_dict(), **benchmark_metrics(res.best_energy, exact)}
        for name, res in classical.items()
    }
    for name, rec in results.classical.items():
        log(f"  {name:22s} gap={rec['absolute_gap']:.3e} "
            f"optimum={rec['reached_optimum']}")

    log("QAOA depth / optimizer sweep")
    qaoa_records: dict = {}
    all_bitstrings: list[np.ndarray] = []
    all_probs: list[float] = []
    for p in config.qaoa_depths:
        for opt in config.qaoa_optimizers:
            key = f"p{p}_{opt}"
            t0 = time.perf_counter()
            res = run_qaoa_ideal(
                problem,
                p=p,
                optimizer=opt,
                maxiter=config.qaoa_maxiter,
                seed=config.seed % (2**31),
                optimal_solutions=exact.optimal_solutions,
                n_restarts=config.qaoa_restarts,
                shots=config.shots,
            )
            rec = res.as_dict()
            rec.update(benchmark_metrics(res.best_sampled_energy, exact))
            rec["success_probability_vs_uniform"] = (
                res.success_probability / uniform_success if uniform_success > 0 else None
            )
            rec["wall_seconds"] = time.perf_counter() - t0
            qaoa_records[key] = rec
            log(f"  p={p} {opt:8s} F={res.final_expectation:10.4f} "
                f"gap={rec['absolute_gap']:.3e} "
                f"succ={res.success_probability:.5f} "
                f"({rec['success_probability_vs_uniform']:.1f}x uniform)")

            # harvest candidates from the best-performing configuration per depth
            top = sorted(res.distribution.items(), key=lambda kv: -kv[1])[:200]
            for bits, prob in top:
                all_bitstrings.append(bitstring_to_x(bits, problem.n_vars))
                all_probs.append(prob)

    # which (p, optimizer) pairing did best, by success probability
    best_key = max(
        qaoa_records, key=lambda k: qaoa_records[k].get("success_probability", 0.0)
    )
    results.qaoa = {
        "runs": qaoa_records,
        "best_configuration": best_key,
        "optimizer_comparison_note": (
            "both optimizers were run rather than assuming COBYLA is best "
            "(spec section 29)"
        ),
        "depth_study": [
            {
                "p": qaoa_records[k]["p"],
                "optimizer": qaoa_records[k]["optimizer"],
                "final_expectation": qaoa_records[k]["final_expectation"],
                "absolute_gap": qaoa_records[k]["absolute_gap"],
                "success_probability": qaoa_records[k]["success_probability"],
                "success_probability_vs_uniform": qaoa_records[k][
                    "success_probability_vs_uniform"
                ],
                "feasible_probability": qaoa_records[k]["feasible_probability"],
                "logical_depth": qaoa_records[k]["circuit_metrics"]["logical_depth"],
                "transpiled_depth": qaoa_records[k]["circuit_metrics"]
                .get("transpiled", {})
                .get("depth"),
                "logical_two_qubit_gates": qaoa_records[k]["circuit_metrics"][
                    "logical_two_qubit_gates"
                ],
                "transpiled_two_qubit_gates": qaoa_records[k]["circuit_metrics"]
                .get("transpiled", {})
                .get("two_qubit_gates"),
                "runtime_seconds": qaoa_records[k]["runtime_seconds"],
                "n_function_evaluations": qaoa_records[k]["n_function_evaluations"],
            }
            for k in qaoa_records
        ],
    }

    if include_noise:
        log("noise study")
        results.noise = run_noise_study(problem, qaoa_records, exact, config)
    else:
        results.noise = {"status": "not evaluated", "reason": "disabled for this run"}

    if include_hardware:
        log("hardware")
        results.hardware = run_hardware_stage(problem, qaoa_records, config)
    else:
        from backend.optimization.hardware import ibm_credentials_available

        results.hardware = {
            "status": "not evaluated",
            "reason": (
                "hardware execution not requested"
                if ibm_credentials_available()
                else "hardware execution not requested and no IBM token in environment"
            ),
        }

    if include_surrogate_validation:
        log("surrogate validation beyond pairs")
        # Two regions, reported separately. Aggregating them is misleading: the
        # optimizer can only ever return selections with at most K mutations, so error
        # measured on larger selections does not bound the error of anything the solver
        # can produce. The beyond-budget figure is still reported, as a measure of how
        # badly the quadratic model extrapolates.
        within = validate_surrogate_beyond_pairs(
            landscape, scorer,
            min_mutations=3,
            max_mutations=config.budget_k,
            n_samples=300,
            seed=config.seed % (2**31),
            budget=config.budget_k,
        )
        beyond = validate_surrogate_beyond_pairs(
            landscape, scorer,
            min_mutations=config.budget_k + 1,
            max_mutations=min(config.budget_k + 4, problem.n_mutation_vars),
            n_samples=300,
            seed=(config.seed + 1) % (2**31),
            budget=None,
        )
        # Exhaustive, per mutation count, wherever the count of feasible selections is
        # affordable. This is the headline validity measurement, so sampling error is
        # removed from it where possible.
        per_k = {}
        for k in range(3, min(config.budget_k, problem.n_mutation_vars) + 1):
            per_k[str(k)] = surrogate_fidelity_exhaustive(landscape, scorer, k)

        results.surrogate_validation = {
            "exhaustive_by_mutation_count": per_k,
            "within_budget": {
                **within,
                "region": (
                    f"3 to {config.budget_k} mutations: the feasible region the solvers "
                    "actually search"
                ),
            },
            "beyond_budget": {
                **beyond,
                "region": (
                    f"{config.budget_k + 1}+ mutations: INFEASIBLE under the budget; "
                    "reported only to show how the quadratic model extrapolates"
                ),
            },
            "exactness_regime": {
                "exact_for_n_mutations": [0, 1, 2],
                "reason": (
                    "Delta_i and Delta_ij are computed as exact finite differences of the "
                    "ML score, so the second-order surrogate reproduces the true score "
                    "identically for up to two selected mutations"
                ),
                "implication": (
                    f"with budget_mode={config.budget_mode.value} and K<=2 the QUBO is an "
                    "EXACT representation of the ML landscape over the feasible set, and "
                    "its optimum is the true best candidate under the models. For K>=3 "
                    "the surrogate omits third- and higher-order interactions and its "
                    "fidelity must be read from exhaustive_by_mutation_count."
                ),
            },
            "note": (
                "the surrogate is exact by construction for 0, 1 and 2 mutations; these "
                "figures measure the regime where it is an approximation"
            ),
        }

        # An explicit verdict, so the finding cannot be glossed over downstream.
        verdict_parts = []
        worst_rank_pct = None
        for k, rec in per_k.items():
            if rec.get("status") != "evaluated":
                continue
            tp = rec.get("surrogate_top_pick", {})
            rho = rec.get("spearman_rho")
            pct = tp.get("percentile")
            if pct is not None:
                worst_rank_pct = pct if worst_rank_pct is None else min(worst_rank_pct, pct)
            verdict_parts.append(
                f"k={k}: Spearman={rho if rho is None else round(rho, 3)}, "
                f"surrogate's top pick ranks {tp.get('true_rank')}/{tp.get('out_of')} "
                f"by true ML score"
            )
        # The verdict combines TWO distinct measurements that must not be conflated:
        #   (a) fidelity within a fixed mutation count k >= 3, and
        #   (b) whether the QUBO's overall optimum is the true best feasible candidate.
        # These can disagree. Under AT_MOST_K the optimizer may choose fewer than K
        # mutations, and selections of size <= 2 sit in the surrogate's exact regime. So
        # the end-to-end answer can be correct even when the k=3 slice is badly ranked --
        # it just is not correct *because* the surrogate is good there.
        e2e = results.verification.get("qubo_optimum_vs_true_best", {})
        e2e_ok = e2e.get("qubo_optimum_is_true_best")
        n_opt_muts = None
        if e2e.get("status") == "evaluated":
            n_opt_muts = len(e2e["qubo_optimum"]["mutations"])

        if config.budget_k <= 2:
            fidelity = (
                "EXACT: the budget keeps every feasible selection inside the regime where "
                "the surrogate reproduces the ML score identically."
            )
        elif worst_rank_pct is None:
            fidelity = "UNDETERMINED: exhaustive fidelity could not be evaluated."
        elif worst_rank_pct >= 80:
            fidelity = (
                "GOOD: within a fixed mutation count the surrogate's preferred selection "
                "is near the true best."
            )
        elif worst_rank_pct >= 60:
            fidelity = (
                "WEAK: within a fixed mutation count the surrogate's preferred selection "
                "is better than average but well short of the true best."
            )
        elif worst_rank_pct >= 40:
            fidelity = (
                "MEDIOCRE: within a fixed mutation count of 3 or more, the surrogate's "
                "preferred selection lands around the middle of the true ranking, i.e. "
                "roughly what picking at random would achieve. The quadratic model "
                "carries little usable signal at that size."
            )
        else:
            fidelity = (
                "POOR: within a fixed mutation count of 3 or more, the surrogate's "
                "preferred selection is worse than an average feasible selection. "
                "Third- and higher-order interactions, which the quadratic model omits, "
                "dominate at that size."
            )

        if e2e.get("status") != "evaluated":
            outcome = "UNDETERMINED: the end-to-end check did not run."
        elif e2e_ok and (n_opt_muts is not None and n_opt_muts <= 2):
            outcome = (
                f"CORRECT, and for an understood reason: the QUBO optimum is the true best "
                f"feasible candidate, and it selects only {n_opt_muts} mutation(s), which "
                "is inside the surrogate's exact regime. This is a property of this "
                "instance, not a guarantee -- had the true optimum required 3 or more "
                "mutations, the poor fidelity above would have been load-bearing."
            )
        elif e2e_ok:
            outcome = (
                f"CORRECT: the QUBO optimum is the true best feasible candidate, with "
                f"{n_opt_muts} mutations selected."
            )
        else:
            outcome = (
                f"INCORRECT: the QUBO optimum ranks "
                f"{e2e['qubo_optimum']['true_rank']}/{e2e['qubo_optimum']['out_of']} by "
                "true ML score. The surrogate error is load-bearing here, so the QUBO "
                "optimum must not be presented as the best candidate."
            )

        results.surrogate_validation["verdict"] = {
            "fidelity_within_fixed_mutation_count": fidelity,
            "end_to_end_outcome": outcome,
            "detail": verdict_parts,
            "banding": (
                "top-pick percentile: >=80 GOOD, >=60 WEAK, >=40 MEDIOCRE, <40 POOR. "
                "50 is what random selection within that mutation count would achieve."
            ),
            "consequence": (
                "the optimisation machinery (QUBO construction, Ising mapping, exact and "
                "QAOA solvers) is verified correct independently of this; what is measured "
                "here is whether the QUBO faithfully represents the ML landscape, which is "
                "a separate question. Because fidelity degrades at 3+ mutations, final "
                "candidate ranking is taken from direct ML re-scoring rather than from "
                "QUBO energy."
            ),
            "validity_regime": (
                "K <= 2: the QUBO is an exact representation of the ML landscape. "
                "K >= 3: the QUBO is a local approximation whose within-size ranking is "
                "unreliable for these models."
            ),
        }
        if within.get("status") == "evaluated":
            log(f"  within budget (<={config.budget_k}): MAE={within['mae']:.4f} "
                f"RMSE={within['rmse']:.4f} nRMSE={within.get('normalised_rmse')} "
                f"rho={within.get('spearman_rho')}")
        else:
            log(f"  within budget: {within.get('reason')}")
        if beyond.get("status") == "evaluated":
            log(f"  beyond budget: MAE={beyond['mae']:.4f} RMSE={beyond['rmse']:.4f}")

    log("independent ML re-scoring")
    # candidates from exact, classical and QAOA, all re-scored with the real models
    candidate_bits: list[np.ndarray] = []
    sources: list[str] = []
    probs: list[float | None] = []
    for x in exact.optimal_feasible_solutions[:20]:
        candidate_bits.append(np.asarray(x))
        sources.append("exact")
        probs.append(None)
    for name, res in classical.items():
        candidate_bits.append(np.asarray(res.best_x))
        sources.append(f"classical:{name}")
        probs.append(None)

    cands: list[Candidate] = []
    for x, src in zip(candidate_bits, sources):
        cands.extend(rescore_candidates(problem, [x], scorer, src))
    qaoa_cands = rescore_candidates(
        problem, all_bitstrings, scorer, "qaoa", probabilities=all_probs
    )
    cands.extend(qaoa_cands)

    # deduplicate by sequence, preferring the record with a probability
    by_seq: dict[str, Candidate] = {}
    for c in cands:
        prev = by_seq.get(c.sequence)
        if prev is None:
            by_seq[c.sequence] = c
        else:
            if prev.probability is None and c.probability is not None:
                c.source = f"{prev.source}+{c.source}"
                by_seq[c.sequence] = c
            elif prev.source != c.source and c.source not in prev.source:
                prev.source = f"{prev.source}+{c.source}"
    final_cands = sorted(by_seq.values(), key=lambda c: -c.direct_delta_score)

    # Trustworthiness diagnostics. The models are fitted on natural peptides but applied
    # to point mutants, so each candidate carries how far it sits from anything in the
    # training set and how much the bootstrap ensemble disagrees about it. See
    # results/experiments/mutation_extrapolation_report.json for the measured accuracy of
    # exactly this kind of extrapolation on real held-out mutant pairs.
    log("trustworthiness diagnostics")
    trust = attach_trustworthiness(final_cands)
    results.candidates = [c.as_dict() for c in final_cands]

    results.surrogate_validation = {
        **results.surrogate_validation,
        "on_returned_candidates": surrogate_error_summary(final_cands),
    }

    results.trustworthiness = trust
    for target in ("activity", "hemolysis"):
        rec = trust.get(f"{target}_nearest_identity", {})
        if rec.get("status") == "evaluated":
            log(f"  {target}: nearest-training identity median "
                f"{rec['median']:.3f} (min {rec['min']:.3f})")

    log("Pareto analysis")
    results.pareto = pareto_analysis(
        final_cands,
        {
            "activity_norm": results.parent_analysis["activity_normalized"],
            "hemolysis_norm": results.parent_analysis["hemolysis_normalized"],
        },
    )

    improved = [c for c in final_cands if c.direct_delta_score > 0]
    results.pareto["summary"] = {
        "n_candidates": len(final_cands),
        "n_improving_on_parent_scalar_score": len(improved),
        "best_direct_delta_score": (
            final_cands[0].direct_delta_score if final_cands else None
        ),
        "best_candidate_sequence": final_cands[0].sequence if final_cands else None,
        "best_candidate_mutations": final_cands[0].mutations if final_cands else None,
        "language_note": (
            "all values are MODEL-PREDICTED; no experimental activity or hemolysis is "
            "claimed. Experimental validation would be required."
        ),
    }
    log(f"  {len(final_cands)} unique candidates; "
        f"{len(improved)} improve the parent's predicted score")
    return results


def run_noise_study(
    problem: QuboProblem, qaoa_records: dict, exact: ExactResult, config: OptimizationConfig
) -> dict:
    """Sample the best QAOA parameters under several documented noise models."""
    from backend.optimization.hardware import (
        build_depolarizing_noise,
        build_thermal_noise,
        noise_sweep_specs,
    )

    best_key = max(
        qaoa_records, key=lambda k: qaoa_records[k].get("success_probability", 0.0)
    )
    params = np.array(qaoa_records[best_key]["final_params"], dtype=float)
    p = qaoa_records[best_key]["p"]
    uniform = len(exact.optimal_solutions) / exact.n_assignments

    runs: dict = {}
    for label, two_q in noise_sweep_specs():
        if two_q == 0.0:
            model, spec = None, None
        else:
            model, spec = build_depolarizing_noise(two_qubit_error=two_q, name=label)
        out = sample_distribution(
            problem, p, params, shots=config.shots, noise_model=model,
            seed=config.seed % (2**31),
        )
        runs[label] = {
            **out,
            **benchmark_metrics(out["best_sampled_energy"], exact),
            "noise_spec": spec.as_dict() if spec else {"name": "noiseless"},
            "success_probability_vs_uniform": (
                out["success_probability"] / uniform if uniform > 0 else None
            ),
        }
        runs[label].pop("distribution", None)  # keep the record compact

    # What is DONE about the noise, not just how bad it is. Readout mitigation is
    # applied to the representative noise level and the improvement reported either way.
    try:
        from backend.optimization.ising import diagonal_energies, qubo_to_ising
        from backend.optimization.mitigation import evaluate_mitigation

        hamiltonian, _ = qubo_to_ising(problem)
        diag = diagonal_energies(hamiltonian, problem.n_vars)
        rep_model, _ = build_depolarizing_noise(two_qubit_error=1.0e-2)
        mitigation = evaluate_mitigation(
            problem, p, params, rep_model, diag,
            shots=config.shots, seed=config.seed % (2**31),
        )
    except Exception as exc:
        mitigation = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    thermal_model, thermal_spec = build_thermal_noise()
    out = sample_distribution(
        problem, p, params, shots=config.shots, noise_model=thermal_model,
        seed=config.seed % (2**31),
    )
    runs["thermal_relaxation"] = {
        **out,
        **benchmark_metrics(out["best_sampled_energy"], exact),
        "noise_spec": thermal_spec.as_dict(),
        "success_probability_vs_uniform": (
            out["success_probability"] / uniform if uniform > 0 else None
        ),
    }
    runs["thermal_relaxation"].pop("distribution", None)

    return {
        "status": "evaluated",
        "parameters_from": best_key,
        "p": p,
        "shots": config.shots,
        "runs": runs,
        "mitigation": mitigation,
        "note": (
            "noise models are simulated with Qiskit Aer using the documented rates; they "
            "are not measurements from a real device"
        ),
    }


def run_hardware_stage(
    problem: QuboProblem, qaoa_records: dict, config: OptimizationConfig
) -> dict:
    """Optional real-QPU execution with the best QAOA parameters."""
    from backend.optimization.hardware import run_on_hardware

    best_key = max(
        qaoa_records, key=lambda k: qaoa_records[k].get("success_probability", 0.0)
    )
    params = np.array(qaoa_records[best_key]["final_params"], dtype=float)
    p = qaoa_records[best_key]["p"]
    record = run_on_hardware(problem, p, params, shots=min(config.shots, 4096))
    record["parameters_from"] = best_key
    return record


def save_results(results: PipelineStageResults, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(results.as_dict(), indent=2, default=_json_default))
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
