"""The complete scientific experiment (spec section 57), and the answers it supports.

    parent AMP -> candidate mutations -> ML mutation landscape
               -> individual effects -> pairwise effects -> QUBO
               -> exact -> classical -> QAOA p=1,2,3 -> noise -> optional QPU
               -> independent ML re-scoring -> Pareto analysis

Two budgets are run, because the measured validity regime of the second-order surrogate
is K <= 2:

    K = 2   the QUBO is an EXACT representation of the ML landscape over the feasible
            set, so the optimisation result is meaningful without caveat
    K = 3   the QUBO becomes an approximation; fidelity is measured rather than assumed

Run:  python -m backend.experiments.final
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.models.property_models import BiologicalScorer, ScoreWeights
from backend.optimization.hardware import ibm_credentials_available
from backend.optimization.pipeline import (
    DEFAULT_PARENT,
    OptimizationConfig,
    _json_default,
    run_pipeline,
    save_results,
)
from backend.optimization.qubo import BudgetMode

from backend.utils.env import load_dotenv

load_dotenv()  # credentials live in .env, not the shell profile

RESULTS_DIR = Path("results/experiments")


def research_answers(runs: dict[str, dict]) -> dict:
    """Answer the seven research questions from what was actually measured."""
    k2 = runs.get("final_K2", {})
    k3 = runs.get("final_K3", {})

    def pareto_n(r: dict) -> int | None:
        return r.get("pareto", {}).get("n_candidates_dominating_parent")

    def best_delta(r: dict) -> float | None:
        return r.get("pareto", {}).get("summary", {}).get("best_direct_delta_score")

    def qaoa_rows(r: dict) -> list[dict]:
        return r.get("qaoa", {}).get("depth_study", [])

    def by_depth(r: dict, p: int) -> dict | None:
        rows = [x for x in qaoa_rows(r) if x.get("p") == p]
        if not rows:
            return None
        return max(rows, key=lambda x: x.get("success_probability", 0.0))

    answers: dict = {}

    # Q1 -- biological optimization
    answers["Q1_biological_optimization"] = {
        "question": (
            "Does the optimized candidate have a better model-predicted "
            "activity/hemolysis trade-off than the parent?"
        ),
        "K2": {
            "best_delta_score_vs_parent": best_delta(k2),
            "n_candidates_dominating_parent": pareto_n(k2),
            "best_candidate": k2.get("pareto", {}).get("summary", {}).get("best_candidate_sequence"),
            "best_mutations": k2.get("pareto", {}).get("summary", {}).get("best_candidate_mutations"),
        },
        "K3": {
            "best_delta_score_vs_parent": best_delta(k3),
            "n_candidates_dominating_parent": pareto_n(k3),
            "best_candidate": k3.get("pareto", {}).get("summary", {}).get("best_candidate_sequence"),
            "best_mutations": k3.get("pareto", {}).get("summary", {}).get("best_candidate_mutations"),
        },
        "caveat": (
            "'better' here means a higher MODEL-PREDICTED score. The hemolysis model is "
            "the weaker of the two (see the ML report), so this answer is only as good as "
            "that model. No experimental claim is made."
        ),
    }

    # Q2 -- QUBO validity
    answers["Q2_qubo_validity"] = {
        "question": (
            "Does the second-order QUBO reproduce the local ML mutation landscape "
            "sufficiently well?"
        ),
        "exact_regime": "0, 1 and 2 mutations: exact by construction",
        "K2_verdict": k2.get("surrogate_validation", {}).get("verdict"),
        "K3_verdict": k3.get("surrogate_validation", {}).get("verdict"),
        "K3_exhaustive_by_count": k3.get("surrogate_validation", {}).get(
            "exhaustive_by_mutation_count"
        ),
        "end_to_end_K2": k2.get("verification", {}).get("qubo_optimum_vs_true_best"),
        "end_to_end_K3": k3.get("verification", {}).get("qubo_optimum_vs_true_best"),
    }

    # Q3 -- how close QAOA gets
    answers["Q3_qaoa_vs_exact"] = {
        "question": "How close does QAOA get to the exact optimum on small instances?",
        "metric_note": (
            "absolute energy gap is primary; success probability is the sum over ALL "
            "degenerate optima and is reported as a multiple of uniform random sampling"
        ),
        "K2": {
            "uniform_random_success": k2.get("exact", {}).get("uniform_random_success_probability"),
            "by_depth": {f"p{p}": by_depth(k2, p) for p in (1, 2, 3)},
        },
        "K3": {
            "uniform_random_success": k3.get("exact", {}).get("uniform_random_success_probability"),
            "by_depth": {f"p{p}": by_depth(k3, p) for p in (1, 2, 3)},
        },
    }

    # Q4 -- classical comparison
    answers["Q4_classical_comparison"] = {
        "question": "How does QAOA compare with classical optimization on identical QUBOs?",
        "note": (
            "all solvers receive the same QUBO and no biological information. QAOA runtimes "
            "are statevector-simulation cost, not quantum execution cost, so they are not "
            "a speed comparison."
        ),
        "K2_classical": {
            name: {k: v.get(k) for k in
                   ("best_energy", "absolute_gap", "reached_optimum", "runtime_seconds",
                    "n_evaluations", "mean_energy", "std_energy")}
            for name, v in k2.get("classical", {}).items()
        },
        "K3_classical": {
            name: {k: v.get(k) for k in
                   ("best_energy", "absolute_gap", "reached_optimum", "runtime_seconds",
                    "n_evaluations", "mean_energy", "std_energy")}
            for name, v in k3.get("classical", {}).items()
        },
    }

    # Q5 -- depth
    answers["Q5_depth"] = {
        "question": "How does increasing QAOA depth affect solution quality and circuit cost?",
        "K3_depth_study": qaoa_rows(k3),
    }

    # Q6 -- noise
    answers["Q6_noise"] = {
        "question": "How much does realistic noise degrade the solution?",
        "K3_noise": {
            name: {k: v.get(k) for k in
                   ("expectation", "best_sampled_energy", "absolute_gap",
                    "success_probability", "feasible_probability", "n_distinct_outcomes")}
            for name, v in k3.get("noise", {}).get("runs", {}).items()
        } if k3.get("noise", {}).get("status") == "evaluated" else {
            "status": "not evaluated"
        },
    }

    # Q7 -- hardware
    hw = k3.get("hardware", {})
    answers["Q7_hardware"] = {
        "question": (
            "How does real hardware compare with ideal and noisy simulation, if hardware "
            "execution is available?"
        ),
        "status": hw.get("status", "not evaluated"),
        "detail": hw if hw.get("status") == "executed" else {
            "reason": hw.get("reason") or hw.get("error") or "not requested"
        },
    }

    answers["claims_not_made"] = [
        "No quantum advantage is claimed. The instances are tiny, QAOA runs on a "
        "simulator, and the closest comparable study found QAOA matched by random "
        "sampling on a peptide problem.",
        "No novelty is claimed for quantum optimization of nonhemolytic AMPs; that was "
        "done and wet-lab validated in 2023 (see research/papers.md section 1.1).",
        "No experimental biological efficacy is claimed for any sequence produced here.",
    ]
    return answers


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scorer = BiologicalScorer.load(weights=ScoreWeights(1.0, 1.0))
    include_hw = ibm_credentials_available()
    if include_hw:
        print("IBM Quantum token detected: hardware execution will be attempted.\n")
    else:
        print("No IBM Quantum token: hardware stages will be recorded as "
              "'not evaluated'.\n")

    runs: dict[str, dict] = {}
    for k in (2, 3):
        name = f"final_K{k}"
        print(f"=== {name} (budget K={k}, at most K) ===")
        cfg = OptimizationConfig(
            parent=DEFAULT_PARENT,
            budget_k=k,
            budget_mode=BudgetMode.AT_MOST_K,
            alpha=1.0,
            beta=1.0,
            target_n=14,
            max_per_position=2,
            qaoa_depths=(1, 2, 3),
            qaoa_optimizers=("COBYLA", "Powell"),
            qaoa_maxiter=300,
            qaoa_restarts=3,
            shots=8192,
        )
        res = run_pipeline(
            cfg,
            scorer=scorer,
            include_noise=True,
            include_hardware=include_hw,
            include_surrogate_validation=True,
            verbose=True,
        )
        path = save_results(res, name)
        runs[name] = res.as_dict()
        print(f"  -> {path}\n")

    answers = research_answers(runs)
    out = RESULTS_DIR / "research_answers.json"
    out.write_text(json.dumps(answers, indent=2, default=_json_default))
    print(f"research answers -> {out}")

    # concise console summary
    print("\n" + "=" * 78)
    print("RESEARCH QUESTION SUMMARY")
    print("=" * 78)
    for key in ("Q1_biological_optimization", "Q2_qubo_validity", "Q3_qaoa_vs_exact"):
        print(f"\n{key}")
        print(f"  {answers[key]['question']}")
    q1 = answers["Q1_biological_optimization"]
    for kk in ("K2", "K3"):
        print(f"\n  {kk}: best ΔS = {q1[kk]['best_delta_score_vs_parent']}, "
              f"{q1[kk]['n_candidates_dominating_parent']} candidates dominate the parent")
        print(f"       {q1[kk]['best_candidate']}  {q1[kk]['best_mutations']}")
    q2 = answers["Q2_qubo_validity"]
    for kk, v in (("K2", q2["K2_verdict"]), ("K3", q2["K3_verdict"])):
        if v:
            print(f"\n  {kk} fidelity : {v['fidelity_within_fixed_mutation_count'][:110]}")
            print(f"  {kk} end-to-end: {v['end_to_end_outcome'][:110]}")
    q3 = answers["Q3_qaoa_vs_exact"]
    for kk in ("K2", "K3"):
        print(f"\n  {kk} QAOA (uniform baseline "
              f"{q3[kk]['uniform_random_success']:.2e}):")
        for p in (1, 2, 3):
            d = q3[kk]["by_depth"].get(f"p{p}")
            if d:
                print(f"    p={p} {d['optimizer']:8s} gap={d['absolute_gap']:.2e} "
                      f"succ={d['success_probability']:.5f} "
                      f"({d['success_probability_vs_uniform']:.1f}x uniform)")


if __name__ == "__main__":
    main()
