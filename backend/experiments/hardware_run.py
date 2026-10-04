"""Execute one QAOA circuit on real IBM Quantum hardware, and compare honestly.

Answers Research Question 7: how does real hardware compare with ideal and noisy
simulation?

Deliberately sized small. The headline configuration (N=14, p=3) transpiles to roughly 720
two-qubit gates and would return something indistinguishable from uniform noise, which
measures nothing. N=6 at p=1 is about 56 two-qubit gates — small enough that a real device
has a chance of showing structure, which is what makes the comparison informative.

Parameters are optimised on the simulator first; hardware is used only to SAMPLE the final
circuit. That is one job, not hundreds.

    ./.venv/bin/python -m backend.experiments.hardware_run
    ./.venv/bin/python -m backend.experiments.hardware_run --target-n 8 --p 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.models.property_models import BiologicalScorer, ScoreWeights
from backend.optimization.hardware import (
    build_depolarizing_noise,
    get_ibm_service,
    ibm_credentials_available,
    run_on_hardware,
    select_backend,
)
from backend.optimization.ising import diagonal_energies, qubo_to_ising
from backend.optimization.landscape import compute_landscape
from backend.optimization.mutations import generate_mutation_set
from backend.optimization.pipeline import (
    DEFAULT_PARENT,
    DEFAULT_SEED,
    _json_default,
    benchmark_metrics,
    environment_record,
)
from backend.optimization.qaoa import run_qaoa_ideal, sample_distribution
from backend.optimization.qubo import BudgetMode, build_qubo
from backend.optimization.solvers import run_all_classical, solve_exact

from backend.utils.env import load_dotenv

load_dotenv()  # credentials live in .env, not the shell profile

RESULTS_DIR = Path("results/experiments")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-n", type=int, default=6,
                    help="candidate mutations (6 -> 8 qubits, ~56 two-qubit gates)")
    ap.add_argument("--p", type=int, default=1, help="QAOA depth")
    ap.add_argument("--budget-k", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--backend", type=str, default=None,
                    help="pin a backend instead of choosing the least busy")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and cost the circuit, but do not submit")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scorer = BiologicalScorer.load(weights=ScoreWeights(1.0, 1.0))

    print("hardware run — building the problem")
    mset = generate_mutation_set(DEFAULT_PARENT, scorer, target_n=args.target_n,
                                 max_per_position=2)
    landscape = compute_landscape(mset, scorer)
    problem = build_qubo(landscape, budget_k=min(args.budget_k, mset.n),
                         budget_mode=BudgetMode.AT_MOST_K)
    exact = solve_exact(problem)
    hamiltonian, _ = qubo_to_ising(problem)
    diag = diagonal_energies(hamiltonian, problem.n_vars)
    uniform = len(exact.optimal_solutions) / exact.n_assignments

    print(f"  qubits                 : {problem.n_vars} "
          f"({problem.n_mutation_vars} mutation + {problem.n_slack_vars} slack)")
    print(f"  exact optimum E*       : {exact.optimal_energy:.6f} "
          f"({len(exact.optimal_solutions)} optimal)")
    print(f"  uniform random success : {uniform:.3e}")

    print("\noptimising QAOA parameters on the simulator")
    ideal = run_qaoa_ideal(problem, p=args.p, optimizer="COBYLA", maxiter=400,
                           seed=DEFAULT_SEED % (2**31),
                           optimal_solutions=exact.optimal_solutions,
                           n_restarts=4, shots=args.shots)
    params = np.array(ideal.final_params)
    cm = ideal.circuit_metrics
    print(f"  ideal success          : {ideal.success_probability:.5f} "
          f"({ideal.success_probability / uniform:.1f}x uniform)")
    print(f"  logical depth / 2q     : {cm['logical_depth']} / "
          f"{cm['logical_two_qubit_gates']}")
    print(f"  transpiled depth / 2q  : {cm['transpiled']['depth']} / "
          f"{cm['transpiled']['two_qubit_gates']}  (abstract basis)")

    print("\nnoisy simulation at representative hardware error rates")
    noise_model, noise_spec = build_depolarizing_noise(two_qubit_error=1.0e-2)
    noisy = sample_distribution(problem, args.p, params, shots=args.shots,
                                noise_model=noise_model, seed=DEFAULT_SEED % (2**31))
    print(f"  noisy success          : {noisy['success_probability']:.5f} "
          f"({noisy['success_probability'] / uniform:.1f}x uniform)")
    print(f"  noisy feasible         : {noisy['feasible_probability']:.4f}")

    record: dict = {
        "experiment": "hardware_run",
        "config": {"target_n": args.target_n, "p": args.p, "budget_k": problem.budget_k,
                   "shots": args.shots, "seed": DEFAULT_SEED},
        "n_qubits": problem.n_vars,
        "exact": {"optimal_energy": exact.optimal_energy,
                  "n_optimal_solutions": len(exact.optimal_solutions),
                  "n_feasible": exact.n_feasible,
                  "n_assignments": exact.n_assignments,
                  "uniform_random_success_probability": uniform},
        "classical": {
            name: {"best_energy": r.best_energy, **benchmark_metrics(r.best_energy, exact)}
            for name, r in run_all_classical(problem, seed=DEFAULT_SEED).items()
        },
        "ideal_simulation": {
            **ideal.as_dict(max_distribution=32),
            **benchmark_metrics(ideal.best_sampled_energy, exact),
            "success_probability_vs_uniform": ideal.success_probability / uniform,
        },
        "noisy_simulation": {
            **{k: v for k, v in noisy.items() if k != "distribution"},
            **benchmark_metrics(noisy["best_sampled_energy"], exact),
            "noise_spec": noise_spec.as_dict(),
            "success_probability_vs_uniform": noisy["success_probability"] / uniform,
        },
        "environment": environment_record(),
    }

    # --- hardware -----------------------------------------------------------
    if args.dry_run:
        record["hardware"] = {"status": "not evaluated", "reason": "--dry-run requested"}
        print("\n--dry-run: not submitting.")
    elif not ibm_credentials_available():
        record["hardware"] = {
            "status": "not evaluated",
            "reason": "no IBM Quantum API key in the environment",
        }
        print("\nNo IBM credentials found — run `python -m backend.check_ibm` for help.")
    else:
        print("\nconnecting to IBM Quantum")
        service, diag_info = get_ibm_service(explain=True)
        if service is None:
            record["hardware"] = {"status": "failed", "diagnostic": diag_info}
            print(f"  connection failed: {diag_info.get('error')}")
            if diag_info.get("hint"):
                print(f"  {diag_info['hint']}")
        else:
            backend = select_backend(service, problem.n_vars, args.backend)
            if backend is None:
                record["hardware"] = {
                    "status": "not evaluated",
                    "reason": f"no operational backend with >= {problem.n_vars} qubits",
                }
                print("  no suitable backend available")
            else:
                name = backend.name if isinstance(backend.name, str) else backend.name()
                print(f"  backend: {name} — submitting {args.shots} shots")
                print("  (this queues; it may take minutes to hours)")
                hw = run_on_hardware(problem, args.p, params, shots=args.shots,
                                     backend_name=name)
                if hw.get("status") == "executed":
                    hw.update(benchmark_metrics(hw["best_sampled_energy"], exact))
                    hw["success_probability_vs_uniform"] = (
                        hw["success_probability"] / uniform if uniform else None
                    )
                record["hardware"] = hw

    path = RESULTS_DIR / "hardware_run.json"
    path.write_text(json.dumps(record, indent=2, default=_json_default))

    # --- the comparison that answers RQ7 ------------------------------------
    print("\n" + "=" * 74)
    print("RQ7 — ideal vs noisy vs real hardware")
    print("=" * 74)
    print(f"{'source':<26}{'success':>11}{'x uniform':>12}{'feasible':>11}{'best gap':>12}")
    print("-" * 74)
    print(f"{'uniform random':<26}{uniform:>11.5f}{1.0:>12.1f}{'—':>11}{'—':>12}")
    print(f"{'ideal simulation':<26}{ideal.success_probability:>11.5f}"
          f"{ideal.success_probability / uniform:>12.1f}"
          f"{ideal.feasible_probability:>11.4f}"
          f"{record['ideal_simulation']['absolute_gap']:>12.2e}")
    print(f"{'noisy simulation (1e-2)':<26}{noisy['success_probability']:>11.5f}"
          f"{noisy['success_probability'] / uniform:>12.1f}"
          f"{noisy['feasible_probability']:>11.4f}"
          f"{record['noisy_simulation']['absolute_gap']:>12.2e}")

    hw = record.get("hardware", {})
    if hw.get("status") == "executed":
        print(f"{'REAL HARDWARE':<26}{hw['success_probability']:>11.5f}"
              f"{hw.get('success_probability_vs_uniform', 0):>12.1f}"
              f"{hw['feasible_probability']:>11.4f}"
              f"{hw.get('absolute_gap', float('nan')):>12.2e}")
        print(f"\n  backend {hw['backend']} · job {hw['job_id']}")
        print(f"  transpiled depth {hw['transpiled_depth']} · "
              f"{hw['transpiled_two_qubit_gates']} two-qubit gates")
        mit = hw.get("mitigation", {}).get("readout_mitigation", {})
        if mit.get("status") == "evaluated":
            print(f"  readout-mitigated success: {mit['success_probability']:.5f} "
                  f"(raw {hw['success_probability']:.5f})")
        opts = hw.get("mitigation", {}).get("runtime_options", {})
        print(f"  dynamical decoupling: {opts.get('dynamical_decoupling')} · "
              f"twirling: {opts.get('twirling')}")
    else:
        print(f"{'REAL HARDWARE':<26}{'not evaluated':>11}")
        print(f"\n  reason: {hw.get('reason') or hw.get('error') or hw.get('diagnostic')}")

    print(f"\nreport -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
