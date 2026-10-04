"""Builds the demonstration notebook, then executes it so outputs are embedded.

Kept as a script rather than a hand-edited .ipynb so the notebook is reproducible: the
narrative and the code live here in reviewable form, and `python notebooks/build_notebook.py`
regenerates the executed notebook from scratch.

Run from the repository root:
    ./.venv/bin/python notebooks/build_notebook.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks/Q_Peptide_Demo.ipynb")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


cells: list[nbf.NotebookNode] = []

cells.append(md(r"""
# Q-Peptide — quantum-assisted antimicrobial peptide design

**A constrained quadratic binary optimization problem, solved with QAOA and verified
against brute force.**

Given a known antimicrobial peptide, which two or three amino-acid substitutions most
improve the predicted activity-versus-hemolysis trade-off?

With ~14 candidate substitutions there are thousands of combinations, the substitutions
**interact**, and some are **mutually exclusive** (two residues cannot occupy one
position). That is a QUBO:

$$\max_{x}\;\; \widehat{\Delta S}(x) = \sum_i \Delta_i x_i + \sum_{i<j} \Delta_{ij} x_i x_j
\quad\text{s.t.}\quad \sum_i x_i \le K,\;\; x_i + x_j \le 1,\;\; x_i \in \{0,1\}$$

> The quantum computer is **not** simulating protein folding. It solves the
> subset-selection problem above.

> ⚠️ **Every biological value in this notebook is a model prediction.** Nothing has been
> synthesised or assayed.

---

### What makes this formulation different

The coefficients $\Delta_i$ and $\Delta_{ij}$ are **computed by running the ML models on
the actual mutant sequences** — not fitted to sampled scores. That removes a layer of
surrogate error and, critically, makes the error that remains *measurable*.
"""))

cells.append(code("""
import sys, os, json, time
sys.path.insert(0, os.path.abspath(".."))
os.chdir(os.path.abspath(".."))   # run from the repository root

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

from backend.models.property_models import BiologicalScorer, ScoreWeights

SEED = 20261004
PARENT = "GIGKFLHSAKKFGKAFVGEIMNS"   # Magainin 2
print("parent:", PARENT, f"({len(PARENT)} residues)")
"""))

cells.append(md("""
## 1. The parent peptide and the two property models

Two independently trained XGBoost regressors, both on the pharmacological log-dose scale
`6 − log₁₀(c[µM])`:

- **activity** — higher is better (more potent)
- **hemolysis** — higher is **worse** (lyses red blood cells at a lower dose)

Combined into one score, with the sign convention fixed in exactly one place:

$$S(P) = \\alpha\\,\\tilde A(P) - \\beta\\,\\tilde H(P) \\qquad \\textbf{larger is better}$$
"""))

cells.append(code("""
scorer = BiologicalScorer.load(weights=ScoreWeights(alpha=1.0, beta=1.0))
b = scorer.breakdown([PARENT])

print(f"predicted activity   {b.activity_raw[0]:7.3f}   (pMIC scale, higher = more active)")
print(f"predicted hemolysis  {b.hemolysis_raw[0]:7.3f}   (p-dose scale, higher = MORE hemolytic)")
print(f"score S(P)           {b.score[0]:7.4f}")
print()
print("activity model :", scorer.activity.metadata["target_definition"])
print("  held-out MAE :", round(scorer.activity.metadata["heldout_test_mae"], 4),
      " Spearman:", round(scorer.activity.metadata["heldout_test_spearman"], 4))
print("hemolysis model:", scorer.hemolysis.metadata["target_definition"])
print("  held-out MAE :", round(scorer.hemolysis.metadata["heldout_test_mae"], 4),
      " Spearman:", round(scorer.hemolysis.metadata["heldout_test_spearman"], 4))
"""))

cells.append(md("""
## 2. Candidate mutations and the ML mutation landscape

Candidates pass three filters before any optimisation:

1. **structural** — never mutate C/G/P (disulfide, turn flexibility, helix breaking);
   never introduce C/P/M/W (unpaired cysteine, helix breaking, oxidation-prone)
2. **physicochemical** — the substitution must actually move charge or hydrophobicity,
   the two axes the magainin literature identifies as controlling the trade-off
3. **ML pre-screening** — every surviving single mutant is scored with the real models

Then the landscape: `1 + N + C(N,2)` **real model evaluations**.
"""))

cells.append(code("""
from backend.optimization.mutations import generate_mutation_set, compatibility_graph
from backend.optimization.landscape import compute_landscape

mset = generate_mutation_set(PARENT, scorer, target_n=14, max_per_position=2)
landscape = compute_landscape(mset, scorer)

print(f"admissible after filters : {mset.generation_report['n_admissible_after_filters']}")
print(f"selected variables N     : {mset.n}")
print(f"same-position conflicts  : {len(mset.conflict_pairs)}")
print(f"real model evaluations   : {landscape.n_model_evaluations}")
print()
print("variables:", [c.label for c in mset.candidates])
"""))

cells.append(code("""
r = landscape.report
print(f"Delta_i   : {r['delta_i']['n_beneficial']} beneficial, {r['delta_i']['n_deleterious']} deleterious")
print(f"            range [{r['delta_i']['min']:.4f}, {r['delta_i']['max']:.4f}]")
print(f"Delta_ij  : {r['delta_ij']['n_synergistic']} synergistic, "
      f"{r['delta_ij']['n_antagonistic']} antagonistic, {r['delta_ij']['n_additive']} additive")
print()
print(f"interaction strength  mean|Delta_ij| / mean|Delta_i|  =  "
      f"{r['interaction_strength_ratio']:.4f}")
print()
print("A ratio near zero would mean the landscape is additive and the problem collapses to")
print("independent ranking. It does not: the quadratic term carries real signal, which is")
print("what makes this a genuine QUBO rather than a sorting problem.")
"""))

cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})

labels = [c.label for c in mset.candidates]
deltas = landscape.delta
axes[0].bar(range(mset.n), deltas,
            color=["#009E73" if d >= 0 else "#D55E00" for d in deltas])
axes[0].axhline(0, color="#444", lw=0.8)
axes[0].set_xticks(range(mset.n)); axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
axes[0].set_ylabel(r"$\\Delta_i = S(P \\oplus m_i) - S(P)$")
axes[0].set_title("Individual mutation effects")

M = landscape.delta_pair.copy()
vmax = np.nanmax(np.abs(M))
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#dddddd")
im = axes[1].imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax)
axes[1].set_xticks(range(mset.n)); axes[1].set_yticks(range(mset.n))
axes[1].set_xticklabels(labels, rotation=90, fontsize=6)
axes[1].set_yticklabels(labels, fontsize=6)
axes[1].set_title(r"Pairwise interactions $\\Delta_{ij}$" + "\\n(grey = same-position conflict)")
axes[1].grid(False)
fig.colorbar(im, ax=axes[1], fraction=0.046)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 3. Building the QUBO — and verifying it

Three things are easy to get silently wrong here, so each is **derived, then tested**:

| trap | what goes wrong | guard |
|---|---|---|
| matrix convention | `Qᵢⱼ = cᵢⱼ/2` off-diagonal; forget the ½ and every quadratic term doubles | matrix evaluated against an independent polynomial evaluator |
| constraint type | `P(Σx−K)²` enforces **equality**, not `≤ K` | two separate modes; slack encoding for the inequality |
| penalty size | too small and violating a constraint pays for itself | derived from the exact objective span, then verified by enumeration |
"""))

cells.append(code("""
from backend.optimization.qubo import (
    BudgetMode, build_qubo, verify_matrix_matches_polynomial,
    verify_penalty_sufficiency, verify_slack_range,
)

K = 3
problem = build_qubo(landscape, budget_k=K, budget_mode=BudgetMode.AT_MOST_K)

print(f"mutation variables : {problem.n_mutation_vars}")
print(f"slack variables    : {problem.n_slack_vars}  weights {problem.build_report['slack_weights']}")
print(f"total qubits       : {problem.n_vars}")
print(f"budget mode        : {problem.budget_mode.description}")
print()
print(f"objective span     : {problem.build_report['objective_span']['span']:.4f}   (exact, by enumeration)")
print(f"triangle bound B   : {problem.objective_bound:.4f}   (valid but ~2.4x looser)")
print(f"penalty P          : {problem.penalty_budget:.4f}")
"""))

cells.append(code("""
mp = verify_matrix_matches_polynomial(problem, n_samples=2000, seed=0)
ps = verify_penalty_sufficiency(problem)
sr = verify_slack_range(K)

print("VERIFICATION (executed, not asserted)")
print(f"  matrix == polynomial  : {mp['passed']}   max deviation {mp['max_abs_deviation']:.2e} "
      f"over {mp['n_samples']} bitstrings")
print(f"  slack range == 0..{K}  : {sr['passed']}   representable {sr['representable']}")
print(f"  penalties sufficient  : {ps['passed']}   margin {ps['margin']:.4f}")
print(f"     ({ps['n_feasible']:,} feasible of {ps['n_assignments']:,} assignments)")
print()
print("The penalty margin is the gap between the best INFEASIBLE energy and the best")
print("feasible one. Positive means no constraint violation can ever pay for itself.")
"""))

cells.append(md("""
## 4. QUBO → Ising, checked on every basis state

Substituting $x_i = (1 - Z_i)/2$ gives a diagonal Hamiltonian. Because it is diagonal,
correctness is not an argument — it is **exhaustively checkable**: every one of the $2^n$
basis states must satisfy $\\langle x|H_C|x\\rangle = E_{\\mathrm{QUBO}}(x)$.
"""))

cells.append(code("""
from backend.optimization.ising import qubo_to_ising, verify_ising_mapping

H, ising_report = qubo_to_ising(problem)
iv = verify_ising_mapping(problem, H)

print(f"qubits              : {ising_report['n_qubits']}")
print(f"Pauli terms         : {ising_report['n_terms']}  "
      f"({ising_report['n_single_z_terms']} single-Z, {ising_report['n_zz_terms']} ZZ)")
print(f"identity offset     : {ising_report['identity_offset']:.4f}")
print()
print(f"verification mode   : {iv['mode']}")
print(f"states checked      : {iv['n_states_checked']:,}")
print(f"max deviation       : {iv['max_abs_deviation']:.2e}")
print(f"PASSED              : {iv['passed']}")
"""))

cells.append(md("""
## 5. Ground truth: exact enumeration

At this size every assignment can be enumerated, so the true optimum is known. **This is
the point of keeping N small** — without the exact optimum there is no optimality gap, and
the benchmark would be unfalsifiable.
"""))

cells.append(code("""
from backend.optimization.solvers import solve_exact, run_all_classical

exact = solve_exact(problem)
uniform = len(exact.optimal_solutions) / exact.n_assignments

print(f"assignments enumerated : {exact.n_assignments:,}")
print(f"feasible               : {exact.n_feasible:,}")
print(f"optimal energy E*      : {exact.optimal_energy:.6f}")
print(f"degenerate optima      : {len(exact.optimal_solutions)}")
print(f"runtime                : {exact.runtime_seconds:.3f} s")
print()
print(f"uniform random success : {uniform:.3e}   <- the baseline QAOA is measured against")
print()
best = problem.decode(exact.optimal_feasible_solutions[0])
print("optimal candidate:", best["sequence"])
print("  mutations      :", best["mutations"])
"""))

cells.append(md("""
## 6. Classical baselines, on the identical QUBO

The classical solvers receive **only the QUBO** — no biological information QAOA does not
also have. Uniform random sampling is included deliberately: it is the baseline that
matched QAOA on a comparable peptide problem in Boulebnane et al. (2022).
"""))

cells.append(code("""
classical = run_all_classical(problem, seed=SEED)
print(f"{'method':<22}{'best energy':>14}{'abs gap':>12}{'evals':>10}{'runtime s':>11}")
print("-" * 69)
print(f"{'exact enumeration':<22}{exact.optimal_energy:>14.6f}{0.0:>12.2e}"
      f"{exact.n_assignments:>10,}{exact.runtime_seconds:>11.3f}")
for name, res in classical.items():
    gap = res.best_energy - exact.optimal_energy
    print(f"{name:<22}{res.best_energy:>14.6f}{gap:>12.2e}"
          f"{res.n_evaluations:>10,}{res.runtime_seconds:>11.3f}")
"""))

cells.append(md("""
## 7. QAOA

$$|\\psi(\\gamma,\\beta)\\rangle = \\prod_{l=1}^{p}
e^{-i\\beta_l \\sum_i X_i}\\, e^{-i\\gamma_l H_C}\\;|+\\rangle^{\\otimes N}$$

Because $H_C$ is diagonal, $e^{-i\\gamma H_C}$ factorises **exactly** into commuting
rotations, so the circuit is built from `RZ`/`RZZ`/`RX` directly rather than through
generic Pauli-evolution synthesis — which exponentiates a sparse matrix on every objective
evaluation and dominated the runtime before this change.

Two details that keep the numbers honest:

- **Success probability** is summed over **all** degenerate optima, not the frequency of
  one arbitrary bitstring.
- **Best sampled energy** comes from a *finite* shot sample. On an ideal simulator every
  basis state has nonzero amplitude, so reading the minimum off the statevector's support
  would return the global optimum regardless of how good the parameters were.
"""))

cells.append(code("""
from backend.optimization.qaoa import run_qaoa_ideal, verify_simulator_agreement

# The optimisation loop uses a specialised NumPy simulator for speed. It is checked
# against Qiskit's own Statevector rather than trusted.
agree = verify_simulator_agreement(problem, p=2, seed=0)
print(f"fast simulator == Qiskit Statevector : {agree['passed']}")
print(f"  max probability deviation          : {agree['max_probability_deviation']:.2e}")
print()

qaoa_runs = {}
for p in (1, 2, 3):
    res = run_qaoa_ideal(problem, p=p, optimizer="COBYLA", maxiter=300, seed=SEED % (2**31),
                         optimal_solutions=exact.optimal_solutions, n_restarts=3, shots=8192)
    qaoa_runs[p] = res
    cm = res.circuit_metrics
    print(f"p={p}  <H_C>={res.final_expectation:9.3f}  "
          f"success={res.success_probability:.5f} ({res.success_probability/uniform:6.1f}x uniform)  "
          f"feasible={res.feasible_probability:.3f}  "
          f"depth={cm['logical_depth']:3d}/{cm['transpiled']['depth']:3d}  "
          f"2q={cm['logical_two_qubit_gates']:3d}/{cm['transpiled']['two_qubit_gates']:3d}")
"""))

cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

for p, res in qaoa_runs.items():
    axes[0].plot(res.convergence, lw=1.0, label=f"p={p}")
axes[0].set_xlabel("objective evaluation"); axes[0].set_ylabel(r"$\\langle H_C \\rangle$")
axes[0].set_title("Variational convergence"); axes[0].legend(); axes[0].set_yscale("log")

ps = list(qaoa_runs)
axes[1].bar([str(p) for p in ps], [qaoa_runs[p].success_probability for p in ps],
            color="#0072B2")
axes[1].axhline(uniform, color="#999", ls="--", lw=1.2, label="uniform random")
axes[1].set_yscale("log"); axes[1].set_xlabel("QAOA depth $p$")
axes[1].set_ylabel("success probability"); axes[1].set_title("Depth vs success"); axes[1].legend()

axes[2].plot(ps, [qaoa_runs[p].circuit_metrics["logical_depth"] for p in ps], "o-",
             label="logical depth", color="#444")
axes[2].plot(ps, [qaoa_runs[p].circuit_metrics["transpiled"]["depth"] for p in ps], "s-",
             label="transpiled depth", color="#E69F00")
axes[2].plot(ps, [qaoa_runs[p].circuit_metrics["transpiled"]["two_qubit_gates"] for p in ps],
             "^-", label="transpiled 2q gates", color="#CC79A7")
axes[2].set_xticks(ps); axes[2].set_xlabel("QAOA depth $p$"); axes[2].set_ylabel("count")
axes[2].set_title("Depth vs circuit cost"); axes[2].legend()

plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 8. Noise

Aer simulations with **documented** error rates. These are simulated, not device
measurements, and are labelled as such everywhere.
"""))

cells.append(code("""
from backend.optimization.hardware import build_depolarizing_noise, build_thermal_noise
from backend.optimization.qaoa import sample_distribution

best_p = max(qaoa_runs, key=lambda p: qaoa_runs[p].success_probability)
params = np.array(qaoa_runs[best_p].final_params)
print(f"using p={best_p} parameters, 8192 shots\\n")

rows = []
for label, two_q in [("noiseless", 0.0), ("low 1e-3", 1e-3),
                     ("representative 1e-2", 1e-2), ("high 3e-2", 3e-2)]:
    model = None if two_q == 0 else build_depolarizing_noise(two_qubit_error=two_q)[0]
    out = sample_distribution(problem, best_p, params, shots=8192,
                              noise_model=model, seed=SEED % (2**31))
    rows.append((label, out["expectation"], out["success_probability"], out["feasible_probability"]))

thermal, _ = build_thermal_noise()
out = sample_distribution(problem, best_p, params, shots=8192, noise_model=thermal,
                          seed=SEED % (2**31))
rows.append(("thermal T1/T2", out["expectation"], out["success_probability"], out["feasible_probability"]))

print(f"{'noise model':<22}{'<H_C>':>12}{'success':>12}{'feasible':>11}")
print("-" * 57)
for label, e, s, f in rows:
    print(f"{label:<22}{e:>12.2f}{s:>12.5f}{f:>11.3f}")
"""))

cells.append(md("""
## 9. Does the QUBO actually represent the ML landscape?

This is the question most pipelines skip. The second-order surrogate is **exact by
construction** for 0, 1 and 2 selected mutations — the coefficients are exact finite
differences. For three or more it is an approximation, and the size of that approximation
is measurable.

The decision-relevant metric is not RMSE. It is: **where does the selection the QUBO
considers best actually land in the true ranking?**
"""))

cells.append(code("""
from backend.optimization.landscape import surrogate_fidelity_exhaustive

fid = surrogate_fidelity_exhaustive(landscape, scorer, k=3)
tp = fid["surrogate_top_pick"]

print(f"all feasible 3-mutation selections : {fid['n_selections']}")
print(f"surrogate RMSE vs true             : {fid['rmse']:.4f}")
print(f"Spearman rho                       : {fid['spearman_rho']:.4f}")
print()
print(f"the surrogate's own top pick       : {tp['mutations']}")
print(f"  its TRUE rank                    : {tp['true_rank']} of {tp['out_of']}  "
      f"({tp['percentile']:.0f}th percentile)")
print(f"the actual best 3-selection        : {fid['true_best']['mutations']}")
print()
print("50th percentile is what picking at random within that mutation count would achieve.")
print("This is why final candidates are ranked by DIRECT ML re-scoring, never QUBO energy.")
"""))

cells.append(md("""
## 10. Independent re-scoring and the Pareto front

Every candidate any solver returns is **re-scored by running the original models** on the
reconstructed sequence. The QUBO surrogate score and the direct ML score are stored side by
side, so surrogate error is visible per candidate.

A scalar `α/β` weighting encodes one trade-off preference, so the Pareto front over
(activity, −hemolysis) is reported as well, using strict dominance.
"""))

cells.append(code("""
from backend.optimization.analysis import rescore_candidates, pareto_analysis
from backend.optimization.qaoa import bitstring_to_x

bits, probs = [], []
for s, pr in sorted(qaoa_runs[best_p].distribution.items(), key=lambda kv: -kv[1])[:300]:
    bits.append(bitstring_to_x(s, problem.n_vars)); probs.append(pr)

cands = rescore_candidates(problem, bits, scorer, "qaoa", probabilities=probs)
parent_b = scorer.breakdown([PARENT])
pareto = pareto_analysis(cands, {"activity_norm": float(parent_b.activity_norm[0]),
                                 "hemolysis_norm": float(parent_b.hemolysis_norm[0])})

print(f"unique candidates re-scored    : {len(cands)}")
print(f"improving on the parent        : {sum(1 for c in cands if c.direct_delta_score > 0)}")
print(f"Pareto-optimal                 : {pareto['n_pareto_optimal']}")
print(f"dominating the parent          : {pareto['n_candidates_dominating_parent']}")
print()
print(f"{'sequence':<26}{'mutations':<18}{'activity':>9}{'hemolysis':>11}{'dS':>9}")
print("-" * 73)
print(f"{PARENT:<26}{'(parent)':<18}{parent_b.activity_raw[0]:>9.3f}"
      f"{parent_b.hemolysis_raw[0]:>11.3f}{0.0:>9.4f}")
for c in cands[:5]:
    print(f"{c.sequence:<26}{','.join(c.mutations):<18}{c.activity_raw:>9.3f}"
          f"{c.hemolysis_raw:>11.3f}{c.direct_delta_score:>9.4f}")
"""))

cells.append(code("""
pts = pareto["all_points"]
fig, ax = plt.subplots(figsize=(5.4, 4.2))
dom = [p for p in pts if not p["is_pareto_optimal"]]
front = sorted([p for p in pts if p["is_pareto_optimal"]], key=lambda p: p["activity_norm"])

ax.scatter([p["activity_norm"] for p in dom], [-p["hemolysis_norm"] for p in dom],
           s=18, c="#bbbbbb", edgecolors="none", label="candidates")
ax.plot([p["activity_norm"] for p in front], [-p["hemolysis_norm"] for p in front],
        "o-", color="#009E73", ms=7, lw=1.5, label="Pareto front")
ax.scatter([float(parent_b.activity_norm[0])], [-float(parent_b.hemolysis_norm[0])],
           s=170, marker="*", c="#E69F00", edgecolors="#7a5300", zorder=5, label="parent")
ax.set_xlabel("normalised predicted activity  ->  better")
ax.set_ylabel("- normalised predicted hemolysis  ->  better")
ax.set_title("Pareto front (model predictions only)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 11. Can the models predict a mutation's effect at all?

Everything above assumes the ML landscape is meaningful. That assumption is testable
against **real assay data**, without synthesising anything.

DRAMP contains natural analogue series, so the held-out test split already contains pairs
of independently measured peptides differing by 1–3 residues. For each pair the
already-trained model's predicted change is compared against the true measured change.
"""))

cells.append(code("""
report_path = "results/experiments/mutation_extrapolation_report.json"
if os.path.exists(report_path):
    rep = json.load(open(report_path))
    for target in ("activity", "hemolysis"):
        r = rep[target]; h = r["headline"]; same = r["by_study_provenance"]["same_study"]
        cross = r["by_study_provenance"]["cross_study"]
        print(f"--- {target} ---")
        print(f"  held-out point-mutant pairs : {r['n_pairs']}")
        print(f"  SAME-study  n={same['n_pairs']:4d}  rho={same['spearman_rho']:+.3f}  "
              f"directional={same['directional_accuracy']:.1%}   <- the fair comparison")
        print(f"  cross-study n={cross['n_pairs']:4d}  rho={cross['spearman_rho']:+.3f}  "
              f"directional={cross['directional_accuracy']:.1%}")
        print(f"  pooled would say            : {h['pooled_would_have_said']['directional_accuracy']:.1%}")
        print(f"  verdict                     : {h['verdict']}")
        print()
    print("Stratifying by publication is essential and it INVERTS the activity result.")
    print("Pairs from different papers carry inter-laboratory variation no sequence model")
    print("could predict; pooling them drags the activity figure down to a coin flip.")
else:
    print("not evaluated -- run: python -m backend.models.mutation_extrapolation")
"""))

cells.append(md(r"""
## 12. Conclusions

### What this demonstrates

| | |
|---|---|
| **The optimization is correct** | QUBO ≡ polynomial, Ising verified on **every** basis state, penalties verified by enumeration, fast simulator checked against Qiskit |
| **The QUBO is exact for K ≤ 2** | coefficients are exact finite differences of the real model output |
| **QAOA concentrates probability** | up to ~390× uniform random at N=18 in the scaling benchmark |
| **The limits are measured** | surrogate fidelity at k=3, and mutation-effect accuracy against real assay pairs |

### What it does **not** demonstrate

- **No quantum advantage.** Simulated annealing reaches the exact optimum at every size
  tested; QAOA never beats it. Depth does not reliably help — $p{=}3$ is worse than
  $p{=}2$ at several sizes, and in one run reached 0.9× uniform random.
- **No novelty for quantum AMP design.** Done and wet-lab validated in 2023 (Tučs et al.).
  What differs here is the exactly-computed QUBO, the measured surrogate error, and
  exact-optimum benchmarking.
- **No biological claim whatsoever.** ~59% directional accuracy on mutation effects means
  the output is a *shortlist for experiment*, not a prediction to act on.

### The honest one-liner

> A correctly-implemented, fully verified quantum optimization pipeline whose scientific
> limits have been measured rather than assumed — including the limits that make it look bad.

---

**Further reading in this repository:** `README.md` · `docs/ARCHITECTURE.md` ·
`docs/IMPLEMENTATION_AUDIT.md` · `research/scientific_audit.md`
"""))


def main() -> int:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(cells)} cells) — executing to embed outputs…")

    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=1800", str(OUT)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
        return result.returncode
    print("executed successfully — outputs embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
