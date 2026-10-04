# Q-Peptide — Architecture

High-level design, low-level design, and the data contracts between layers.

---

# Part 1 — High-Level Design

## 1.1 The problem being solved

Given a parent antimicrobial peptide `P` and a set of candidate amino-acid substitutions,
choose a small subset of substitutions that maximises a biological objective, subject to
constraints. Formally this is a **constrained quadratic binary optimization problem**:

```
maximise    ΔŜ(x) = Σᵢ Δᵢ xᵢ  +  Σᵢ<ⱼ Δᵢⱼ xᵢxⱼ
subject to  Σᵢ xᵢ ≤ K            (mutation budget)
            xᵢ + xⱼ ≤ 1          for substitutions at the same residue
            xᵢ ∈ {0,1}
```

The quantum computer solves **this** — a discrete subset-selection problem. It is not
simulating protein folding, molecular dynamics, or anything physical about the peptide.

## 1.2 Why this shape of problem

Three properties make it a genuine QUBO rather than a problem dressed up as one:

1. **The objective is quadratic and the quadratic term is not negligible.** Measured
   `mean|Δᵢⱼ| / mean|Δᵢ| ≈ 0.57–0.62` on real runs. If interactions were zero the problem
   would collapse to independent ranking, solvable by sorting.
2. **The constraints are combinatorial**, not box constraints — a budget plus pairwise
   exclusivity.
3. **The coefficients are expensive but finite.** `1 + N + C(N,2)` model evaluations.
   Affordable at `N ≈ 12–20`, which is the regime where exact enumeration is *also*
   affordable — and that overlap is what makes the benchmark meaningful.

## 1.3 System layers

```
┌─────────────────────────────────────────────────────────────────────┐
│  PRESENTATION      React + TypeScript + Tailwind + Recharts         │
│                    8 pages · renders only executed results          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP/JSON (axios)
┌───────────────────────────────┴─────────────────────────────────────┐
│  API               FastAPI + Pydantic                               │
│                    12 endpoints · async run store · validation      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ Python calls
┌───────────────────────────────┴─────────────────────────────────────┐
│  ORCHESTRATION     pipeline.py — the single end-to-end sequence     │
│                    shared by the API and the experiment scripts     │
└──┬──────────────┬──────────────┬──────────────┬─────────────────────┘
   │              │              │              │
┌──┴───────┐ ┌────┴──────┐ ┌─────┴──────┐ ┌─────┴──────────────┐
│ BIOLOGY  │ │ LANDSCAPE │ │   QUBO     │ │  QUANTUM           │
│ features │ │ Δᵢ, Δᵢⱼ   │ │ polynomial │ │  Ising → QAOA      │
│ XGBoost  │ │ conflict  │ │ penalties  │ │  noise → hardware  │
│ scoring  │ │ graph     │ │ matrix     │ │  solvers/baselines │
└──┬───────┘ └───────────┘ └────────────┘ └────────────────────┘
   │
┌──┴──────────────────────────────────────────────────────────────────┐
│  DATA              DRAMP → parse → curate → cluster → split         │
└─────────────────────────────────────────────────────────────────────┘
```

## 1.4 The central data flow

```
parent sequence
   │
   ├─► candidate generation ──► MutationSet { candidates, conflict_pairs }
   │      structural + physicochemical filters, then ML pre-screening
   │
   ├─► landscape ────────────► MutationLandscape { Δ (N,), Δ_pair (N,N) }
   │      1 + N + C(N,2) REAL model evaluations
   │
   ├─► QUBO build ───────────► QuboProblem { polynomial, Q, penalties }
   │      −ΔŜ(x) + budget penalty + conflict penalty
   │
   ├─► Ising map ────────────► SparsePauliOp (diagonal: I, Z, ZZ)
   │
   ├─► solvers ──────────────► exact | SA | greedy | local search | random | QAOA
   │
   └─► re-scoring ───────────► Candidate[] ranked by DIRECT ML score, not QUBO energy
          + Pareto front + trustworthiness diagnostics
```

## 1.5 Design decisions and their rationale

| Decision | Alternative rejected | Why |
|---|---|---|
| Δ coefficients **computed** by running the models | Fit a surrogate (factorization machine) to sampled scores | Removes a whole layer of surrogate error and makes the remaining error *measurable*. Affordable because `N` is small. |
| `N ≈ 12–20` variables | Scale up to look impressive | Exact enumeration must stay feasible — without the exact optimum there is no optimality gap, which is the headline metric. |
| Two separate property models | One multi-task model | Activity and hemolysis have different, partly disjoint label sets (2,328 vs 916 sequences, 550 overlap). |
| Both targets **regression** | Binary classification | Examined the labels: DRAMP `general_amps` is all-positive (no AMP negatives), and both quantities are continuous with no principled threshold. |
| Explicit RZ/RZZ circuit | Generic Pauli-evolution gate | `H_C` is diagonal, so it factorises exactly. Generic synthesis exponentiates a sparse matrix per objective evaluation and dominated runtime. |
| Candidates ranked by **direct ML score** | Rank by QUBO energy | Measured: at `k=3` the surrogate's ranking is near-uninformative. Energy ranking would propagate that error. |
| Slack-variable inequality encoding | Equality penalty for everything | `P(Σx−K)²` enforces *equality*. Using it for `≤` would be silently wrong. |
| Penalty from exact objective span | Hard-coded constant | Derived bound, then *verified by enumeration*. |

## 1.6 Reproducibility model

Every run records: dataset file and provenance, seed, model config and frozen
normalization, feature config, full candidate set, every QUBO coefficient, penalties and
their derivation, QAOA depth/optimizer/parameters/shots, noise model rates, backend, and
all package versions. A run record is self-contained — the JSON alone is enough to
reconstruct what happened.

---

# Part 2 — Low-Level Design

## 2.1 Module map

```
backend/
├── utils/peptide.py           sequence validation, MW, charge, pI, hydrophobic moment,
│                              identity, mutation application
├── features/featurize.py      sequence → feature vector; named feature contract
├── data/
│   ├── download.py            DRAMP + UniProt fetch, licence-aware
│   ├── labels.py              free-text → quantitative labels, unit unification
│   ├── curate.py              → activity.csv, hemolysis.csv, hemolysis_cls.csv
│   └── splits.py              greedy identity clustering, cluster-level splits
├── models/
│   ├── property_models.py     PropertyModel, Normalization, BiologicalScorer
│   ├── train.py               train both regressors, full validation report
│   ├── select_config.py       feature/hyperparameter sweep on VALIDATION only
│   ├── hemolysis_classifier.py  statement-based cross-check classifier
│   └── mutation_extrapolation.py  delta-prediction accuracy on real mutant pairs
├── optimization/
│   ├── mutations.py           candidate generation, compatibility graph
│   ├── landscape.py           Δᵢ, Δᵢⱼ, surrogate fidelity experiments
│   ├── qubo.py                polynomial + matrix, penalties, verification
│   ├── ising.py               QUBO ↔ Ising, bit-order conventions
│   ├── solvers.py             exact, SA, greedy, local search, random
│   ├── qaoa.py                circuit, fast simulator, optimisation, sampling
│   ├── hardware.py            noise models, IBM Quantum execution
│   ├── mitigation.py          readout-error inversion, DD + twirling options
│   ├── analysis.py            re-scoring, Pareto, trustworthiness diagnostics
│   └── pipeline.py            end-to-end orchestration
├── experiments/               final.py, scaling.py, figures.py,
│                              mitigation_study.py
├── schemas/models.py          Pydantic request/response
└── app.py                     FastAPI application
```

## 2.2 Key data structures

```python
@dataclass(frozen=True, order=True)
class MutationCandidate:
    position: int        # 0-based
    new_aa: str
    original_aa: str
    # .label -> "K12A" (1-based, human-facing)

@dataclass
class MutationSet:
    parent: str
    candidates: list[MutationCandidate]   # sorted → deterministic variable indices
    conflict_pairs: list[tuple[int,int]]  # (i,j) with i<j, same residue position

@dataclass
class MutationLandscape:
    delta: np.ndarray        # (N,)    Δᵢ
    delta_pair: np.ndarray   # (N,N)   Δᵢⱼ, symmetric, NaN on conflicts and diagonal
    parent_score: float
    n_model_evaluations: int

@dataclass
class QuboProblem:
    polynomial: QuboPolynomial   # SOURCE OF TRUTH: linear dict + upper-tri quadratic dict
    Q: np.ndarray                # derived symmetric matrix
    constant: float
    n_mutation_vars: int         # variables [0, N)
    n_slack_vars: int            # variables [N, N+S)
    budget_mode: BudgetMode      # AT_MOST_K | EXACTLY_K
```

## 2.3 The variable-mapping contract

The binding contract of the whole system:

```
binary variable index  ↔  MutationCandidate  ↔  (position, original_aa, new_aa)
```

- **Deterministic**: candidates are sorted by `(position, new_aa)` before indices are
  assigned, so the same inputs always produce the same mapping.
- **Serialisable**: `MutationSet.variable_map()` emits it as JSON; every run record and
  every API response carries it.
- **Bijective**: enforced in `__post_init__`; duplicates raise.
- Variables `[0, N)` are mutations; `[N, N+S)` are slack bits with no biological meaning.

## 2.4 Mathematical conventions, each fixed once and tested

**Matrix form.** `E(x) = xᵀQx + c`, `Q` symmetric. Since `xᵢ² = xᵢ`:

```
E(x) = Σᵢ Qᵢᵢxᵢ + 2Σᵢ<ⱼ Qᵢⱼxᵢxⱼ + c      ⟹      Qᵢⱼ = Qⱼᵢ = cᵢⱼ/2
```

The factor of 2 is the classic QUBO bug. `verify_matrix_matches_polynomial` evaluates the
matrix form against an independently written polynomial evaluator on 2,000 bitstrings
including all structured corners.

**Budget constraint — two genuinely different constraints.**

| mode | constraint | encoding | slack qubits |
|---|---|---|---|
| `AT_MOST_K` | `Σxᵢ ≤ K` | `s ∈ [0,K]` binary, penalise `P(Σxᵢ + s − K)²` | `⌈log₂(K+1)⌉` |
| `EXACTLY_K` | `Σxᵢ = K` | penalise `P(Σxᵢ − K)²` | 0 |

Slack weights are `[1, 2, 4, …, K − 2^(m−1) + 1]` — the top weight is clamped so the
representable set is exactly `{0,…,K}`, asserted for every `K` from 0 to 17.

**Penalty derivation.**

```
span = max_x ΔŜ(x) − min_x ΔŜ(x)      (exact, by enumeration)
P    = λ (span + ε),  λ > 1           (default λ = 2)
```

Any `P > span` is sufficient: satisfying the constraints costs at most `span`, so a
violation can never pay for itself. Then **verified by enumeration** — every infeasible
assignment must have strictly higher energy than the best feasible one. A test asserts
that a deliberately tiny penalty *fails* this check, because a verifier that always
passes is worthless.

**Ising mapping.** Via `xᵢ = (1 − Zᵢ)/2`:

```
offset = c + ½Σᵢcᵢ + ¼Σᵢ<ⱼcᵢⱼ
hᵢ     = −½cᵢ − ¼Σⱼ≠ᵢ cᵢⱼ
Jᵢⱼ    =  ¼cᵢⱼ
```

The identity offset is retained so `⟨x|H_C|x⟩ = E_QUBO(x)` **exactly**, which makes the
mapping exhaustively testable. Qubit ordering is Qiskit little-endian, isolated in
`pauli_label` and `bitstring_to_x`.

**Feasibility — two distinct predicates.** A selection can respect the mutation budget
while the *slack bits are set wrongly*, in which case the assignment still carries a large
penalty:

- `is_mutation_feasible(x)` — biological: no conflict, budget respected. Used for decoding.
- `is_feasible(x)` — all penalty terms vanish, including slack balance. Used for feasible
  probability and exact-solver counting.

Conflating them would overstate feasible probability and understate the QUBO's difficulty.

## 2.5 The quantum layer in detail

**Circuit construction.** `H_C` is diagonal, so `exp(−iγH_C)` factorises exactly into
commuting rotations:

```
exp(−iγ hᵢ Zᵢ)      = RZ(2γhᵢ)        on qubit i
exp(−iγ Jᵢⱼ ZᵢZⱼ)   = RZZ(2γJᵢⱼ)      on (i,j)
exp(−iβ Xᵢ)         = RX(2β)          mixer
```

**Two equivalent simulators, one checked against the other.**

- Optimisation loop: a specialised NumPy simulator — diagonal phase as one vectorised
  multiply, each RX as a reshape. `O(p·n·2ⁿ)` per evaluation.
- Everything else (sampling, noise, hardware): the real Qiskit circuit.
- `verify_simulator_agreement` asserts they match to ~1e-16, compared modulo global phase.
  The fast path is **checked, not trusted**.

**γ reparameterisation.** Penalised QUBOs have large coefficients, so `exp(−iγH_C)` wraps
through many full rotations and the landscape becomes violently oscillatory. The circuit
divides cost coefficients by their largest magnitude. This is a pure change of variables —
spectrum ordering is untouched, and every reported energy uses the *unscaled* diagonal.

**Finite-shot honesty.** On an ideal simulator every basis state has nonzero amplitude, so
reading "best energy" off the statevector support returns the global optimum regardless of
parameter quality. Best-sampled energy is therefore drawn from a finite multinomial sample.

**Success probability** is summed over **all** degenerate optima, and reported as a
multiple of uniform random sampling.

## 2.6 Complexity

| Stage | Cost | At N=14, K=3 |
|---|---|---|
| Candidate generation | `O(L·20)` admissible + `O(A)` model evals | ~245 scored |
| Landscape | `1 + N + C(N,2)` model evals | 92–106 evals |
| QUBO build | `O(N²)` | instant |
| Exact solve | `O(2ⁿ·n)` vectorised | 65,536 states, 0.03 s |
| Penalty verify | `O(2ⁿ·n)` vectorised | 0.1 s |
| Ising verify | `O(2ⁿ·terms)` | exhaustive |
| QAOA objective | `O(p·n·2ⁿ)` per evaluation | ~900 evals |

The binding constraint is `2ⁿ`. At `n = 20` (N=18 + 2 slack) exact enumeration is ~1M
states and still comfortable; beyond that the exact baseline — the entire point of the
benchmark — disappears.

## 2.7 Testing strategy

**189 tests.** The principle: test the *mathematics against independent recomputation*,
not against itself.

| Area | What is asserted |
|---|---|
| Features | valid/invalid/empty sequences; composition sums to 1; pI zeroes net charge; determinism |
| Mutations | protected residues never mutated; conflicts match positions; deterministic indexing |
| Landscape | `Δᵢ` and `Δᵢⱼ` vs **independently recomputed** score differences |
| QUBO | matrix ≡ polynomial (hand-built case + exhaustive); slack range for every K 0–17; **insufficient penalty must FAIL** |
| Ising | `⟨x\|H_C\|x⟩ = E(x)` for **every** basis state; hand-computed 1- and 2-variable cases |
| Solvers | exact optimum vs brute force; degeneracy count complete; incremental Δenergy vs full recompute |
| QAOA | fast simulator ≡ Qiskit at p=1,2,3; zero params → uniform superposition; success prob = manual sum over optima |
| Extrapolation | strata partition exactly; evaluated pairs disjoint from training |

---

# Part 3 — Request/response contracts

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | model, Aer, credential status + versions |
| `POST /api/peptide/analyze` | descriptors + model-predicted properties |
| `POST /api/mutations/generate` | candidate set, compatibility graph, landscape |
| `POST /api/qubo/build` | coefficients, penalties, mapping, Ising, verification |
| `POST /api/benchmark/exact` | brute-force ground truth |
| `POST /api/benchmark/classical` | SA, greedy, local search, random |
| `POST /api/qaoa/run` | one depth; optional noise; **real bound circuit** |
| `POST /api/benchmark/run` | exact + classical + QAOA side by side |
| `POST /api/optimize` | full async pipeline run |
| `GET /api/results/{id}` | fetch a run (API store or saved experiment) |
| `GET /api/experiments[/{name}]` | saved experiment reports |
| `GET /api/research` | literature review, audit, curation, ML + extrapolation reports |

**Validation.** Sequences are checked for the canonical 20 residues and length 5–60;
`budget_mode` is a required explicit enum — the API never guesses whether "≤ K" or "= K"
was meant. Invalid input returns 422 with a specific message; missing models return 503
with the command to fix it.

**The "not evaluated" contract.** No endpoint ever returns a placeholder number. A stage
that did not run returns `{"status": "not evaluated", "reason": "..."}`, and the UI renders
that state explicitly.
