# Q-Peptide — Literature Review and Formulation Decisions

This document records (a) what the literature establishes, (b) the design decisions that
follow, and (c) the decisions the literature *forbids*. Full bibliographic entries are in
[`papers.md`](papers.md); this document cites them by section number (e.g. P§1.1).

All dataset counts quoted here were measured by running the parser against the downloaded
DRAMP release, not estimated.

---

## 1. Where Q-Peptide sits in the literature

### 1.1 The idea is not new; the formulation is the contribution

**Quantum optimization of nonhemolytic antimicrobial peptides has already been done and
experimentally validated** (P§1.1, Tučs et al. 2023, ACS Med. Chem. Lett.). That work paired a
binary VAE latent space with a factorization-machine surrogate and a D-Wave annealer, and
confirmed three active peptides in the wet lab.

Q-Peptide therefore **must not claim** novelty for "using quantum computing to design
non-hemolytic AMPs". What remains genuinely different, and what this project is actually
about:

| Dimension | Tučs et al. 2023 (P§1.1) | Q-Peptide |
|---|---|---|
| Binary variables | 64 opaque VAE latent bits | explicit `(position, aa_from, aa_to)` substitutions |
| QUBO coefficients | **learned** by fitting an FM to sampled scores | **computed exactly** as finite differences of the ML score |
| Surrogate fidelity | not quantified | measured against the true ML landscape for ≥3 mutations |
| Optimality baseline | none (annealer only) | brute-force exact optimum at small `N` |
| Hardware paradigm | analog annealing | gate-based QAOA |

The second row is the methodological core. Because the mutation set is small (`N ≈ 12–20`),
the QUBO coefficients need not be *fit* — each one can be obtained by **actually evaluating
the ML model** on the corresponding mutant. That removes one entire layer of surrogate error
present in the prior work, and it makes the remaining error *measurable* (§6).

### 1.2 The cautionary result that sets the tone

P§1.2 (Boulebnane et al., npj Quantum Information 2023) applied QAOA to lattice peptide
conformational sampling on up to 20 qubits and found that **"the performance of QAOA can be
matched by random sampling up to a small overhead"**, concluding that the results *"cast
serious doubt on the ability of QAOA to address the protein folding problem in the near
term."*

Consequences adopted as hard rules:

1. **Uniform random bitstring sampling is a mandatory baseline.** It is cheap and it is the
   baseline that embarrassed QAOA in the closest comparable study. If QAOA does not beat it,
   Q-Peptide reports that.
2. QAOA is framed as an **object of measurement**, not a solver assumed to work. The research
   questions are "how close does it get" and "what does depth/noise cost", never "quantum
   wins".
3. Note the problems differ: P§1.2 optimised *conformation* under a hard self-avoidance
   constraint; Q-Peptide optimises *sequence choice* with a much denser, lower-`N` QUBO. The
   negative result does not transfer automatically — which is exactly why it must be
   re-measured here rather than assumed either way.

---

## 2. Data: what the labels actually are

### 2.1 Source selection

**Primary: DRAMP** (P§6.1) — chosen for an explicit **CC BY 4.0** licence, bulk download
without registration, and MIC values with target organisms. DBAASP (P§6.2) is the secondary
source because it records erythrocyte cytotoxicity alongside MIC. UniProt (P§6.3) supplies
background sequences only if needed.

### 2.2 Measured label availability (DRAMP `general_amps`, 12,784 rows)

| Stage | Count |
|---|---|
| Rows in release | 12,784 |
| Valid sequences over the canonical 20 amino acids | 10,951 |
| …and length 5–60 | **10,074** |
| Unique sequences among those | 9,391 |
| Rows with ≥1 parseable MIC | 2,429 (**2,324 unique**) |
| Rows with a quantitative 50%-effect hemolytic dose | 1,000 (**894 unique**) |
| Rows with only a qualitative hemolysis statement | 383 (370 unique) |
| Rows with an explicit "no hemolysis data" sentinel | 5,852 |
| Hemolysis statements not parseable into a label | 2,646 |
| Rows with **both** activity and hemolysis labels | 771 |

Three facts here drive the whole modelling design:

**(a) DRAMP `general_amps` is an all-positive set.** Every row is annotated antimicrobial.
There are **no AMP negatives**. Therefore *AMP-vs-non-AMP binary classification is
impossible from this source alone*, and manufacturing negatives by random sampling from
Swiss-Prot reproduces the known artefact in the HemoPI-1 construction (P§2.3) where negatives
are separable on composition alone. **Decision: the activity model is a regression, not a
classifier.** This is precisely the situation spec §5 warns about ("do NOT force all datasets
into binary classification without examining their labels").

**(b) Both target quantities are available as continuous measurements.** MIC and 50%-effect
hemolytic dose are both real concentrations. Spec §6 forbids arbitrarily binarising continuous
experimental measurements, and there is no principled threshold for "hemolytic" in the
literature (reported HD50 values here span 0.005–8000 µM with a median of 100 µM; any cut is
a judgement call). **Decision: the hemolysis model is also a regression.** A statement-based
classifier is fitted *additionally* as a cross-check
(`backend/models/hemolysis_classifier.py`), using only labels that come from the
experimenters' own stated determination — never from thresholding a number. Measured:
balanced accuracy **0.672 ± 0.064** against a 0.561 majority baseline, ROC AUC 0.767, and
it agrees with the regression at Spearman **+0.209** despite a completely disjoint label
definition — which is evidence the weak hemolysis signal is real rather than an artefact of
how the regression target was constructed.

**(c) The labels are free text with heterogeneous units.** MIC and dose values appear as
`µM`, `µg/mL`, `mg/L`, `nM`, `ng/mL`, with qualifiers (`≤`, `<`, ranges like `0.025-6.4`).
Unit unification is mandatory and is done by computing molecular weight from the sequence
itself:

```
c[µM] = c[µg/mL] / MW[g/mol] × 1000
```

This is exact for mass-per-volume units and needs no external data, since MW follows from the
sequence. Molar units convert by a pure power of ten.

### 2.3 Target transformations

Both targets use the standard pharmacological potency transform (a log-dose scale), which is
monotonic and makes multiplicative dose differences additive:

$$y_A \;=\; 6 - \log_{10}\!\big(\mathrm{MIC}\,[\mu M]\big) \qquad\text{(higher = more active)}$$

$$y_H \;=\; 6 - \log_{10}\!\big(D_{50}\,[\mu M]\big) \qquad\text{(higher = more hemolytic)}$$

The constant 6 converts µM to molar inside the logarithm, so both are `−log₁₀(molar)`, i.e.
the conventional `pMIC` scale. Measured ranges: `y_A ∈ [1.68, 10.52]`, mean 5.54;
`y_H ∈ [2.10, 8.27]`, mean 4.31.

**Sign convention, fixed once:** *higher `y_A` is better, higher `y_H` is worse.*

**Aggregation convention.** A peptide with MICs against several organisms is reduced to its
**minimum MIC** (most-susceptible organism), i.e. maximum `y_A`. This is the standard
"best-case potency" convention. It is a choice, not a fact, and is recorded as a limitation:
it biases toward peptides tested against sensitive strains.

### 2.4 Leakage control

The literature is unambiguous that **random splits are invalid** for peptide data (P§3.1,
DataSAIL; P§3.2, CD-HIT practice). AMP studies conventionally cluster at **70% identity** and
split at cluster level; hazard-screening work uses ≤40%.

**Decisions:**
- Exact duplicate sequences are collapsed before anything else (labels aggregated by median).
- Sequences are clustered by identity and the **cluster is the grouping unit**; splits use
  `StratifiedGroupKFold` / `GroupShuffleSplit` on cluster id so no cluster spans train and
  test.
- Default threshold **70% identity**, matching AMP convention, with the value recorded in
  every experiment manifest.
- CD-HIT is an external binary; to keep the pipeline reproducible without system packages, an
  equivalent **greedy incremental identity clustering** is implemented in process: sort by
  descending length, make each unassigned sequence a cluster representative, and assign any
  sequence whose identity to that representative exceeds the threshold. This mirrors CD-HIT's
  documented greedy strategy (P§3.2). Identity is computed from an alignment-free
  normalised-gapless comparison for equal lengths and a local alignment otherwise; the exact
  definition is in the code and is stated in the report.
- **A random-split model is also trained, purely to report the optimism gap** between random
  and cluster-level splitting. That difference is a headline validity number, not a
  performance claim.
- `k`-mer leakage is a live threat because dipeptide composition is in the feature set
  (P§3.3). Reported as a limitation; cluster-level splitting mitigates but does not eliminate
  it.

---

## 3. The biological objective

### 3.1 Why two terms are required

P§5.1 (Dathe et al. 2001, FEBS Lett.) is the empirical anchor. For magainin II amide analogues
spanning net charge +3 to +7:

- activity and selectivity improve with charge **up to about +5**;
- beyond +5, there is *"a dramatic increase of hemolytic activity and loss of antimicrobial
  selectivity"*;
- selectivity is **restored by reducing hydrophobicity** of the hydrophobic face.

This yields three formulation consequences:

1. **A single-objective "maximise activity" formulation is known to fail**, reproducing
   over-cationic hemolytic analogues. The objective must penalise hemolysis explicitly.
2. The response is **non-monotonic with a threshold**. A purely additive first-order model
   over substitutions cannot express a threshold. This is independent biological motivation
   for the pairwise `Δᵢⱼ` terms — they are not a mathematical convenience to manufacture a
   quadratic problem.
3. Charge and hydrophobicity **interact**: a charge-increasing substitution plus a
   hydrophobicity-reducing substitution should be **synergistic**. This is a falsifiable
   prediction that the mutation-landscape experiment can test, and it is the specific
   structure the QUBO is meant to capture.

### 3.2 Normalization

Raw `y_A` and `y_H` are on different scales (means 5.54 vs 4.31, sds 0.89 vs 0.92) and are
*not* interchangeable units. Combining them raw would make the weights uninterpretable, so
each is standardised using statistics **frozen from the training distribution**:

$$\tilde A(P) = \frac{A(P)-\mu_A}{\sigma_A}, \qquad \tilde H(P) = \frac{H(P)-\mu_H}{\sigma_H}$$

`μ_A, σ_A, μ_H, σ_H` are computed once on the training split, serialised, and reused for every
prediction. **They are never recomputed per candidate** — doing so would change the objective
between candidates and silently invalidate every comparison (spec §9).

### 3.3 Scalar score and sign discipline

$$S(P) \;=\; \alpha\,\tilde A(P) \;-\; \beta\,\tilde H(P), \qquad \alpha,\beta \ge 0$$

**Larger `S` is better.** Optimisation maximises `S`, and the QUBO minimises `E = −ΔŜ + penalties`.
The sign flip happens in exactly one place in the code, and a test asserts that the exact
solver's optimum corresponds to the maximum `S` candidate.

Because `Ã` and `H̃` are both standardised, `α` and `β` are in comparable units and
`α = β = 1` is a meaningful neutral default.

### 3.4 Parent-relative objective

Optimisation targets the **improvement over the parent**, not the raw score:

$$\Delta S(P') \;=\; S(P') - S(P)$$

This makes `ΔS = 0` the "do nothing" baseline, so the sign of the result directly answers
Research Question 1, and it cancels any constant offset in the standardisation.

### 3.5 Pareto analysis is not optional

A scalar `α/β` weighting imposes a single trade-off preference. Following the non-dominated
sorting used in P§1.1, candidates are additionally reported as a **Pareto front** over
`(Ã, −H̃)`, with strict dominance: `P₁` dominates `P₂` iff it is no worse in both objectives
and strictly better in at least one. No candidate is called "best" without naming the
objective.

---

## 4. Mutation variables and the quadratic model

### 4.1 Problem size justification

`N ≈ 12–20` binary variables is not an arbitrary hackathon-scale choice; it is pinned from
three directions:

- **Experimental practice.** P§5.2 (amino-acid scanning) uses 7–12 candidate substitutions per
  position and 1–9 changes per variant. `N ≈ 12–20` with `K ≈ 3–4` sits inside the regime
  real AMP engineering campaigns operate in.
- **Exact ground truth must remain computable.** The exact solver enumerates `2^N`. At
  `N = 20` that is ~10⁶ evaluations — fast. This is the binding constraint, and it is a
  *scientific* requirement: without the exact optimum there is no optimality gap, which is the
  gap left open by P§1.1.
- **Coefficient cost is quadratic in `N`.** Every `Δᵢ` and `Δᵢⱼ` requires real ML inference:
  `1 + N + C(N,2)` evaluations, i.e. 191 at `N = 19`. Affordable; it would not be at `N = 200`.

Raising `N` to look more impressive would destroy the exact baseline, which is the most
scientifically valuable part of the benchmark. Recorded explicitly.

### 4.2 Exact coefficients, not fitted ones

With `P_i = P ⊕ m_i`:

$$\Delta_i = S(P_i) - S(P)$$
$$\Delta_{ij} = S(P_{ij}) - S(P_i) - S(P_j) + S(P)$$

`Δᵢⱼ` is the standard second-order finite difference (a discrete mixed partial). Both are
obtained by **running the ML models on the actual mutant sequences**. `Δᵢⱼ = 0` means
additive, `> 0` synergistic, `< 0` antagonistic, under the chosen score.

The second-order surrogate is

$$\widehat{\Delta S}(x) = \sum_i \Delta_i x_i + \sum_{i<j} \Delta_{ij} x_i x_j$$

which by construction is **exact** for 0, 1 and 2 selected mutations. It is an approximation
only for ≥3, which is why §6 measures it there.

---

## 5. Constraint encoding

### 5.1 Mutation budget — the inequality/equality distinction

Spec §17 is a real mathematical trap. `P(Σxᵢ − K)²` enforces **equality** `Σxᵢ = K`, not
`Σxᵢ ≤ K`. Both modes are implemented and **named in the API and UI**, never conflated:

- **`AT_MOST_K`** — slack-variable encoding (P§4.2, standard construction): introduce integer
  slack `s ∈ [0, K]` as `⌈log₂(K+1)⌉` binary variables with binary weights, impose
  `Σᵢxᵢ + s = K`, and penalise `P(Σᵢxᵢ + s − K)²`.
- **`EXACTLY_K`** — penalise `P(Σᵢxᵢ − K)²` directly, no slack qubits.

The slack encoding is the exact default because exactness is the stated priority and the
qubit overhead is small at this `N`. P§4.2 and P§4.3 show slack encodings degrade at scale and
offer **unbalanced penalization** as the slack-free alternative; this is documented as the
qubit-efficient option and noted as future work, not silently substituted.

A standard binary expansion covers `[0, K]` exactly when `K+1` is a power of two; otherwise
the top weight is clamped so the representable range is exactly `[0, K]` with no
over-coverage. A test asserts the representable slack set equals `{0,…,K}`.

### 5.2 Same-position exclusivity

Two substitutions at the same residue cannot both apply. For each conflicting pair,
`xᵢ + xⱼ ≤ 1`, encoded as `P_pos · xᵢxⱼ` — a pure quadratic penalty needing no slack, since
the constraint is violated only by the single assignment `xᵢ = xⱼ = 1`.

Conflicts are held in an explicit **compatibility graph** (nodes = mutations, edges =
conflicts) built once and passed to the QUBO builder as data. The optimizer never parses
strings to decide compatibility (spec §19).

### 5.3 Penalty magnitude

Penalties are derived, not guessed. With `B` a bound on the achievable objective gain,

$$B \;=\; \sum_i |\Delta_i| \;+\; \sum_{i<j} |\Delta_{ij}|$$

(the maximum possible magnitude of `ΔŜ` over all `x`, by the triangle inequality), the
penalty is set to

$$P \;=\; \lambda\,(B + \epsilon), \qquad \lambda > 1$$

with a default safety factor `λ = 10` that is recorded per experiment. This follows the
Lucas 2014 principle (P§4.1): the penalty must exceed the largest objective gain obtainable
by violating the constraint. `P = 1000` or any other hard-coded constant is explicitly
rejected.

This bound is then **verified empirically**, not trusted: at small `N`, enumerate all `2^N`
assignments, partition into feasible and infeasible, and assert
`min E(infeasible) > min E(feasible)`. If that fails, `λ` is raised and the failure recorded.

### 5.4 Matrix convention

One convention, stated once and tested:

$$E(x) = x^\top Q x + c, \qquad Q \text{ symmetric}$$

so that `xᵢ² = xᵢ` for binary `x` gives

$$E(x) = \sum_i Q_{ii}x_i + 2\sum_{i<j}Q_{ij}x_ix_j + c$$

Hence a desired polynomial term `c_ij·xᵢxⱼ` requires **`Q_ij = Q_ji = c_ij/2`**, and a linear
term `c_i·xᵢ` requires `Q_ii = c_i`. The factor of two is the single most common QUBO bug, so
a test evaluates the matrix form and an independently-written polynomial form on random
bitstrings and asserts agreement (spec §22).

---

## 6. Validating the surrogate

The second-order model is exact for `|x| ≤ 2` by construction and unvalidated beyond. The
experiment (spec §15, §37): sample feasible `x` with `Σxᵢ ≥ 3`, compute the **true** ML score
`S(P ⊕ x) − S(P)` by building the mutant and running the models, compare against
`ΔŜ(x)`, and report MAE, RMSE, Pearson and Spearman, plus the max absolute residual.

This is the honest measure of whether the QUBO represents the ML landscape at all. A poor
result is a **reportable finding**, not a reason to quietly add cubic terms. It also bounds
what any solver — quantum or classical — can mean: if the surrogate is a poor proxy, finding
its exact optimum is of limited biological value, and that caveat attaches to every downstream
number.

---

## 7. Quantum formulation

### 7.1 Ising mapping

Via `xᵢ = (1 − Zᵢ)/2` (P§4.1), the QUBO becomes a diagonal Ising Hamiltonian `H_C` built as a
`SparsePauliOp` over `I`, `Z`, `ZZ` terms. Because `H_C` is diagonal in the computational
basis, correctness is **exhaustively checkable**: for every basis state at small `N`, assert
`⟨x|H_C|x⟩ = E_QUBO(x)` up to the documented constant offset. This is a test, not an argument.

### 7.2 Ansatz and mixer

Standard QAOA:

$$|\psi(\gamma,\beta)\rangle = \prod_{l=1}^{p} e^{-i\beta_l \sum_i X_i}\,e^{-i\gamma_l H_C}\;|+\rangle^{\otimes N}$$

The **X mixer** with penalty-based constraints is used for the MVP, as spec §27 permits.
P§1.3 (penalty-free lattice folding) and the maximum-independent-set mixer are the documented
alternative: a constraint-preserving mixer would keep the state inside the feasible subspace
and remove the penalty terms entirely. Not implemented in the MVP; recorded as the single most
promising formulation improvement.

### 7.3 Optimizer

Spec §29 warns against assuming COBYLA. Both **COBYLA** and at least one alternative
(**Nelder–Mead** / **SPSA**-style) are run, with optimizer, initial parameters, final
parameters, iteration and evaluation counts, and final expectation recorded. Noisy/sampled
objectives favour methods tolerant of stochastic evaluations, which is why the comparison is
run rather than assumed.

### 7.4 What gets reported

Per spec §32–36: absolute energy gap `E_found − E*` as the primary metric (**never** the ratio
`E_found/E*`, which is meaningless for sign-indefinite QUBO energies); success probability as
`Σ_{x ∈ X*} p(x)` over **all** degenerate optima; logical **and** transpiled depth; two-qubit
gate counts; runtime; evaluation counts. Ideal, noisy and (if a token is available) hardware
results are kept in separate, clearly-labelled records.

---

## 8. Claims this project may not make

Fixed before any results exist, so results cannot bend them:

1. **No quantum advantage.** P§1.2 found QAOA matched by random sampling on a peptide problem.
   Any favourable result here is a small-instance observation, not evidence of asymptotic
   advantage, and the scaling study (`N = 6…20`) is far too small to support such a claim.
2. **No novelty for quantum AMP design.** P§1.1 did it in 2023 with wet-lab validation.
3. **No biological efficacy.** Every output is a *model-predicted candidate*. The models are
   trained on heterogeneous literature-extracted values with the limitations in §2 and §9.
   Experimental validation would be required. Language is restricted to "predicted",
   "computational candidate", "model-predicted activity/hemolysis".
4. **No claim that near-99% accuracies are achievable or meaningful** (P§2.2, P§2.3).

---

## 9. Known limitations, stated up front

| # | Limitation | Consequence |
|---|---|---|
| 1 | MIC/dose values parsed from **free text**; 2,646 hemolysis statements were unparseable and dropped | Label noise; dropped rows may not be missing at random (e.g. percent-at-concentration reporting styles) |
| 2 | Unit conversion assumes the sequence's own MW and linear, unmodified peptides | Rows with terminal modifications or non-standard residues are excluded, shrinking and biasing the set |
| 3 | **Minimum-MIC** aggregation across organisms | Biases toward peptides tested on susceptible strains; a single `y_A` hides organism specificity |
| 4 | Hemolysis set is small (894 unique) and pools HD50/HC50/LD50/EC50 across species (human, rabbit, sheep, guinea pig) and assay protocols | The weakest link in the pipeline; `H` is noisier than `A` |
| 5 | DRAMP `general_amps` is **all-positive** | No AMP classifier is possible from this source; activity is regression-only |
| 6 | Dipeptide composition invites **k-mer leakage** (P§3.3) | Cluster-level splitting mitigates but does not remove it |
| 7 | Identity clustering is an in-process greedy reimplementation, not CD-HIT | Clusters may differ slightly from the CD-HIT convention; threshold is reported |
| 8 | Models are evaluated on *natural* peptides but applied to *point mutants* of a parent | **Now measured directly** (see `scientific_audit.md` §9): on real held-out point-mutant pairs from the same publication, directional accuracy is 59.4% (activity) and 58.9% (hemolysis), ρ ≈ 0.22–0.25. Weak but above chance. Note §6 measures only *surrogate* error; this measures *model* error |
| 9 | Scalar `α/β` encodes one trade-off preference | Mitigated by Pareto reporting, not eliminated |
| 10 | Prediction uncertainty is model uncertainty, never experimental error | Must not be read as an error bar on a measurement |

Limitation **8** deserves emphasis: it is the deepest one. The QUBO can be a perfect
representation of the ML landscape (§6) while the ML landscape is itself wrong about mutants.
Measuring surrogate error does **not** validate the biology.
