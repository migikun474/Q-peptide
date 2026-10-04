# Q-Peptide — Scientific Audit

An adversarial pass over the project, asking what could be wrong rather than what works.
Every number cited is from an executed run in `results/experiments/`; nothing here is
asserted from the design alone.

**Verdict in one line:** the optimization machinery is correct and verified at every
problem size; the *scientific* weak points are the hemolysis model, the second-order
surrogate at three or more mutations, and the models' weak (≈59% directional) ability to
rank mutation effects — all three measured and reported rather than hidden.

---

## 1. Machine learning

### Is there leakage?

**Controlled, not eliminated.** Exact duplicates are collapsed before anything else.
Sequences are clustered at 70% identity and split at the **cluster** level, so no cluster
spans train and test.

The optimism gap was measured rather than assumed, by training the same model on a random
split purely for comparison:

| model | cluster-split test RMSE | random-split test RMSE | difference |
|---|---|---|---|
| activity | 0.759 | 0.683 | +0.076 |
| hemolysis | 0.714 | 0.636 | +0.078 |

A random split understates error by about 0.08 RMSE in both cases — modest, but real, and
it confirms the cluster split was necessary.

**Residual risk.** The greedy identity clustering is a reimplementation of CD-HIT's
documented strategy, not CD-HIT itself, so cluster boundaries may differ. Sequences below
70% identity that nonetheless share motifs can still straddle the split.

### Are labels correctly interpreted?

**Yes, and the interpretation changed the design.** Looking at the actual label columns
rather than assuming forced two corrections:

1. DRAMP `general_amps` is **all-positive** — there are no AMP negatives. An AMP/non-AMP
   classifier is therefore impossible from this source, so activity is a regression. Had
   this not been checked, the obvious move would have been to generate negatives by random
   sampling, reproducing a known artefact.
2. Both targets are **continuous measurements** with no principled threshold (observed
   D₅₀ spans 0.005–8000 µM), so hemolysis is also a regression. A statement-based
   classifier is fitted separately as a cross-check, labelled only from the experimenters'
   own stated determinations.

### Are transformations documented?

Yes. Both targets use `6 − log₁₀(c[µM])`, i.e. `−log₁₀(c[M])`, the conventional pMIC
potency scale. Mass-per-volume units are converted using molecular weight computed from
the sequence itself. Sign convention: **higher `y_A` is better, higher `y_H` is worse**,
stated once and enforced in a single class.

### Are metrics appropriate?

Yes, with one caveat worth stating plainly. Regression metrics (MAE, RMSE, R², Pearson,
Spearman) are reported against a **mean-prediction baseline**, which is the only threshold
that makes a model worth having:

| model | test RMSE | mean-baseline RMSE | beats baseline by | Spearman ρ |
|---|---|---|---|---|
| activity | 0.760 | 0.904 | 16% | 0.559 |
| hemolysis | 0.720 | 0.746 | **3.5%** | 0.423 |

**The hemolysis model is weak.** It clears the baseline, but by 3.5%. Rank correlation of
0.42 means it carries usable ordering information, which is what the objective actually
needs, but any conclusion about hemolysis inherits this.

An independent cross-check supports the signal being real. A classifier trained on a
disjoint labelling — the experimenters' own stated determinations, with no thresholding of
any numeric dose — reaches balanced accuracy **0.672 ± 0.064** against a 0.561 majority
baseline (ROC AUC 0.767), and its predictions correlate with the regression's at Spearman
**+0.209**. Two different label definitions pointing the same way is harder to explain as
an artefact than either result alone.

### Was the test set protected?

Yes. Feature blocks and hyperparameters were selected by
`backend.models.select_config` on the **validation** split only, over a 4×4 grid; the test
split was scored once afterwards. The selection is recorded in
`results/experiments/model_selection_sweep.json`.

That sweep produced a finding worth keeping: including the 400-dimensional dipeptide block
**lowered** validation performance for both targets, most sharply for hemolysis (validation
Spearman 0.62 without vs 0.30 with). This is the k-mer / dimensionality problem the
literature warns about, measured directly. The block is implemented, as specified, but not
selected.

---

## 2. Biological objective

**Are activity and hemolysis on compatible scales?** Not raw — means 5.54 vs 4.31. Both
are standardised using μ/σ computed once from **training-split predictions** and frozen
thereafter, so `α` and `β` are comparable and `α = β = 1` is a meaningful default.
Standardisation uses predictions rather than labels because it is predictions the objective
combines.

**Are the statistics genuinely frozen?** Yes. They are serialised into the model metadata
and reused for every prediction. Recomputing per candidate would silently change the
objective between candidates; the code has no path that does so.

**Is the parent-relative score correct?** Verified directly: `ΔS = 0` for the empty
selection is asserted by a test, and `Δᵢ`/`Δᵢⱼ` are checked against independently
recomputed score differences for every single mutant and every compatible pair.

**Is the scalar objective sufficient?** No, and it is not relied upon alone. A Pareto front
over `(Ã, −H̃)` with strict dominance is reported alongside. At `K=3`, **12 candidates
dominate the parent** — better in one objective and no worse in the other — which is a
stronger statement than any single weighted score.

---

## 3. QUBO

**Are `Δᵢ` correct?** Verified for all 14 variables against directly recomputed
`S(P ⊕ mᵢ) − S(P)`, tolerance 1e-9.

**Are `Δᵢⱼ` correct?** Verified for every compatible pair against the mixed second
difference. At `K=3`, 316 of these were computed from real model evaluations.

**Are constraints correctly encoded?** The two budget modes are kept distinct and both
tested. `P(Σxᵢ − K)²` enforces **equality**; using it for an inequality would be a silent
error, so `AT_MOST_K` uses a slack construction whose representable range is asserted to be
exactly `{0,…,K}` for every `K` from 0 to 17.

A distinction the implementation makes that is easy to get wrong: a selection can respect
the mutation budget while the **slack bits are set wrongly**, in which case the assignment
still carries a large penalty. Counting such states as feasible would overstate feasible
probability. `is_feasible` (all penalties vanish) and `is_mutation_feasible` (biological
feasibility) are therefore separate predicates, and the vectorised mask used in the hot
path is asserted to agree with the scalar one.

**Are penalty coefficients sufficient?** Derived from the exact objective span computed by
enumeration, then **verified by enumeration** rather than trusted:

| run | penalty | margin (best infeasible − best feasible) | feasible states |
|---|---|---|---|
| final K=2 | 24.9 | **48.54** | 116 / 65,536 |
| final K=3 | 24.9 | **47.68** | 418 / 65,536 |

A positive margin means no infeasible assignment can beat the feasible optimum. The checker
itself is guarded: a test builds a landscape where every mutation is beneficial (so the
unconstrained optimum over-selects) and asserts that a 1e-6 penalty **fails** the check. A
verifier that always passes would be worthless.

**Is the matrix convention consistent?** `E(x) = xᵀQx + c` with symmetric `Q` and
`Qᵢⱼ = cᵢⱼ/2`. Checked by evaluating the matrix form against an independently written
polynomial evaluator on 2,000 bitstrings including every structured corner case; maximum
deviation 7e-11. This is the single most common QUBO bug and it is the one check that runs
at every problem size.

---

## 4. Ising mapping

**Is the binary-to-spin mapping correct?** `xᵢ = (1 − Zᵢ)/2`, with the identity offset
retained so `⟨x|H_C|x⟩` equals `E_QUBO(x)` exactly.

**Does every basis state have the correct energy?** Checked **exhaustively**, not sampled:

| run | qubits | states checked | max deviation |
|---|---|---|---|
| final K=2 | 16 | 65,536 | 9.1e-12 |
| final K=3 | 16 | 65,536 | 9.1e-12 |
| scaling N=6…18 | 8–20 | all, to 18 qubits | passed at every size |

Qubit ordering is Qiskit's little-endian convention, isolated in `pauli_label` and
`bitstring_to_x`, with a round-trip test.

---

## 5. QAOA

**Is the circuit generated from the actual Hamiltonian?** Yes. `h` and `J` are extracted
from the `SparsePauliOp` and laid out as `RZ(2γhᵢ)` and `RZZ(2γJᵢⱼ)`. The circuit shown in
the UI is the bound circuit that was executed.

**Is the fast simulator trustworthy?** It is not trusted — it is checked.
`verify_simulator_agreement` compares the specialised NumPy simulator against Qiskit's own
`Statevector` on the explicit circuit at p=1,2,3; maximum probability deviation 4.6e-16.
This check runs at every problem size in the scaling benchmark.

**Is the classical optimizer minimising the expected energy?** Yes, and the reported
expectation is computed from the unscaled Hamiltonian diagonal. The `γ` rescaling is a pure
reparameterisation; a test asserts that zero parameters give the uniform superposition and
that `⟨H_C⟩` there equals the mean QUBO energy.

**Is a measurement artefact avoided?** Yes, and this was caught during development. On an
ideal simulator every basis state has nonzero amplitude, so reading "best energy" from the
statevector's support returns the global optimum regardless of parameter quality. Best
sampled energy is therefore drawn from a **finite** shot sample.

**Are results compared against exact ground truth?** Yes, at every size. Success
probability is the sum over **all** degenerate optima, asserted by a test to equal the
manual sum over the optimum set.

**Was COBYLA assumed optimal?** No — and it mattered. At `K=3`, p=3: COBYLA reached
0.00024 success probability, Powell reached 0.00001, which is **0.9× uniform random**, i.e.
worse than random sampling.

### The uncomfortable result

Depth does not reliably help. From the scaling benchmark, `p=3` is worse than `p=2` at
N=12, 14 and 18. QAOA's concentration over uniform random grows with problem size (up to
390× at N=18 for p=2), but:

- absolute success probability still **falls** with size (8.3e-3 → 3.7e-4);
- **simulated annealing reaches the exact optimum at every size tested**, gaps 1e-12 to
  1e-14, and QAOA never beats it;
- uniform random sampling, the baseline that matched QAOA in Boulebnane et al. (2022),
  is beaten here — but it also fails outright at N=14 and N=18, so it is a weaker baseline
  at these sizes than that paper's result would suggest.

No quantum advantage is claimed and none is visible.

---

## 6. Hardware

**Are logical and transpiled circuits distinguished?** Yes, everywhere. At 16 qubits, p=3:
logical depth 70 vs transpiled 209; logical two-qubit gates 198 vs transpiled 396.
Reporting only logical depth would understate hardware cost by roughly 3×.

**Are two-qubit gates counted correctly?** This was a real bug, found by a test. Counting
by gate *name* returned **zero** two-qubit gates for circuits transpiled for Aer, which
supports `rzz` natively and therefore contains no `cx`. Counting is now by gate **arity**,
which is basis-independent.

**Are hardware results separated from simulations?** Yes. Noise runs are labelled as Aer
simulations with documented rates and are never presented as device measurements. Hardware
is recorded as `{"status": "not evaluated", "reason": "..."}`.

**Noise degradation (final K=2, p=2, 8192 shots):**

| model | ⟨H_C⟩ | success probability | feasible probability |
|---|---|---|---|
| noiseless | 31.3 | 0.00525 | 0.544 |
| low (1e-3 two-qubit) | 57.0 | 0.00281 | 0.442 |
| representative (1e-2) | 301.4 | 0.00085 | 0.192 |
| high (3e-2) | 982.7 | 0.00049 | 0.036 |
| thermal T1/T2 | 138.6 | 0.00317 | 0.265 |

At representative current-hardware error rates, success probability falls **6×** and
feasible probability falls from 54% to 19%. At high noise only 3.6% of shots land on a
constraint-satisfying state. With 364–570 transpiled two-qubit gates, real-device execution
should be expected to perform near or below the high-noise row.

---

## 7. What would most change the conclusions

Ranked by how much each would move the science, not by effort:

1. **A better hemolysis model.** It beats the mean baseline by 3.5%. Everything about the
   activity/hemolysis trade-off rests on it. More data, or a protein-language-model
   representation, would matter more than any quantum improvement.
2. **Experimental validation.** None exists. This remains the deepest limitation, but the
   *extrapolation* half of it is no longer unmeasured — see section 9 below.
3. **A constraint-preserving mixer.** With an XY- or independent-set-style mixer the state
   stays in the feasible subspace and the penalty terms disappear, removing the large
   coefficient spread that makes this QUBO hard for QAOA.
4. **Better QAOA parameter optimisation.** `p=3` underperforming `p=2` looks like an
   optimisation failure rather than an expressivity limit; more restarts, warm starts, or
   counter-diabatic driving are the obvious tests.
5. **Third-order terms, but only if measured.** The `k=3` fidelity result argues for them.
   They would also make exact enumeration the binding constraint much sooner, and the spec
   is right that adding them silently would be the wrong move.

---

## 9. Mutation extrapolation, measured

The limitation originally recorded here — "models trained on natural peptides, applied to
point mutants" — was an unmeasured caveat. It is now measured directly, on real assay data.

**Method.** DRAMP contains natural analogue series, so the cluster-split test set already
contains pairs of independently assayed peptides differing by 1–3 residues, both withheld
from training. The already-trained model is evaluated on `model(B) − model(A)` against the
true `y_B − y_A`. No synthetic mutants, no retraining, no change to the shipped model.

**The stratification is the finding.** Pairs whose members come from different
publications carry inter-laboratory and inter-organism variation in their true delta. That
is label noise, not model error, and pooling the strata inverts the conclusion for
activity:

| target | stratum | n | Spearman ρ | directional accuracy |
|---|---|---|---|---|
| activity | **same study** | 736 | **+0.216** | **59.4%** |
| activity | cross study | 317 | −0.439 | 31.2% |
| activity | pooled | 1,053 | −0.050 | 49.8% |
| hemolysis | **same study** | 265 | **+0.251** | **58.9%** |
| hemolysis | cross study | 106 | +0.140 | 59.4% |
| hemolysis | pooled | 371 | +0.215 | 59.1% |

Taken naively, the pooled activity figure says the model is a coin flip. That would have
been a wrong and damaging conclusion. The cross-study stratum being *strongly* negative
(ρ = −0.44), rather than merely uninformative, is consistent with shrinkage against
extreme labels: cross-study pairs have roughly twice the true-delta magnitude
(baseline MAE 1.04 vs 0.49), and a model that pulls extreme values toward the mean produces
deltas with the opposite sign when the true delta is driven by an outlier measurement.

**Honest verdict.** Both models carry **weak but real** mutation sensitivity: ~59%
directional accuracy against a 50% chance baseline, ρ ≈ 0.22–0.25. Neither beats a
"predict no change" baseline on MAE, which is expected at this signal level and is why
correlation and direction are the metrics reported.

**What this does and does not license.**

- It does **not** validate the biology. 59% directional accuracy means the Δᵢ values the
  QUBO is built from are informative but noisy; the optimum of that landscape is a
  reasonable shortlist, not a confident prediction.
- It does mean the earlier framing was too pessimistic. The models are not blind to
  mutation effects — the apparent blindness was an artefact of mixing measurement sources.
- Candidates are additionally annotated with nearest-training-set identity (typically
  87–91%, so near-manifold) and bootstrap-ensemble spread, so a reader can see per
  candidate whether the prediction is an interpolation or a reach.

**Residual confounds.** Same-study pairs still differ in target organism in some cases, and
min-MIC aggregation can select different organisms for the two members. The dead zone of
0.1 log units is almost certainly smaller than true assay repeatability, so some pairs
scored for direction have true deltas within noise. Both push the measured accuracy
*down*, so 59% is a conservative estimate.

---

## 8. Claims audit

| claim that could be made | made? | why not |
|---|---|---|
| quantum advantage | **no** | annealing matches or beats QAOA at every size; QAOA runs on a simulator |
| novel application of quantum computing to AMP design | **no** | done and wet-lab validated in 2023 (papers.md §1.1) |
| QAOA is a good solver for this problem | **no** | it never beats simulated annealing here |
| the designed peptide is an antibiotic | **no** | model prediction only; no experiment |
| the QUBO faithfully represents the ML landscape | **only for K ≤ 2** | measured; at k=3 the top pick lands at the 48th percentile |
| the ML models are strong | **no** | hemolysis beats a mean baseline by 3.5% |
| the models can rank mutations reliably | **no** | 59% directional accuracy on real held-out point-mutant pairs |
| the models are blind to mutation effects | **no** | that was the pooled artefact; same-study ρ ≈ 0.22–0.25 |
| the implementation is mathematically correct | **yes** | 164 tests; exhaustive Ising and penalty verification at every size |

The last row is the one this project can actually stand behind.
