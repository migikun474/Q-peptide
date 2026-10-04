# Q-Peptide — Annotated Bibliography

Every entry was retrieved and read (abstract + relevant sections) during the literature
audit. Where a claim below is a direct quote it is marked as such. Entries whose
bibliographic details could not be verified from a primary source are explicitly flagged
`[METADATA UNVERIFIED]` rather than guessed.

Legend for **Relevance**:
- **A** — directly shapes the Q-Peptide formulation
- **B** — supports a methodological choice
- **C** — context / background

---

## 1. Quantum optimization applied to peptide design

### 1.1 Tučs et al. 2023 — Quantum annealing for nonhemolytic AMP design

| Field | Value |
|---|---|
| **Title** | Quantum Annealing Designs Nonhemolytic Antimicrobial Peptides in a Discrete Latent Space |
| **Authors** | Tučs, A.; Berenger, F.; Yumoto, A.; Tamura, R.; Uzawa, T.; Tsuda, K. |
| **Year** | 2023 |
| **Venue** | ACS Medicinal Chemistry Letters, 14(5), 577–582 |
| **DOI** | [10.1021/acsmedchemlett.2c00487](https://doi.org/10.1021/acsmedchemlett.2c00487) |
| **Relevance** | **A — closest prior work** |

**Problem.** Multi-objective de novo AMP design: maximise predicted antimicrobial activity
while minimising predicted hemolysis.

**Method.** A binary variational autoencoder (bVAE) compresses peptide sequences into a
**64-bit discrete latent space**. ML predictors score activity and hemolysis; the two
objectives are merged by **non-dominated sorting**, with the Pareto-front rank `r` mapped to
a scalar score (`y = −1/r` for `r ≤ t`, else `y = −10`). A **factorization machine (FM)** is
then fit to (latent bitvector → score) pairs; because the FM is quadratic in its binary
inputs it *is* a QUBO. A **D-Wave Advantage (system 4.1)** annealer minimises the FM
surrogate, and the resulting bitstrings are decoded back to sequences by the bVAE decoder.

**Mathematical formulation.** FM surrogate
`ŷ(x) = w₀ + Σᵢ wᵢxᵢ + Σ_{i<j} ⟨vᵢ, vⱼ⟩ xᵢxⱼ`, with `xᵢ ∈ {0,1}` the latent bits. The
quadratic coefficient is the inner product of learned latent factor vectors, giving a dense
QUBO directly solvable by annealing.

**Results.** ~200,000 peptides designed; 4 synthesised; 3 showed high antimicrobial
activity, 2 were non-hemolytic.

**Relevance to Q-Peptide.** This is the work Q-Peptide is closest to, and the reason
Q-Peptide **must not claim novelty for "quantum optimization of nonhemolytic AMPs"** — that
is done and wet-lab validated. The differences that remain defensible:

| | Tučs et al. 2023 | Q-Peptide |
|---|---|---|
| Variables | 64 bVAE latent bits | binary indicators for explicit `(position, aa_from, aa_to)` mutations |
| Interpretability | latent bits have no biochemical meaning | every variable is a named substitution |
| QUBO source | FM *fit* to score samples (learned, approximate) | coefficients **computed exactly** as finite differences of the ML score |
| Hardware paradigm | quantum annealing (D-Wave) | gate-based QAOA (Qiskit / IBM) |
| Baselines | annealer only | exact enumeration + simulated annealing on the *identical* QUBO |
| Surrogate error | not quantified | measured explicitly against the true ML landscape (≥3 mutations) |

**Limitations (authors' own).** "The quantum annealer does not always optimize the objective
function completely"; "the accuracy of the predictors should be improved as much as
possible." No exact-optimum baseline is reported, so the annealer's optimality gap is
unknown — a gap Q-Peptide closes by brute-force enumeration at small `N`.

---

### 1.2 Boulebnane et al. 2022/2023 — QAOA for peptide conformational sampling

| Field | Value |
|---|---|
| **Title** | Peptide conformational sampling using the Quantum Approximate Optimization Algorithm |
| **Authors** | Boulebnane, S.; Lucas, X.; Meyder, A.; Adaszewski, S.; Montanaro, A. |
| **Year** | 2022 (preprint); published in npj Quantum Information (2023) |
| **Venue** | arXiv:2204.01821; npj Quantum Information 9, 70 (2023) |
| **DOI** | [10.48550/arXiv.2204.01821](https://doi.org/10.48550/arXiv.2204.01821) |
| **Relevance** | **A — the central cautionary result** |

**Problem.** Simplified lattice protein folding — sampling low-energy conformations of short
peptides. **Not** sequence design.

**Method.** Lattice self-avoiding-walk encoding, QAOA simulated on up to **20 qubits**,
benchmarked first on self-avoiding walks then on folding with a simplified potential.

**Results.** Direct quote: *"the performance of QAOA can be matched by random sampling up to
a small overhead"*, and deep circuits are needed for accuracy.

**Conclusion (direct quote).** *"these results cast serious doubt on the ability of QAOA to
address the protein folding problem in the near term, even in an extremely simplified
setting."*

**Relevance to Q-Peptide.** This is the single most important paper for calibrating
Q-Peptide's claims. It establishes that (a) **random sampling is a mandatory baseline** — if
QAOA is not beaten by uniform random bitstring sampling, that must be reported; and (b) any
claim of QAOA usefulness on a peptide problem requires comparison against exact optima, not
just self-consistency. Q-Peptide therefore treats QAOA as an *object of study* ("how close
does it get?") rather than as a solver assumed to be good.

---

### 1.3 Related quantum-biology optimization work

These were retrieved in the audit and inform context. Bibliographic details were taken from
the arXiv listing pages.

| Paper | arXiv | Relevance | Note |
|---|---|---|---|
| De Novo Design of Protein-Binding Peptides by Quantum Computing | 2503.05458 | B | Peptide design cast as quantum optimization; binder-focused rather than AMP. |
| Protein Design by Integrating Machine Learning with Quantum Annealing and Quantum-inspired Optimization | 2407.07177 | **A** | Same ML→QUBO→annealer architecture as Q-Peptide; supports the surrogate-coefficient approach. |
| Using quantum annealing to design lattice proteins | 2402.09069 | C | Lattice model, not sequence-space. |
| Designing lattice proteins with variational quantum algorithms | 2508.02369 | B | Gate-based VQA for protein design. |
| Penalty-free quantum optimization applied to lattice protein folding | 2606.02104 | B | Constraint-preserving mixers as an alternative to penalty terms — a documented alternative to Q-Peptide's X-mixer + penalty choice. |
| Reduced-Alphabet QUBO/Ising Formulation for Constraint-Driven Cyclic Peptide Sequence Design | 2606.23253 | **A** | Closest formulation analogue: one-hot residue/alphabet variables with conflict constraints. |
| Peptide Structure Prediction Using Counter-Diabatic QAOA (CD-QAOA) | 2606.01611 | B | Counter-diabatic driving improves on plain QAOA; a named future direction, not implemented in the MVP. |

> `[METADATA UNVERIFIED]` — for the 2606.x and 2602.x preprints above only the arXiv
> identifier, title and abstract were confirmed; author lists and any journal publication
> were not individually verified, so they are cited by identifier only and no author
> attribution is made.

---

## 2. AMP and hemolysis machine learning

### 2.1 Plisson, Ramírez-Sánchez & Martínez-Hernández 2020 — non-hemolytic peptide ML

| Field | Value |
|---|---|
| **Title** | Machine learning-guided discovery and design of non-hemolytic peptides |
| **Year** | 2020 |
| **Venue** | Scientific Reports 10, 16581 |
| **DOI** | [10.1038/s41598-020-73644-6](https://doi.org/10.1038/s41598-020-73644-6) |
| **Relevance** | **A** |

Establishes the standard supervised setup for hemolysis prediction from sequence-derived
physicochemical descriptors, and the practice of treating hemolysis as a **separate model**
from activity rather than a single multi-task head. Q-Peptide follows this two-model design.

### 2.2 ConsAMPHemo 2025 — classification *and* regression of hemolysis

| Field | Value |
|---|---|
| **Title** | ConsAMPHemo: A computational framework for predicting hemolysis of antimicrobial peptides based on machine learning approaches |
| **Year** | 2025 |
| **Venue** | (PMC12168091) |
| **Link** | https://pmc.ncbi.nlm.nih.gov/articles/PMC12168091/ |
| **Relevance** | **A** |

Does **both** binary hemolysis classification and regression onto hemolysis concentration.
Reports accuracies of 99.54% / 82.57% / 88.04% across three datasets — the spread across
datasets is itself the lesson: a single headline accuracy is not meaningful for this task,
and near-99% figures are a red flag for redundancy between train and test splits.

**Relevance to Q-Peptide.** Directly supports spec §6: do not force continuous hemolysis
measurements into binary labels when a regression target exists. Also motivates reporting
per-dataset metrics and leakage controls rather than one number.

### 2.3 HemoPI dataset family

| Field | Value |
|---|---|
| **Datasets** | HemoPI-1, HemoPI-2, HemoPI-3 (plus RNN-Hem, HLPpred-Fuse, AMP-Combined used by later work) |
| **Relevance** | **A — candidate hemolysis training data** |

HemoPI-1 = 552 experimentally verified hemolytic peptides from the **Hemolytik** database as
positives, plus an equal number of **random Swiss-Prot-derived** sequences as negatives.

**Critical limitation for Q-Peptide.** Negatives generated by random sampling from Swiss-Prot
are trivially separable from curated hemolytic peptides on composition alone. A model trained
on HemoPI-1 therefore risks learning "is this a real curated peptide?" rather than "is this
hemolytic?". This is recorded as a **known limitation** and is the reason Q-Peptide prefers
datasets whose negatives are experimentally measured non-hemolytic peptides where available,
and reports the negative-set provenance explicitly.

### 2.4 Other ML-for-AMP references

| Paper | Link / DOI | Relevance |
|---|---|---|
| Machine Learning Prediction of Antimicrobial Peptides (review) | PMC9126312 | B — survey of feature sets and model families |
| Enhanced prediction of hemolytic activity in AMPs using deep learning | [10.1186/s12859-024-05983-4](https://doi.org/10.1186/s12859-024-05983-4) | B — DL baseline; shows XGBoost-on-descriptors is competitive |
| HemPepPred (quantitative hemolytic activity, PLM features) | PMC12692575 | B — regression framing + protein-language-model features |
| ToxTeller | PMC11270677 | C — peptide toxicity, four ML approaches |
| MoFormer | arXiv:2406.02610 | C — generative multi-objective AMP design (contrast: generative vs. our constrained-mutation framing) |

---

## 3. Data leakage and evaluation for peptide ML

### 3.1 DataSAIL — similarity-aware data splitting

| Field | Value |
|---|---|
| **Title** | Data splitting against information leakage with DataSAIL |
| **Venue** | (PMC12905371 — addendum) |
| **Relevance** | **A** |

Formalises splitting as an optimisation problem that minimises train/test similarity.
Confirms that **random splits are invalid** for biological sequence data.

### 3.2 CD-HIT and identity-threshold clustering

Established practice, confirmed across retrieved sources:

- CD-HIT performs greedy incremental clustering at a user-set identity threshold, keeping the
  longest sequence as cluster representative.
- **AMP studies commonly cluster at 70% identity** and split at the **cluster** level so a
  cluster never spans train and test.
- Protein hazard-screening work clusters at **≤40% identity** with 80/20 cluster-level
  stratified assignment.

**Relevance to Q-Peptide.** Q-Peptide implements cluster-level splitting with the cluster as
the grouping unit (`GroupShuffleSplit` / `StratifiedGroupKFold` on cluster id), and reports
the identity threshold used. Because CD-HIT is a non-Python external binary, the
implementation uses an equivalent in-process greedy identity clustering (documented in
`literature_review.md` §4) so the pipeline stays reproducible without system packages.

### 3.3 PepBenchmark and k-mer leakage

| Field | Value |
|---|---|
| **Title** | PepBenchmark: A Standardized Benchmark for Peptide Machine Learning |
| **arXiv** | 2604.10531 `[METADATA UNVERIFIED]` |
| **Relevance** | **B** |

Documents **k-mer leakage**: specific k-mers occur at very high frequency among positives and
are dataset-specific, so dipeptide-composition features can encode dataset identity. Because
Q-Peptide *uses* dipeptide composition (spec §8), this is an explicit threat to validity and
is recorded as such.

---

## 4. QUBO constraint encoding

### 4.1 Lucas 2014 — Ising formulations of NP problems

| Field | Value |
|---|---|
| **Title** | Ising formulations of many NP problems |
| **Author** | Lucas, A. |
| **Year** | 2014 |
| **Venue** | Frontiers in Physics 2, 5 |
| **DOI** | [10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005) |
| **Relevance** | **A — canonical reference** |

The standard reference for the `x = (1 − Z)/2` binary↔spin substitution and for penalty-based
constraint encoding. Cited for Q-Peptide's Ising mapping and for the principle that penalty
weights must exceed the largest achievable objective gain from violating the constraint.

### 4.2 Montañez-Barrera & Michielsen — unbalanced penalization

| Field | Value |
|---|---|
| **Title** | Unbalanced penalization: a new approach to encode inequality constraints of combinatorial problems for quantum optimization algorithms |
| **Year** | 2024 |
| **Venue** | Quantum Science and Technology 9, 025022 |
| **DOI** | [10.1088/2058-9565/ad35e4](https://doi.org/10.1088/2058-9565/ad35e4) |
| **arXiv** | 2211.13914 |
| **Relevance** | **A** |

Shows that slack-variable encoding of inequalities inflates qubit count and degrades
performance, and proposes a slack-free penalty that approximates the inequality.

**Relevance to Q-Peptide.** Directly bears on spec §17. Q-Peptide implements the
**slack-variable** encoding as the mathematically exact default (because exactness is the
stated priority and `N ≈ 12–20` keeps the overhead affordable), while documenting unbalanced
penalization as the qubit-efficient alternative. Slack bound and binary width follow the
standard construction: `s ∈ [0, K]`, width `⌈log₂(K+1)⌉`.

### 4.3 Supporting constraint-encoding literature

| Paper | arXiv | Relevance | Note |
|---|---|---|---|
| Improving Performance in Combinatorial Optimization Problems with Inequality Constraints (D-Wave Advantage) | 2305.18757 | B | Empirical evidence that slack encodings underperform at scale. |
| Implementing Slack-Free Custom Penalty Function for QUBO on Gate-Based Quantum Computers | 2504.12611 `[METADATA UNVERIFIED]` | B | Gate-based analogue of the above. |
| Cutting Slack: Quantum Optimization with Slack-Free Methods | 2507.12159 `[METADATA UNVERIFIED]` | C | Survey of slack-free methods. |
| Qubit-Efficient QUBO Formulation for Constrained Optimization | 2509.08080 `[METADATA UNVERIFIED]` | C | Qubit-count reduction. |

---

## 5. AMP mutational engineering (the biology that justifies the variable set)

### 5.1 Dathe et al. 2001 — charge optimisation of magainin

| Field | Value |
|---|---|
| **Title** | Optimization of the antimicrobial activity of magainin peptides by modification of charge |
| **Authors** | Dathe, M.; Nikolenko, H.; Meyer, J.; Beyermann, M.; Bienert, M. |
| **Year** | 2001 |
| **Venue** | FEBS Letters 501(2–3), 146–150 |
| **DOI** | [10.1016/S0014-5793(01)02648-5](https://doi.org/10.1016/S0014-5793(01)02648-5) |
| **Relevance** | **A — the biological basis of the activity/hemolysis trade-off** |

Magainin II amide analogues spanning net charge +3 to +7. Findings:

- Activity and selectivity improve with charge **up to a threshold of about +5**.
- Pushing charge **beyond +5** while keeping other motifs causes *"a dramatic increase of
  hemolytic activity and loss of antimicrobial selectivity."*
- Selectivity can be **restored by reducing hydrophobicity** of the hydrophobic helix face.

**Relevance to Q-Peptide.** This is the empirical justification for three design decisions:

1. The objective **must** be two-term (activity *and* hemolysis). Optimising activity alone
   reproduces the known failure mode of over-cationic, hemolytic analogues.
2. The activity/hemolysis response to charge is **non-monotonic with a threshold**, which is
   exactly the kind of structure a purely additive (first-order) model cannot represent —
   independent biological motivation for the `Δᵢⱼ` pairwise terms.
3. Charge and hydrophobicity **interact** (hydrophobicity reduction rescues a charge-driven
   selectivity loss). A charge-increasing and a hydrophobicity-reducing substitution are
   therefore expected to be **synergistic**, i.e. `Δᵢⱼ ≠ 0`. This is a falsifiable prediction
   the mutation-landscape experiment can check.

### 5.2 Further mutational-engineering references

| Paper | Link / DOI | Relevance |
|---|---|---|
| Rational AMP engineering through amino acid scanning and targeted point mutations | [S0196978126000239](https://www.sciencedirect.com/science/article/abs/pii/S0196978126000239) | **A** — uses 7–12 single-residue mutations per position and 1–9 changes per variant, which brackets Q-Peptide's `N ≈ 12–20` / `K ≈ 3–4` regime in real experimental practice |
| Synthetic magainin analogues with improved antimicrobial activity | [0014579388800772](https://www.sciencedirect.com/science/article/pii/0014579388800772) | B — origin of the pexiganan/MSI-78 line |
| "Specificity Determinants" Improve Therapeutic Indices of Piscidin 1 and Dermaseptin S4 | PMC4014698 | **A** — therapeutic-index improvement by targeted substitution |
| Design of Protegrin-1 Analogs with Improved Antibacterial Selectivity | PMC10458893 | B — selectivity-driven analogue design |
| Innovative Strategies and Methodologies in AMP Design (review) | [10.3390/jfb15110320](https://doi.org/10.3390/jfb15110320) | C — design-strategy survey |

---

## 6. Data sources

### 6.1 DRAMP — primary dataset choice

| Field | Value |
|---|---|
| **Title** | DRAMP 4.0: an open-access data repository dedicated to the clinical translation of antimicrobial peptides |
| **Year** | 2025 |
| **Venue** | Nucleic Acids Research 53(D1), D403–… |
| **Link** | https://academic.oup.com/nar/article/53/D1/D403/7889245 |
| **Download** | http://dramp.cpu-bioinfor.org/downloads/ |
| **Licence** | **CC BY 4.0** |
| **Relevance** | **A — chosen primary source** |

Open access, explicit CC BY 4.0 licence, bulk download in FASTA/xlsx/txt, ~half of general
entries carry **MIC values with target organisms**. Earlier versions: DRAMP 1.0
([10.1038/srep24482](https://doi.org/10.1038/srep24482)), DRAMP 2.0
([10.1038/s41597-019-0154-y](https://doi.org/10.1038/s41597-019-0154-y)), DRAMP 3.0.

**Why chosen.** The licence permits redistribution of processed derivatives, the MIC fields
support a regression target (spec §5), and bulk download makes the pipeline reproducible
without registration.

### 6.2 DBAASP v3

| Field | Value |
|---|---|
| **Title** | DBAASP v3: Database of antimicrobial/cytotoxic activity and structure of peptides as a resource for development of new therapeutics |
| **Year** | 2021 |
| **Venue** | Nucleic Acids Research 49(D1) |
| **Relevance** | **A — secondary source; the hemolysis/cytotoxicity source** |

>15,000 entries with activity annotations and a documented REST API. Crucially it records
**cytotoxic/hemolytic activity against erythrocytes** alongside antimicrobial MIC, which is
what makes a *paired* activity-and-hemolysis objective possible from one provenance.

### 6.3 UniProt — negative-set source

| Field | Value |
|---|---|
| **Resource** | UniProtKB REST API, `https://rest.uniprot.org` |
| **Access** | public, no API key |
| **Relevance** | **B** |

Used only where a non-AMP background set is required. Keyword `KW-0929` ("Antimicrobial")
identifies AMP-annotated entries; its complement supplies background sequences. Per §2.3 the
limitations of random-background negatives are recorded explicitly wherever such negatives
are used.

---

## 7. Quantum software

| Resource | Link | Relevance |
|---|---|---|
| Qiskit documentation | https://quantum.cloud.ibm.com/docs | **A** — API ground truth; V2 primitives (`StatevectorSampler`, `StatevectorEstimator`), `transpile`, `QiskitRuntimeService` |
| Qiskit Patterns (Map / Optimize / Execute / Post-process) | https://quantum.cloud.ibm.com/docs/en/guides/intro-to-patterns | **A** — the workflow structure the implementation follows |
| `qiskit.quantum_info.SparsePauliOp` | Qiskit API reference | **A** — cost-Hamiltonian representation |
| Qiskit Aer (`AerSimulator`, `NoiseModel`) | Qiskit Aer docs | **A** — noise study |

Exact installed versions are recorded in `results/experiments/environment.json` at run time;
no version numbers are asserted in this document.

---

## 8. What the literature does *not* support

Recorded here so the project cannot drift into unsupported claims:

1. **No quantum advantage for peptide design has been demonstrated.** §1.2 found QAOA matched
   by random sampling on a peptide problem. Q-Peptide may not claim advantage.
2. **Quantum optimization of nonhemolytic AMPs is not novel** (§1.1, 2023, wet-lab
   validated). Q-Peptide's defensible contributions are the *exactly-computed* mutation-space
   QUBO, the measured surrogate error, and the exact-optimum benchmarking — not the
   application.
3. **No experimental validation is claimed.** All outputs are model predictions.
4. **Headline accuracies near 99% in this field are not reproducible across datasets** (§2.2)
   and often reflect negative-set artefacts (§2.3). Q-Peptide reports leakage-controlled
   metrics and names the negative-set provenance.
