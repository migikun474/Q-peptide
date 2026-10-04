"""Mutation landscape: individual and pairwise effects on the biological score.

Definitions (spec sections 12-13), with ``P`` the parent and ``P_i = P (+) m_i``:

    Delta_i  = S(P_i) - S(P)
    Delta_ij = S(P_ij) - S(P_i) - S(P_j) + S(P)

``Delta_ij`` is the discrete mixed second difference. It is zero when the two mutations
act additively, positive when they are synergistic and negative when antagonistic, under
the chosen score. Every value is obtained by building the actual mutant sequence and
running the real property models -- no fitted or heuristic coefficients anywhere.

For conflicting pairs (two substitutions at the same residue) ``P_ij`` does not exist, so
``Delta_ij`` is undefined. Those entries are held as NaN in the matrix and are excluded
from the QUBO objective; their exclusivity is enforced by a constraint penalty instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from backend.optimization.mutations import MutationSet
from backend.utils.peptide import apply_mutations


@dataclass
class MutationLandscape:
    """Computed first- and second-order effects for a mutation set."""
    mutation_set: MutationSet
    parent_score: float
    delta: np.ndarray            # shape (N,)
    delta_pair: np.ndarray       # shape (N, N), symmetric, NaN on conflicts/diagonal
    n_model_evaluations: int
    report: dict

    @property
    def n(self) -> int:
        return self.mutation_set.n

    def compatible_pairs(self) -> list[tuple[int, int]]:
        """Index pairs with a defined interaction term."""
        return [
            (i, j)
            for i, j in combinations(range(self.n), 2)
            if not np.isnan(self.delta_pair[i, j])
        ]

    def surrogate_delta(self, x: np.ndarray) -> float:
        """Second-order surrogate ``sum_i d_i x_i + sum_{i<j} d_ij x_i x_j``.

        Conflicting pairs contribute nothing here; a selection that violates
        exclusivity is handled by the QUBO's penalty term, not by this function.
        """
        x = np.asarray(x, dtype=float)[: self.n]
        total = float(np.dot(self.delta, x))
        for i, j in self.compatible_pairs():
            if x[i] and x[j]:
                total += float(self.delta_pair[i, j])
        return total

    def as_dict(self) -> dict:
        pairs = [
            {
                "i": i,
                "j": j,
                "label_i": self.mutation_set.candidates[i].label,
                "label_j": self.mutation_set.candidates[j].label,
                "delta_ij": float(self.delta_pair[i, j]),
                "interaction": (
                    "synergistic" if self.delta_pair[i, j] > 1e-9
                    else "antagonistic" if self.delta_pair[i, j] < -1e-9
                    else "additive"
                ),
            }
            for i, j in self.compatible_pairs()
        ]
        return {
            "parent": self.mutation_set.parent,
            "parent_score": self.parent_score,
            "n_variables": self.n,
            "individual_effects": [
                {
                    "index": i,
                    "label": c.label,
                    "delta_i": float(self.delta[i]),
                    "beneficial": bool(self.delta[i] > 0),
                }
                for i, c in enumerate(self.mutation_set.candidates)
            ],
            "pairwise_effects": pairs,
            "n_model_evaluations": self.n_model_evaluations,
            "report": self.report,
        }


def compute_landscape(mset: MutationSet, scorer) -> MutationLandscape:
    """Compute all Delta_i and Delta_ij by running the property models.

    Evaluation count is ``1 + N + (number of compatible pairs)``. The scorer memoises, so
    repeated sequences cost nothing extra.
    """
    parent = mset.parent
    n = mset.n

    parent_score = scorer.score_one(parent)

    # --- single mutants ---
    single_seqs = [
        apply_mutations(parent, [(c.position, c.new_aa)]) for c in mset.candidates
    ]
    single_scores = np.asarray(scorer.score(single_seqs), dtype=float)
    delta = single_scores - parent_score

    # --- double mutants, compatible pairs only ---
    delta_pair = np.full((n, n), np.nan, dtype=float)
    pair_list = [
        (i, j) for i, j in combinations(range(n), 2) if mset.are_compatible(i, j)
    ]
    if pair_list:
        double_seqs = [
            apply_mutations(
                parent,
                [
                    (mset.candidates[i].position, mset.candidates[i].new_aa),
                    (mset.candidates[j].position, mset.candidates[j].new_aa),
                ],
            )
            for i, j in pair_list
        ]
        double_scores = np.asarray(scorer.score(double_seqs), dtype=float)
        for (i, j), s_ij in zip(pair_list, double_scores):
            val = s_ij - single_scores[i] - single_scores[j] + parent_score
            delta_pair[i, j] = val
            delta_pair[j, i] = val

    n_evals = 1 + n + len(pair_list)

    finite_pairs = delta_pair[np.isfinite(delta_pair)]
    # each unordered pair appears twice in the symmetric matrix
    report = {
        "n_single_mutants_evaluated": n,
        "n_double_mutants_evaluated": len(pair_list),
        "n_conflict_pairs_skipped": len(mset.conflict_pairs),
        "delta_i": {
            "min": float(delta.min()) if n else 0.0,
            "max": float(delta.max()) if n else 0.0,
            "mean": float(delta.mean()) if n else 0.0,
            "n_beneficial": int(np.sum(delta > 0)),
            "n_deleterious": int(np.sum(delta < 0)),
        },
        "delta_ij": {
            "n_defined": len(pair_list),
            "min": float(finite_pairs.min()) if finite_pairs.size else None,
            "max": float(finite_pairs.max()) if finite_pairs.size else None,
            "mean_abs": float(np.abs(finite_pairs).mean()) if finite_pairs.size else None,
            "n_synergistic": int(np.sum(finite_pairs > 1e-9) // 2),
            "n_antagonistic": int(np.sum(finite_pairs < -1e-9) // 2),
            "n_additive": int(np.sum(np.abs(finite_pairs) <= 1e-9) // 2),
        },
        "interaction_strength_ratio": (
            float(np.abs(finite_pairs).mean() / np.abs(delta).mean())
            if finite_pairs.size and np.abs(delta).mean() > 0
            else None
        ),
        "interpretation": (
            "interaction_strength_ratio compares mean |Delta_ij| to mean |Delta_i|. A "
            "ratio near zero means the landscape is essentially additive and the problem "
            "reduces to independent selection; a larger ratio means the quadratic term "
            "genuinely matters, which is what makes this a QUBO rather than a knapsack"
        ),
    }
    return MutationLandscape(mset, parent_score, delta, delta_pair, n_evals, report)


def verify_landscape_identities(
    landscape: MutationLandscape, scorer, tol: float = 1e-8
) -> dict:
    """Assert the surrogate reproduces the true score for 0, 1 and 2 mutations.

    This is exactness by construction, so a failure means a bug in the landscape or the
    surrogate -- not approximation error. Returns a report; raises nothing, so callers
    can record the result.
    """
    mset = landscape.mutation_set
    n = landscape.n
    failures: list[dict] = []

    # zero mutations
    zero = landscape.surrogate_delta(np.zeros(n))
    if abs(zero) > tol:
        failures.append({"case": "empty_selection", "surrogate": zero, "expected": 0.0})

    # single mutations
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1
        sur = landscape.surrogate_delta(x)
        true = scorer.score_one(mset.apply([i])) - landscape.parent_score
        if abs(sur - true) > tol:
            failures.append(
                {"case": f"single_{mset.candidates[i].label}",
                 "surrogate": sur, "true": true}
            )

    # compatible pairs
    for i, j in landscape.compatible_pairs():
        x = np.zeros(n)
        x[i] = x[j] = 1
        sur = landscape.surrogate_delta(x)
        true = scorer.score_one(mset.apply([i, j])) - landscape.parent_score
        if abs(sur - true) > tol:
            failures.append(
                {
                    "case": f"pair_{mset.candidates[i].label}_{mset.candidates[j].label}",
                    "surrogate": sur,
                    "true": true,
                }
            )

    return {
        "tolerance": tol,
        "n_cases_checked": 1 + n + len(landscape.compatible_pairs()),
        "n_failures": len(failures),
        "passed": not failures,
        "failures": failures[:20],
        "note": (
            "the second-order surrogate is exact by construction for |x| <= 2; a failure "
            "here indicates an implementation bug, not approximation error"
        ),
    }


def enumerate_feasible_selections(
    landscape: MutationLandscape, k: int, limit: int = 20000
) -> list[tuple[int, ...]] | None:
    """All feasible selections of exactly ``k`` mutations, or None if there are too many.

    Exhaustive enumeration is preferred over sampling whenever it is affordable: at these
    problem sizes the count is often in the hundreds, and an exhaustive answer removes
    sampling error from what is a headline validity measurement.
    """
    n = landscape.n
    if k > n:
        return []
    conflict = set(landscape.mutation_set.conflict_pairs)
    out: list[tuple[int, ...]] = []
    for sel in combinations(range(n), k):
        if any((a, b) in conflict for a, b in combinations(sel, 2)):
            continue
        out.append(sel)
        if len(out) > limit:
            return None
    return out


def surrogate_fidelity_exhaustive(
    landscape: MutationLandscape, scorer, k: int
) -> dict:
    """Exhaustive surrogate-vs-truth comparison over ALL feasible k-mutation selections.

    Beyond the usual error metrics this reports the **decision-relevant** quantity: where
    the surrogate's own top-ranked selection actually sits in the true ranking. A model
    can have a respectable RMSE and still be useless for choosing, which is exactly what
    an optimizer does with it.
    """
    sels = enumerate_feasible_selections(landscape, k)
    if sels is None:
        return {"status": "not evaluated",
                "reason": f"too many feasible {k}-selections to enumerate"}
    if len(sels) < 3:
        return {"status": "not evaluated",
                "reason": f"only {len(sels)} feasible {k}-selections"}

    mset = landscape.mutation_set
    sequences = [mset.apply(list(s)) for s in sels]
    true = np.asarray(scorer.score(sequences), dtype=float) - landscape.parent_score

    sur = np.empty(len(sels), dtype=float)
    for idx, sel in enumerate(sels):
        x = np.zeros(landscape.n)
        x[list(sel)] = 1
        sur[idx] = landscape.surrogate_delta(x)

    resid = sur - true
    out = {
        "status": "evaluated",
        "mode": "exhaustive",
        "k": k,
        "n_selections": len(sels),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs_error": float(np.max(np.abs(resid))),
        "mean_bias": float(np.mean(resid)),
        "true_delta_std": float(np.std(true)),
        "n_distinct_true_values": int(len(np.unique(np.round(true, 9)))),
    }
    spread = out["true_delta_std"]
    out["normalised_rmse"] = float(out["rmse"] / spread) if spread > 1e-12 else None

    if np.std(sur) > 1e-12 and spread > 1e-12:
        from scipy.stats import pearsonr, spearmanr

        out["pearson_r"] = float(pearsonr(sur, true)[0])
        out["spearman_rho"] = float(spearmanr(sur, true)[0])
    else:
        out["pearson_r"] = None
        out["spearman_rho"] = None

    # decision-relevant: how good is the surrogate's argmax, in truth?
    sur_best = int(np.argmax(sur))
    order = np.argsort(-true)
    rank = int(np.where(order == sur_best)[0][0]) + 1
    out["surrogate_top_pick"] = {
        "mutations": [mset.candidates[i].label for i in sels[sur_best]],
        "surrogate_delta": float(sur[sur_best]),
        "true_delta": float(true[sur_best]),
        "true_rank": rank,
        "out_of": len(sels),
        "percentile": float(100.0 * (1.0 - (rank - 1) / max(len(sels) - 1, 1))),
    }
    true_best = int(np.argmax(true))
    out["true_best"] = {
        "mutations": [mset.candidates[i].label for i in sels[true_best]],
        "true_delta": float(true[true_best]),
        "surrogate_delta": float(sur[true_best]),
    }
    out["interpretation"] = (
        "surrogate_top_pick.true_rank is the metric that matters for optimisation: it is "
        "where the selection the QUBO considers best actually lands in the true ML "
        "ranking. A rank near 1 means the surrogate is a usable proxy; a rank near "
        "out_of means it is actively misleading at this mutation count."
    )
    return out


def validate_surrogate_beyond_pairs(
    landscape: MutationLandscape,
    scorer,
    min_mutations: int = 3,
    max_mutations: int = 6,
    n_samples: int = 300,
    seed: int = 0,
    budget: int | None = None,
) -> dict:
    """Measure surrogate error where it is NOT exact: 3 or more mutations.

    This is the experiment in spec sections 15 and 37. Feasible selections (respecting
    same-position exclusivity, and the budget if given) are sampled, and the quadratic
    surrogate is compared against the true ML score difference.

    A poor result here is a finding to report, not a reason to add higher-order terms
    silently.
    """
    rng = np.random.default_rng(seed)
    mset = landscape.mutation_set
    n = landscape.n
    conflict = set(mset.conflict_pairs)

    hi = min(max_mutations, n)
    if budget is not None:
        hi = min(hi, budget)
    if hi < min_mutations:
        return {
            "status": "not evaluated",
            "reason": (
                f"cannot sample {min_mutations}+ mutation selections: effective maximum "
                f"is {hi} (N={n}, budget={budget})"
            ),
        }

    def feasible(sel: list[int]) -> bool:
        for a, b in combinations(sorted(sel), 2):
            if (a, b) in conflict:
                return False
        return True

    seen: set[tuple[int, ...]] = set()
    samples: list[list[int]] = []
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 60:
        attempts += 1
        k = int(rng.integers(min_mutations, hi + 1))
        sel = sorted(rng.choice(n, size=k, replace=False).tolist())
        key = tuple(sel)
        if key in seen or not feasible(sel):
            continue
        seen.add(key)
        samples.append(sel)

    if not samples:
        return {
            "status": "not evaluated",
            "reason": "no feasible multi-mutation selections could be sampled",
        }

    sequences = [mset.apply(sel) for sel in samples]
    true_scores = np.asarray(scorer.score(sequences), dtype=float)
    true_delta = true_scores - landscape.parent_score

    sur_delta = np.empty(len(samples))
    for k, sel in enumerate(samples):
        x = np.zeros(n)
        x[sel] = 1
        sur_delta[k] = landscape.surrogate_delta(x)

    resid = sur_delta - true_delta
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    spread = float(np.std(true_delta))

    out = {
        "status": "evaluated",
        "n_samples": len(samples),
        "mutations_per_sample": {
            "min": int(min(len(s) for s in samples)),
            "max": int(max(len(s) for s in samples)),
        },
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": float(np.max(np.abs(resid))),
        "mean_bias": float(np.mean(resid)),
        "true_delta_std": spread,
        "normalised_rmse": float(rmse / spread) if spread > 1e-12 else None,
        "note": (
            "normalised_rmse is RMSE divided by the spread of the true score differences; "
            "a value well below 1 means the surrogate explains most of the variation, "
            "around or above 1 means it does not"
        ),
    }
    if len(samples) >= 3 and np.std(sur_delta) > 1e-12 and spread > 1e-12:
        from scipy.stats import pearsonr, spearmanr
        out["pearson_r"] = float(pearsonr(sur_delta, true_delta)[0])
        out["spearman_rho"] = float(spearmanr(sur_delta, true_delta)[0])
    else:
        out["pearson_r"] = None
        out["spearman_rho"] = None

    # error growth with mutation count -- does the approximation degrade?
    by_k: dict[str, dict] = {}
    counts = np.array([len(s) for s in samples])
    for k in sorted(set(counts.tolist())):
        m = counts == k
        by_k[str(int(k))] = {
            "n": int(m.sum()),
            "mae": float(np.mean(np.abs(resid[m]))),
            "rmse": float(np.sqrt(np.mean(resid[m] ** 2))),
        }
    out["error_by_mutation_count"] = by_k
    return out
