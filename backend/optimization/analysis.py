"""Independent re-scoring of candidates, and Pareto analysis.

Independent re-scoring (spec section 38). Every candidate a solver returns is scored
again by running the original property models on the reconstructed sequence. The QUBO
surrogate score and the direct ML score are stored side by side, so surrogate error is
visible per candidate rather than only in aggregate.

Pareto analysis (spec section 39). A scalar ``alpha``/``beta`` weighting encodes one
trade-off preference. The Pareto front over ``(A_tilde, -H_tilde)`` is reported as well,
using strict dominance, so no candidate is called "best" without naming the objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.optimization.qubo import QuboProblem


@dataclass
class Candidate:
    """A decoded, independently re-scored candidate."""
    sequence: str
    mutations: list[str]
    n_mutations: int
    selected_indices: list[int]
    feasible: bool
    qubo_energy: float
    surrogate_delta_score: float
    # direct ML re-scoring
    activity_raw: float
    hemolysis_raw: float
    activity_norm: float
    hemolysis_norm: float
    direct_score: float
    direct_delta_score: float
    similarity_to_parent: float
    source: str
    probability: float | None = None
    uncertainty: dict | None = None
    # How far this mutant sits from anything the models were trained on, and how much the
    # bootstrap ensemble disagrees about it. Both are diagnostics on trustworthiness, not
    # predictions: a candidate far from training data with wide ensemble spread deserves
    # much less confidence than its point estimate suggests.
    nearest_training_identity: dict | None = None
    prediction_spread: dict | None = None

    @property
    def surrogate_error(self) -> float:
        """Surrogate minus truth, for this candidate."""
        return self.surrogate_delta_score - self.direct_delta_score

    def as_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "mutations": self.mutations,
            "n_mutations": self.n_mutations,
            "selected_indices": self.selected_indices,
            "feasible": self.feasible,
            "qubo_energy": self.qubo_energy,
            "surrogate_delta_score": self.surrogate_delta_score,
            "direct_ml_activity": self.activity_raw,
            "direct_ml_hemolysis": self.hemolysis_raw,
            "activity_normalized": self.activity_norm,
            "hemolysis_normalized": self.hemolysis_norm,
            "direct_ml_score": self.direct_score,
            "direct_ml_delta_score": self.direct_delta_score,
            "surrogate_error": self.surrogate_error,
            "similarity_to_parent": self.similarity_to_parent,
            "source": self.source,
            "probability": self.probability,
            "uncertainty": self.uncertainty,
            "nearest_training_identity": self.nearest_training_identity,
            "prediction_spread": self.prediction_spread,
        }


def sequence_similarity(a: str, b: str) -> float:
    """Fraction of identical positions for equal-length sequences."""
    if len(a) != len(b):
        from backend.utils.peptide import sequence_identity

        return sequence_identity(a, b)
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def rescore_candidates(
    problem: QuboProblem,
    bitstrings: list[np.ndarray],
    scorer,
    source: str,
    probabilities: list[float] | None = None,
    include_infeasible: bool = False,
) -> list[Candidate]:
    """Decode bitstrings and re-score the feasible ones with the real models.

    Duplicate sequences are collapsed, keeping the highest probability, because several
    bitstrings can decode to the same mutant only if they select the same mutations --
    and because an unfiltered QAOA distribution contains many repeats.
    """
    parent = problem.landscape.mutation_set.parent
    parent_breakdown = scorer.breakdown([parent])
    parent_score = float(parent_breakdown.score[0])

    decoded: list[tuple[dict, float | None]] = []
    for k, x in enumerate(bitstrings):
        d = problem.decode(np.asarray(x))
        if not d["feasible"] and not include_infeasible:
            continue
        prob = probabilities[k] if probabilities is not None else None
        decoded.append((d, prob))

    # collapse duplicates by sequence, summing probability mass
    merged: dict[str, tuple[dict, float | None]] = {}
    for d, prob in decoded:
        key = d["sequence"] if d["sequence"] is not None else f"__infeasible_{id(d)}"
        if key in merged:
            prev_d, prev_p = merged[key]
            total = None
            if prev_p is not None or prob is not None:
                total = (prev_p or 0.0) + (prob or 0.0)
            # keep the lower-energy representative
            keep = prev_d if prev_d["qubo_energy"] <= d["qubo_energy"] else d
            merged[key] = (keep, total)
        else:
            merged[key] = (d, prob)

    items = list(merged.values())
    sequences = [d["sequence"] for d, _ in items if d["sequence"] is not None]
    breakdown = scorer.breakdown(sequences) if sequences else None

    out: list[Candidate] = []
    seq_cursor = 0
    for d, prob in items:
        if d["sequence"] is None:
            continue
        i = seq_cursor
        seq_cursor += 1
        assert breakdown is not None
        out.append(
            Candidate(
                sequence=d["sequence"],
                mutations=d["mutations"],
                n_mutations=d["n_mutations"],
                selected_indices=d["selected_indices"],
                feasible=d["feasible"],
                qubo_energy=d["qubo_energy"],
                surrogate_delta_score=d["surrogate_delta_score"],
                activity_raw=float(breakdown.activity_raw[i]),
                hemolysis_raw=float(breakdown.hemolysis_raw[i]),
                activity_norm=float(breakdown.activity_norm[i]),
                hemolysis_norm=float(breakdown.hemolysis_norm[i]),
                direct_score=float(breakdown.score[i]),
                direct_delta_score=float(breakdown.score[i]) - parent_score,
                similarity_to_parent=sequence_similarity(d["sequence"], parent),
                source=source,
                probability=prob,
            )
        )
    out.sort(key=lambda c: -c.direct_delta_score)
    return out


def surrogate_error_summary(candidates: list[Candidate]) -> dict:
    """Aggregate surrogate-vs-truth error over re-scored candidates."""
    if not candidates:
        return {"status": "not evaluated", "reason": "no candidates"}
    sur = np.array([c.surrogate_delta_score for c in candidates])
    true = np.array([c.direct_delta_score for c in candidates])
    resid = sur - true
    out = {
        "status": "evaluated",
        "n_candidates": len(candidates),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs_error": float(np.max(np.abs(resid))),
        "mean_bias": float(np.mean(resid)),
        "true_delta_std": float(np.std(true)),
    }
    if len(candidates) >= 3 and np.std(sur) > 1e-12 and np.std(true) > 1e-12:
        from scipy.stats import pearsonr, spearmanr

        out["pearson_r"] = float(pearsonr(sur, true)[0])
        out["spearman_rho"] = float(spearmanr(sur, true)[0])
    else:
        out["pearson_r"] = None
        out["spearman_rho"] = None
    return out


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------
def pareto_front(points: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of a maximisation problem.

    ``points`` has shape (M, D) and ALL objectives are maximised. Row ``i`` dominates
    row ``j`` iff ``points[i] >= points[j]`` component-wise and strictly greater in at
    least one component. Returned indices are the rows dominated by nobody.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.array([], dtype=int)
    m = points.shape[0]
    dominated = np.zeros(m, dtype=bool)
    for i in range(m):
        if dominated[i]:
            continue
        # rows at least as good everywhere and strictly better somewhere
        ge = np.all(points >= points[i] - 1e-12, axis=1)
        gt = np.any(points > points[i] + 1e-12, axis=1)
        if np.any(ge & gt):
            dominated[i] = True
    return np.flatnonzero(~dominated)


def pareto_analysis(candidates: list[Candidate], parent_point: dict) -> dict:
    """Pareto front over (normalised activity, negated normalised hemolysis).

    Both coordinates are maximised: higher normalised activity is better, and hemolysis
    is negated so that lower hemolysis is also "higher is better".
    """
    if not candidates:
        return {"status": "not evaluated", "reason": "no candidates"}

    pts = np.array(
        [[c.activity_norm, -c.hemolysis_norm] for c in candidates], dtype=float
    )
    front_idx = pareto_front(pts)
    front = sorted(
        (candidates[i] for i in front_idx), key=lambda c: -c.activity_norm
    )

    parent_pt = np.array(
        [[parent_point["activity_norm"], -parent_point["hemolysis_norm"]]], dtype=float
    )
    # does any candidate dominate the parent?
    ge = np.all(pts >= parent_pt - 1e-12, axis=1)
    gt = np.any(pts > parent_pt + 1e-12, axis=1)
    dominating = np.flatnonzero(ge & gt)

    return {
        "status": "evaluated",
        "objectives": [
            "activity_normalized (maximised)",
            "-hemolysis_normalized (maximised, i.e. less hemolytic is better)",
        ],
        "dominance_definition": (
            "candidate A dominates B iff A is no worse in both objectives and strictly "
            "better in at least one"
        ),
        "n_candidates": len(candidates),
        "n_pareto_optimal": len(front_idx),
        "pareto_front": [c.as_dict() for c in front],
        "parent_point": {
            "activity_norm": parent_point["activity_norm"],
            "hemolysis_norm": parent_point["hemolysis_norm"],
        },
        "n_candidates_dominating_parent": int(len(dominating)),
        "candidates_dominating_parent": [
            candidates[i].as_dict() for i in dominating[:20]
        ],
        "all_points": [
            {
                "sequence": c.sequence,
                "mutations": c.mutations,
                "activity_norm": c.activity_norm,
                "hemolysis_norm": c.hemolysis_norm,
                "direct_score": c.direct_score,
                "n_mutations": c.n_mutations,
                "is_pareto_optimal": bool(i in set(front_idx.tolist())),
            }
            for i, c in enumerate(candidates)
        ],
    }


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def bootstrap_uncertainty(
    sequences: list[str], model_dir: str = "models", target: str = "activity"
) -> dict:
    """Predictive spread across the bootstrap ensemble.

    This is MODEL uncertainty, not experimental error. Returns a ``not evaluated``
    record when the ensemble is absent.
    """
    from pathlib import Path

    import xgboost as xgb

    from backend.features.featurize import featurize_many
    from backend.models.property_models import PropertyModel

    ens_dir = Path(model_dir) / f"{target}_bootstrap"
    if not ens_dir.exists():
        return {
            "status": "not evaluated",
            "reason": f"{ens_dir} not found; run backend.models.train",
        }
    files = sorted(ens_dir.glob("boot_*.json"))
    if not files:
        return {"status": "not evaluated", "reason": "ensemble directory is empty"}

    point = PropertyModel.load(target, model_dir)
    X = featurize_many(sequences, point.config)

    preds = []
    for f in files:
        b = xgb.XGBRegressor()
        b.load_model(str(f))
        preds.append(b.predict(X))
    P = np.stack(preds)
    return {
        "status": "evaluated",
        "target": target,
        "n_models": len(files),
        "interpretation": (
            "standard deviation across bootstrap-resampled models: MODEL/prediction "
            "uncertainty, NOT experimental measurement error"
        ),
        "per_sequence": [
            {
                "sequence": s,
                "mean": float(P[:, i].mean()),
                "std": float(P[:, i].std()),
                "min": float(P[:, i].min()),
                "max": float(P[:, i].max()),
            }
            for i, s in enumerate(sequences)
        ],
    }


# ---------------------------------------------------------------------------
# Trustworthiness diagnostics
# ---------------------------------------------------------------------------
_TRAIN_SEQ_CACHE: dict[str, list[str]] = {}


def training_sequences(target: str, processed_dir: str = "data/processed") -> list[str]:
    """Reconstruct the sequences a target's model was fitted on.

    Rebuilt deterministically from the curated CSV and the same clustering and split
    parameters training used, rather than stored in the model artefact. The split is a
    pure function of (sequences, identity threshold, seed, fractions), so this returns
    exactly the fitted set without bloating the saved model.
    """
    if target in _TRAIN_SEQ_CACHE:
        return _TRAIN_SEQ_CACHE[target]

    from pathlib import Path

    import numpy as np
    import pandas as pd

    from backend.data.splits import DEFAULT_IDENTITY_THRESHOLD, add_clusters, cluster_split
    from backend.models.train import SEED

    csv = Path(processed_dir) / f"{target}.csv"
    if not csv.exists():
        _TRAIN_SEQ_CACHE[target] = []
        return []
    df = pd.read_csv(csv)
    df, _ = add_clusters(df, DEFAULT_IDENTITY_THRESHOLD)
    cs = cluster_split(
        df["cluster"].to_numpy(), test_frac=0.2, val_frac=0.1, seed=SEED,
        identity_threshold=DEFAULT_IDENTITY_THRESHOLD,
    )
    fit_idx = np.concatenate([cs.train_idx, cs.val_idx])
    seqs = df["sequence"].to_numpy()[fit_idx].tolist()
    _TRAIN_SEQ_CACHE[target] = seqs
    return seqs


def nearest_training_identity(
    sequence: str, train_seqs: list[str]
) -> tuple[float, str | None]:
    """Highest sequence identity between ``sequence`` and any training sequence.

    A length-ratio guard skips comparisons that cannot possibly beat the best identity
    found so far, since identity is bounded above by the shorter/longer length ratio.
    """
    from backend.utils.peptide import sequence_identity

    best, best_seq = 0.0, None
    n = len(sequence)
    for t in train_seqs:
        shorter, longer = (n, len(t)) if n <= len(t) else (len(t), n)
        if longer == 0 or shorter / longer <= best:
            continue
        ident = sequence_identity(sequence, t)
        if ident > best:
            best, best_seq = ident, t
    return best, best_seq


def attach_trustworthiness(
    candidates: list[Candidate], model_dir: str = "models"
) -> dict:
    """Annotate candidates with training-set proximity and ensemble spread.

    Returns a summary suitable for the run record. Failures are reported, never silently
    swallowed -- an absent ensemble should read as "not evaluated", not as zero
    uncertainty.
    """
    import numpy as np

    if not candidates:
        return {"status": "not evaluated", "reason": "no candidates"}

    sequences = [c.sequence for c in candidates]
    summary: dict = {"status": "evaluated", "n_candidates": len(candidates)}

    # --- nearest training neighbour ---
    for target in ("activity", "hemolysis"):
        train_seqs = training_sequences(target)
        if not train_seqs:
            summary[f"{target}_nearest_identity"] = {
                "status": "not evaluated",
                "reason": f"data/processed/{target}.csv not found",
            }
            continue
        idents = []
        for c in candidates:
            ident, nearest = nearest_training_identity(c.sequence, train_seqs)
            c.nearest_training_identity = c.nearest_training_identity or {}
            c.nearest_training_identity[target] = {
                "max_identity": ident,
                "nearest_training_sequence": nearest,
            }
            idents.append(ident)
        summary[f"{target}_nearest_identity"] = {
            "status": "evaluated",
            "n_training_sequences": len(train_seqs),
            "min": float(np.min(idents)),
            "median": float(np.median(idents)),
            "max": float(np.max(idents)),
            "meaning": (
                "fraction of positions identical to the closest sequence the model was "
                "fitted on; high values mean the candidate is an interpolation, low "
                "values mean the prediction is an extrapolation"
            ),
        }

    # --- bootstrap ensemble spread ---
    for target in ("activity", "hemolysis"):
        try:
            unc = bootstrap_uncertainty(sequences, model_dir=model_dir, target=target)
        except Exception as exc:
            unc = {"status": "not evaluated",
                   "reason": f"{type(exc).__name__}: {exc}"}
        if unc.get("status") != "evaluated":
            summary[f"{target}_prediction_spread"] = unc
            continue
        stds = []
        for c, rec in zip(candidates, unc["per_sequence"]):
            c.prediction_spread = c.prediction_spread or {}
            c.prediction_spread[target] = {
                "mean": rec["mean"], "std": rec["std"],
                "min": rec["min"], "max": rec["max"],
            }
            stds.append(rec["std"])
        summary[f"{target}_prediction_spread"] = {
            "status": "evaluated",
            "n_models": unc["n_models"],
            "median_std": float(np.median(stds)),
            "max_std": float(np.max(stds)),
            "interpretation": unc["interpretation"],
        }

    return summary
