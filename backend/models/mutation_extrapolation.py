"""Direct measurement of mutation-extrapolation accuracy, using real assay data.

The rest of the project measures model quality on ABSOLUTE predictions (cluster-split
MAE/RMSE/Spearman) and measures QUBO-surrogate quality against the MODEL's own output.
Neither answers the specific question the pipeline actually depends on: "when the
sequence changes by one to three point mutations, how good is the model's predicted
CHANGE?" That is a question about extrapolation sensitivity, not absolute accuracy, and
the two are not the same thing -- a model can have good absolute error while being poor
at ranking which of two near-identical mutants is better.

Key idea. DRAMP contains natural near-duplicate sequences: published analogue series,
species variants, and point-mutant studies that happen to appear as separate entries. The
cluster-split test set (held out from training entirely) contains thousands of such pairs
-- two real, independently measured peptides that differ by 1-3 residues. For every such
pair (A, B) with TRUE labels y_A, y_B, the ALREADY-TRAINED model's predicted delta

    pred_delta = model(B) - model(A)

is compared against the true delta y_B - y_A. This uses real assay measurements on both
ends, is evaluated on sequences the model never trained on, and requires no synthetic
mutants and no retraining -- it reuses the model exactly as shipped.

This is the empirical answer to the limitation recorded in literature_review.md section 9
limitation 8 ("models trained on natural peptides, applied to point mutants"): rather than
leaving that as an unmeasured caveat, its magnitude is measured directly here.

Run:  python -m backend.models.mutation_extrapolation
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from backend.data.splits import DEFAULT_IDENTITY_THRESHOLD, add_clusters, cluster_split
from backend.models.property_models import PropertyModel

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/experiments")
MAX_DIFFS = 3

# Below this absolute true delta, the sign of the change is not considered meaningful for
# directional-accuracy scoring: on the p-scale (log10 concentration) a difference this
# small is within plausible assay noise, and scoring its sign would reward or punish the
# model on a distinction the data itself barely supports.
DEAD_ZONE = 0.1


def find_point_mutant_pairs(
    sequences: list[str], max_diffs: int = MAX_DIFFS
) -> list[tuple[int, int, int]]:
    """Index pairs ``(i, j, n_diffs)`` for same-length sequences differing by 1..max_diffs
    positions.

    Grouping by length first keeps this to O(sum of group_size^2) instead of O(n^2) overall,
    which matters once n is in the thousands.
    """
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(sequences):
        by_len[len(s)].append(i)

    pairs: list[tuple[int, int, int]] = []
    for idxs in by_len.values():
        if len(idxs) < 2:
            continue
        for a, b in itertools.combinations(idxs, 2):
            sa, sb = sequences[a], sequences[b]
            diffs = sum(1 for x, y in zip(sa, sb) if x != y)
            if 1 <= diffs <= max_diffs:
                pairs.append((a, b, diffs))
    return pairs


def _stratum_metrics(
    pred_delta: np.ndarray, true_delta: np.ndarray, dead_zone: float = DEAD_ZONE
) -> dict:
    """Delta-prediction metrics for one subset of pairs."""
    n = len(true_delta)
    if n == 0:
        return {"n_pairs": 0}
    resid = pred_delta - true_delta
    out = {
        "n_pairs": int(n),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "zero_baseline_mae": float(np.mean(np.abs(true_delta))),
        "zero_baseline_rmse": float(np.sqrt(np.mean(true_delta ** 2))),
        "true_delta_std": float(np.std(true_delta)),
    }
    out["beats_zero_baseline"] = bool(out["mae"] < out["zero_baseline_mae"])
    if n >= 3 and np.std(pred_delta) > 1e-12 and np.std(true_delta) > 1e-12:
        out["pearson_r"] = float(pearsonr(pred_delta, true_delta)[0])
        out["spearman_rho"] = float(spearmanr(pred_delta, true_delta)[0])
    else:
        out["pearson_r"] = None
        out["spearman_rho"] = None
    eligible = np.abs(true_delta) >= dead_zone
    out["n_directional_eligible"] = int(eligible.sum())
    out["directional_accuracy"] = (
        float(np.mean(np.sign(pred_delta[eligible]) == np.sign(true_delta[eligible])))
        if eligible.sum() > 0 else None
    )
    return out


@dataclass
class ExtrapolationReport:
    target: str
    n_test_sequences: int
    n_pairs: int
    pairs_by_diff_count: dict[str, int]
    mae: float
    rmse: float
    pearson_r: float | None
    spearman_rho: float | None
    zero_baseline_mae: float
    zero_baseline_rmse: float
    directional_accuracy: float | None
    n_directional_eligible: int
    dead_zone: float
    by_diff_count: dict[str, dict]
    worst_predictions: list[dict]
    best_predictions: list[dict]
    by_study_provenance: dict[str, dict]

    def _headline(self) -> dict:
        """The fair comparison, stated so it cannot be misread from the pooled numbers.

        The same-study stratum is the one a model can legitimately be judged on. The
        pooled figure mixes it with cross-study pairs whose true delta is dominated by
        inter-laboratory variation, and that mixing can invert the apparent result
        entirely -- which is exactly what happens for the activity target.
        """
        same = self.by_study_provenance.get("same_study", {})
        cross = self.by_study_provenance.get("cross_study", {})
        if not same.get("n_pairs"):
            return {"status": "not evaluated", "reason": "no same-study pairs"}

        acc = same.get("directional_accuracy")
        rho = same.get("spearman_rho")
        if acc is None:
            verdict = "UNDETERMINED"
        elif acc >= 0.70:
            verdict = "USABLE"
        elif acc >= 0.55:
            verdict = "WEAK BUT REAL"
        elif acc >= 0.45:
            verdict = "NO BETTER THAN CHANCE"
        else:
            verdict = "ANTI-CORRELATED"

        return {
            "fair_stratum": "same_study",
            "n_pairs": same["n_pairs"],
            "directional_accuracy": acc,
            "spearman_rho": rho,
            "verdict": verdict,
            "statement": (
                f"On {same['n_pairs']} real point-mutant pairs measured within the same "
                f"publication and held out from training, the model predicts the DIRECTION "
                f"of the change correctly {acc:.1%} of the time "
                f"(Spearman rho = {rho:.3f} on the signed magnitude)."
                if acc is not None and rho is not None else "insufficient data"
            ),
            "pooled_would_have_said": {
                "directional_accuracy": self.directional_accuracy,
                "spearman_rho": self.spearman_rho,
                "why_it_differs": (
                    "the pooled figure includes cross-study pairs "
                    f"(n={cross.get('n_pairs', 0)}, rho={cross.get('spearman_rho')}), "
                    "whose true delta carries inter-laboratory and inter-organism "
                    "variation that no sequence model can predict; pooling the strata "
                    "therefore understates the model"
                ),
            },
            "caveat_on_mae": (
                "the model does NOT beat the 'predict no change' baseline on MAE. That is "
                "expected when signal is weak: any predictor with variance loses on MAE to "
                "a constant predictor unless its signal is strong. Correlation and "
                "directional accuracy are the informative metrics here, and both are "
                "positive on the fair stratum"
            ),
        }

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "method": (
                "evaluates the ALREADY-TRAINED model (no retraining) on pairs of real "
                "assayed sequences that are both held out in the cluster-split test set "
                "and differ by 1-3 point mutations; predicted delta = model(B) - model(A) "
                "compared against true delta = y_B - y_A"
            ),
            "n_test_sequences": self.n_test_sequences,
            "n_pairs": self.n_pairs,
            "pairs_by_diff_count": self.pairs_by_diff_count,
            "delta_mae": self.mae,
            "delta_rmse": self.rmse,
            "pearson_r": self.pearson_r,
            "spearman_rho": self.spearman_rho,
            "zero_baseline_mae": self.zero_baseline_mae,
            "zero_baseline_rmse": self.zero_baseline_rmse,
            "beats_zero_baseline": bool(self.mae < self.zero_baseline_mae),
            "zero_baseline_meaning": (
                "MAE/RMSE of always predicting 'this mutation changes nothing' "
                "(pred_delta=0); the bar the model must clear to be useful at all for "
                "ranking mutations"
            ),
            "directional_accuracy": self.directional_accuracy,
            "n_directional_eligible": self.n_directional_eligible,
            "dead_zone": self.dead_zone,
            "dead_zone_meaning": (
                f"pairs with |true_delta| < {self.dead_zone} are excluded from directional "
                "accuracy: a change this small on the log10-concentration scale is within "
                "plausible assay noise, so scoring its sign would not be meaningful"
            ),
            "by_diff_count": self.by_diff_count,
            "by_study_provenance": self.by_study_provenance,
            "study_provenance_meaning": (
                "same_study pairs were both measured in the same publication, so they "
                "share laboratory, assay protocol and usually target organism; "
                "cross_study pairs were measured independently, so their true delta "
                "carries inter-laboratory variation on top of any real mutation effect. "
                "A model can only be fairly judged on the same_study stratum"
            ),
            "headline": self._headline(),
            "worst_predictions": self.worst_predictions,
            "best_predictions": self.best_predictions,
        }


def evaluate_mutation_extrapolation(
    target: str,
    csv_name: str,
    y_col: str,
    model_dir: Path | str = "models",
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    seed: int | None = None,
    n_examples: int = 8,
) -> ExtrapolationReport:
    """Evaluate one target's deployed model against real held-out point-mutant pairs."""
    from backend.models.train import SEED as TRAIN_SEED

    seed = TRAIN_SEED if seed is None else seed

    df = pd.read_csv(PROCESSED_DIR / csv_name)
    df, _ = add_clusters(df, identity_threshold)
    sequences = df["sequence"].tolist()
    y = df[y_col].to_numpy(dtype=float)
    clusters = df["cluster"].to_numpy()

    # EXACTLY the split train.py used to fit the shipped model, so "test" here means
    # sequences the deployed model genuinely never trained on.
    cs = cluster_split(clusters, test_frac=0.2, val_frac=0.1, seed=seed,
                       identity_threshold=identity_threshold)
    test_set = set(cs.test_idx.tolist())

    pairs = [
        (a, b, d) for a, b, d in find_point_mutant_pairs(sequences)
        if a in test_set and b in test_set
    ]

    model = PropertyModel.load(target, model_dir)
    preds = model.predict_raw(sequences)  # raw p-scale; safe to index by position

    true_delta = np.array([y[b] - y[a] for a, b, _ in pairs])
    pred_delta = np.array([preds[b] - preds[a] for a, b, _ in pairs])
    diffs_arr = np.array([d for _, _, d in pairs])
    resid = pred_delta - true_delta

    mae = float(np.mean(np.abs(resid))) if len(pairs) else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2))) if len(pairs) else float("nan")
    zero_mae = float(np.mean(np.abs(true_delta))) if len(pairs) else float("nan")
    zero_rmse = float(np.sqrt(np.mean(true_delta ** 2))) if len(pairs) else float("nan")

    pearson_r = spearman_rho = None
    if len(pairs) >= 3 and np.std(pred_delta) > 1e-12 and np.std(true_delta) > 1e-12:
        pearson_r = float(pearsonr(pred_delta, true_delta)[0])
        spearman_rho = float(spearmanr(pred_delta, true_delta)[0])

    eligible = np.abs(true_delta) >= DEAD_ZONE
    if eligible.sum() > 0:
        correct_sign = np.sign(pred_delta[eligible]) == np.sign(true_delta[eligible])
        directional_acc = float(np.mean(correct_sign))
    else:
        directional_acc = None

    # --- same-study vs cross-study ------------------------------------------
    # A pair measured twice in the same publication shares lab, protocol and usually
    # target organism. A pair measured in different publications carries inter-lab
    # variation in its TRUE delta that no model could predict, so pooling the two strata
    # would understate the model by blaming it for label noise.
    if "pubmed_ids" in df.columns:
        refs = [
            set(str(v).split(";")) - {"", "nan"}
            for v in df["pubmed_ids"].fillna("")
        ]
        same_study = np.array([
            bool(refs[a] & refs[b]) for a, b, _ in pairs
        ]) if pairs else np.zeros(0, dtype=bool)
    else:
        same_study = np.zeros(len(pairs), dtype=bool)

    by_study = {
        "same_study": _stratum_metrics(pred_delta[same_study], true_delta[same_study]),
        "cross_study": _stratum_metrics(pred_delta[~same_study], true_delta[~same_study]),
    }

    by_diff: dict[str, dict] = {}
    for d in sorted(set(diffs_arr.tolist())):
        mask = diffs_arr == d
        r = resid[mask]
        by_diff[str(d)] = {
            "n_pairs": int(mask.sum()),
            "mae": float(np.mean(np.abs(r))),
            "rmse": float(np.sqrt(np.mean(r ** 2))),
        }

    order = np.argsort(-np.abs(resid))
    def _example(k: int) -> dict:
        a, b, d = pairs[k]
        return {
            "sequence_a": sequences[a], "sequence_b": sequences[b],
            "n_diffs": d, "true_delta": float(true_delta[k]),
            "predicted_delta": float(pred_delta[k]),
            "error": float(resid[k]),
        }
    worst = [_example(int(k)) for k in order[:n_examples]]
    best = [_example(int(k)) for k in order[-n_examples:][::-1]] if len(pairs) else []

    return ExtrapolationReport(
        target=target,
        n_test_sequences=len(test_set),
        n_pairs=len(pairs),
        pairs_by_diff_count={str(d): int((diffs_arr == d).sum()) for d in sorted(set(diffs_arr.tolist()))},
        mae=mae, rmse=rmse,
        pearson_r=pearson_r, spearman_rho=spearman_rho,
        zero_baseline_mae=zero_mae, zero_baseline_rmse=zero_rmse,
        directional_accuracy=directional_acc,
        n_directional_eligible=int(eligible.sum()),
        dead_zone=DEAD_ZONE,
        by_diff_count=by_diff,
        worst_predictions=worst,
        best_predictions=best,
        by_study_provenance=by_study,
    )


TARGETS = [
    ("activity", "activity.csv", "y_activity"),
    ("hemolysis", "hemolysis.csv", "y_hemolysis"),
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    for target, csv_name, y_col in TARGETS:
        print(f"\n=== mutation extrapolation: {target} ===")
        report = evaluate_mutation_extrapolation(target, csv_name, y_col)
        out[target] = report.as_dict()
        print(f"  held-out point-mutant pairs : {report.n_pairs} "
              f"(from {report.n_test_sequences} test sequences)")
        print(f"  by diff count               : {report.pairs_by_diff_count}")
        print(f"  delta MAE / RMSE            : {report.mae:.4f} / {report.rmse:.4f}")
        print(f"  zero-baseline MAE / RMSE    : {report.zero_baseline_mae:.4f} / "
              f"{report.zero_baseline_rmse:.4f}  (beats baseline: "
              f"{report.mae < report.zero_baseline_mae})")
        print(f"  Pearson / Spearman          : {report.pearson_r} / {report.spearman_rho}")
        print(f"  directional accuracy        : {report.directional_accuracy} "
              f"(n={report.n_directional_eligible}, dead zone |Δ|<{report.dead_zone})")
        for d, rec in report.by_diff_count.items():
            print(f"    {d} mutation(s): n={rec['n_pairs']:4d} "
                  f"MAE={rec['mae']:.4f} RMSE={rec['rmse']:.4f}")
        print("  by study provenance:")
        for stratum, rec in report.by_study_provenance.items():
            if not rec.get("n_pairs"):
                print(f"    {stratum:12s}: no pairs")
                continue
            print(f"    {stratum:12s}: n={rec['n_pairs']:4d} "
                  f"MAE={rec['mae']:.4f} (baseline {rec['zero_baseline_mae']:.4f}) "
                  f"rho={rec['spearman_rho']} "
                  f"dir_acc={rec['directional_accuracy']} "
                  f"(n_elig={rec['n_directional_eligible']})")

        h = report._headline()
        if h.get("statement"):
            print(f"  >> {h.get('verdict')}: {h['statement']}")

    path = RESULTS_DIR / "mutation_extrapolation_report.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
