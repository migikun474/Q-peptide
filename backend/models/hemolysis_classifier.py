"""Statement-based hemolysis classifier -- an independent cross-check on the regression.

Why this exists. The hemolysis REGRESSION target is a 50%-effect dose parsed from text,
and it is the weaker of the two property models. This classifier is trained on a
completely different labelling: the experimenters' own stated determination
("no hemolytic activity was detected" vs "induced hemolysis"), with **no thresholding of
any numeric value**. If the two disagree wildly, the regression's signal is suspect; if
they agree, the signal survives a change of label definition.

The label sets are related but not identical, so the comparison is made only on the
sequences both cover.

Run:  python -m backend.models.hemolysis_classifier
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from backend.data.splits import DEFAULT_IDENTITY_THRESHOLD, add_clusters
from backend.features.featurize import FeatureConfig, featurize_many
from backend.models.property_models import PropertyModel

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/experiments")
MODEL_DIR = Path("models")
SEED = 20261004

# Same feature block the regressors use (composition + physicochemical). Keeping it
# identical means a disagreement between the two models is about the LABELS, not about
# the representation.
FEATURES = FeatureConfig(use_composition=True, use_dipeptide=False, use_physchem=True)

PARAMS = {
    "n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 4,
    "reg_lambda": 3.0, "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "n_jobs": 4,
}


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, pred)),
        # Balanced accuracy is the honest headline when classes are uneven: plain
        # accuracy can look respectable while the model only ever predicts one class.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(set(y_true)) > 1 else None,
        "majority_class_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
    }
    if len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        out["roc_auc"] = None
    out["beats_majority_baseline"] = bool(out["balanced_accuracy"] > 0.5 + 1e-9)
    return out


def train_and_evaluate(identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD) -> dict:
    path = PROCESSED_DIR / "hemolysis_cls.csv"
    if not path.exists():
        return {"status": "not evaluated",
                "reason": f"{path} not found; run backend.data.curate"}

    df = pd.read_csv(path)
    if len(df) < 40 or df["label"].nunique() < 2:
        return {"status": "not evaluated",
                "reason": f"insufficient labelled data (n={len(df)})"}

    df, redundancy = add_clusters(df, identity_threshold)
    X = featurize_many(df["sequence"].tolist(), FEATURES)
    y = df["label"].to_numpy(dtype=int)
    groups = df["cluster"].to_numpy()

    # The set is small (hundreds), so a single hold-out split would be dominated by
    # which clusters happened to land in it. Cross-validation grouped by identity
    # cluster gives a usable estimate with a spread attached.
    n_groups = len(np.unique(groups))
    n_splits = int(min(5, n_groups, np.bincount(y).min()))
    if n_splits < 2:
        return {"status": "not evaluated",
                "reason": f"too few clusters or minority examples for CV "
                          f"(clusters={n_groups}, min class={int(np.bincount(y).min())})"}

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    fold_metrics = []
    for tr, te in skf.split(X, y, groups=groups):
        clf = xgb.XGBClassifier(**PARAMS, random_state=SEED)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te] = p
        fold_metrics.append(_metrics(y[te], p))

    overall = _metrics(y, oof)

    # Final model on everything, saved for the agreement check and for reuse.
    final = xgb.XGBClassifier(**PARAMS, random_state=SEED)
    final.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final.save_model(str(MODEL_DIR / "hemolysis_classifier.json"))

    return {
        "status": "evaluated",
        "purpose": (
            "independent cross-check on the hemolysis REGRESSION, using the "
            "experimenters' own stated determination as the label rather than any "
            "threshold applied to a numeric dose"
        ),
        "n_sequences": int(len(df)),
        "class_balance": {"non_hemolytic": int((y == 0).sum()),
                          "hemolytic": int((y == 1).sum())},
        "features": FEATURES.as_dict(),
        "redundancy": redundancy,
        "cv_scheme": f"StratifiedGroupKFold(n_splits={n_splits}) grouped by identity cluster",
        "out_of_fold_metrics": overall,
        "fold_metrics": fold_metrics,
        "balanced_accuracy_std": float(
            np.std([f["balanced_accuracy"] for f in fold_metrics])
        ),
        "model_path": str(MODEL_DIR / "hemolysis_classifier.json"),
    }


def agreement_with_regression() -> dict:
    """Do the two differently-labelled hemolysis models rank sequences the same way?

    The classifier's P(hemolytic) and the regressor's predicted p-dose should correlate
    POSITIVELY: both increase with hemolytic propensity. They were trained on disjoint
    label definitions, so agreement is evidence the signal is real rather than an
    artefact of how one target was constructed.
    """
    cls_path = MODEL_DIR / "hemolysis_classifier.json"
    data_path = PROCESSED_DIR / "hemolysis_cls.csv"
    if not cls_path.exists() or not data_path.exists():
        return {"status": "not evaluated", "reason": "classifier or labels missing"}
    try:
        reg = PropertyModel.load("hemolysis")
    except FileNotFoundError as exc:
        return {"status": "not evaluated", "reason": str(exc)}

    df = pd.read_csv(data_path)
    seqs = df["sequence"].tolist()

    clf = xgb.XGBClassifier()
    clf.load_model(str(cls_path))
    proba = clf.predict_proba(featurize_many(seqs, FEATURES))[:, 1]
    reg_pred = reg.predict_raw(seqs)

    from scipy.stats import pearsonr, spearmanr

    if np.std(proba) < 1e-12 or np.std(reg_pred) < 1e-12:
        return {"status": "not evaluated", "reason": "degenerate predictions"}

    rho = float(spearmanr(proba, reg_pred)[0])
    return {
        "status": "evaluated",
        "n_sequences": len(seqs),
        "pearson_r": float(pearsonr(proba, reg_pred)[0]),
        "spearman_rho": rho,
        "expected_sign": "positive",
        "agrees": bool(rho > 0),
        "caveat": (
            "the classifier was trained on these sequences, so this measures whether the "
            "two label definitions describe the same underlying property -- it is NOT an "
            "out-of-sample performance estimate for either model"
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = train_and_evaluate()
    report["agreement_with_regression"] = agreement_with_regression()

    path = RESULTS_DIR / "hemolysis_classifier_report.json"
    path.write_text(json.dumps(report, indent=2))

    print("hemolysis classifier (statement-based labels, no thresholding)")
    if report.get("status") != "evaluated":
        print(f"  {report.get('status')}: {report.get('reason')}")
        return
    m = report["out_of_fold_metrics"]
    print(f"  sequences          : {report['n_sequences']} "
          f"({report['class_balance']['hemolytic']}+ / "
          f"{report['class_balance']['non_hemolytic']}-)")
    print(f"  clusters           : {report['redundancy']['n_clusters']}")
    print(f"  scheme             : {report['cv_scheme']}")
    print(f"  balanced accuracy  : {m['balanced_accuracy']:.3f} "
          f"± {report['balanced_accuracy_std']:.3f}  "
          f"(majority baseline {m['majority_class_accuracy']:.3f})")
    print(f"  ROC AUC / MCC / F1 : {m['roc_auc']:.3f} / {m['mcc']:.3f} / {m['f1']:.3f}")
    print(f"  beats baseline     : {m['beats_majority_baseline']}")
    ag = report["agreement_with_regression"]
    if ag.get("status") == "evaluated":
        print(f"  agreement with regression: Spearman {ag['spearman_rho']:.3f} "
              f"(expected positive; agrees={ag['agrees']})")
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
