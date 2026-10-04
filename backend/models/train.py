"""Train and validate the activity and hemolysis property models.

Reports, for each target:
  * cluster-level split metrics  (the honest numbers)
  * random split metrics        (invalid; reported only to size the optimism gap)
  * cluster-grouped cross-validation
  * a bootstrap ensemble for prediction uncertainty

Run:  python -m backend.models.train
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from backend.data.splits import (
    DEFAULT_IDENTITY_THRESHOLD,
    add_clusters,
    cluster_split,
    random_split,
)
from backend.features.featurize import DEFAULT_CONFIG, FeatureConfig, featurize_many
from backend.models.property_models import (
    DEFAULT_XGB_PARAMS,
    MODEL_DIR,
    PropertyModel,
    fit_normalization_from_predictions,
)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/experiments")
SEED = 20261004
N_BOOTSTRAP = 10


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / R2 / Pearson / Spearman, guarding degenerate inputs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"n": 0}
    out = {
        "n": int(n),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "y_true_std": float(np.std(y_true)),
    }
    # R2 and correlations are undefined / meaningless for n<3 or constant input
    if n >= 3 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        out["r2"] = float(r2_score(y_true, y_pred))
        out["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
        out["spearman_rho"] = float(spearmanr(y_true, y_pred)[0])
    else:
        out["r2"] = None
        out["pearson_r"] = None
        out["spearman_rho"] = None
    # a baseline that always predicts the training mean, for context
    out["rmse_mean_baseline"] = float(np.sqrt(np.mean((y_true - y_true.mean()) ** 2)))
    return out


@dataclass
class TargetSpec:
    name: str
    csv: str
    y_col: str
    description: str
    direction: str
    # Chosen by backend.models.select_config on the VALIDATION split only.
    # See results/experiments/model_selection_sweep.json for the full grid.
    feature_config: FeatureConfig = DEFAULT_CONFIG
    xgb_overrides: dict | None = None
    selection_note: str = ""


# Feature block selected for both targets: amino-acid composition + physicochemical
# descriptors (35 features). The 400-dimensional dipeptide block is implemented and
# available, but it LOWERED validation performance for both targets -- most sharply for
# hemolysis (validation Spearman 0.30 with dipeptides vs 0.62 without, n_val=92). That is
# the k-mer / dimensionality problem the literature warns about, measured directly rather
# than assumed. It is therefore not used in the fitted models.
_SELECTED_FEATURES = FeatureConfig(
    use_composition=True, use_dipeptide=False, use_physchem=True
)

TARGETS = [
    TargetSpec(
        "activity",
        "activity.csv",
        "y_activity",
        "y = 6 - log10(MIC[uM]); minimum MIC across organisms",
        "higher = more active (better)",
        feature_config=_SELECTED_FEATURES,
        xgb_overrides=None,  # default capacity won the validation sweep
        selection_note=(
            "features=comp_physchem, params=default; selected on validation RMSE "
            "(0.9652 vs mean-baseline 1.0694)"
        ),
    ),
    TargetSpec(
        "hemolysis",
        "hemolysis.csv",
        "y_hemolysis",
        "y = 6 - log10(D50[uM]); 50%-effect hemolytic dose",
        "higher = more hemolytic (worse)",
        feature_config=_SELECTED_FEATURES,
        xgb_overrides={
            "n_estimators": 400, "max_depth": 3, "learning_rate": 0.03,
            "min_child_weight": 8, "reg_lambda": 3.0, "colsample_bytree": 0.7,
        },
        selection_note=(
            "features=comp_physchem, params=shallow_reg; selected on validation RMSE "
            "(0.7236 vs mean-baseline 0.8492). Lower capacity was necessary: n=916 with "
            "435 features gave a model no better than the mean."
        ),
    ),
]


def _fit(X: np.ndarray, y: np.ndarray, seed: int, params: dict | None = None) -> xgb.XGBRegressor:
    p = dict(DEFAULT_XGB_PARAMS)
    if params:
        p.update(params)
    p["random_state"] = seed
    model = xgb.XGBRegressor(**p)
    model.fit(X, y)
    return model


def train_target(
    spec: TargetSpec,
    config: FeatureConfig | None = None,
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
) -> tuple[PropertyModel, dict]:
    config = config or spec.feature_config
    params = spec.xgb_overrides
    path = PROCESSED_DIR / spec.csv
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m backend.data.curate` first."
        )
    df = pd.read_csv(path)
    df, redundancy = add_clusters(df, identity_threshold)

    sequences = df["sequence"].tolist()
    y = df[spec.y_col].to_numpy(dtype=float)
    clusters = df["cluster"].to_numpy()

    t0 = time.time()
    X = featurize_many(sequences, config)
    featurize_seconds = time.time() - t0

    report: dict = {
        "target": spec.name,
        "target_definition": spec.description,
        "direction": spec.direction,
        "n_sequences": int(len(df)),
        "n_features": int(X.shape[1]),
        "feature_config": config.as_dict(),
        "featurize_seconds": round(featurize_seconds, 2),
        "redundancy": redundancy,
        "seed": SEED,
        "xgb_params": {**DEFAULT_XGB_PARAMS, **(params or {})},
        "model_selection": {
            "selected_on": "validation split only; test never consulted",
            "note": spec.selection_note,
            "sweep_report": "results/experiments/model_selection_sweep.json",
        },
    }

    # --- cluster-level split (the honest evaluation) -----------------------
    cs = cluster_split(clusters, test_frac=0.2, val_frac=0.1, seed=SEED,
                       identity_threshold=identity_threshold)
    model_cs = _fit(X[cs.train_idx], y[cs.train_idx], SEED, params)
    report["split_cluster"] = cs.summary(y)
    report["metrics_cluster_split"] = {
        "train": regression_metrics(y[cs.train_idx], model_cs.predict(X[cs.train_idx])),
        "val": regression_metrics(y[cs.val_idx], model_cs.predict(X[cs.val_idx])),
        "test": regression_metrics(y[cs.test_idx], model_cs.predict(X[cs.test_idx])),
    }

    # --- random split (INVALID; optimism gap only) -------------------------
    rs = random_split(len(df), test_frac=0.2, val_frac=0.1, seed=SEED)
    model_rs = _fit(X[rs.train_idx], y[rs.train_idx], SEED, params)
    report["split_random"] = rs.summary(y)
    report["metrics_random_split"] = {
        "test": regression_metrics(y[rs.test_idx], model_rs.predict(X[rs.test_idx])),
        "WARNING": (
            "random split leaks near-duplicate sequences across train/test and is NOT a "
            "valid estimate of generalisation; reported only to quantify the optimism gap"
        ),
    }
    cl_rmse = report["metrics_cluster_split"]["test"]["rmse"]
    rd_rmse = report["metrics_random_split"]["test"]["rmse"]
    report["optimism_gap_rmse"] = {
        "cluster_split_test_rmse": cl_rmse,
        "random_split_test_rmse": rd_rmse,
        "absolute_difference": float(cl_rmse - rd_rmse),
        "interpretation": (
            "a positive difference means the random split understates error, i.e. "
            "reports optimistic performance"
        ),
    }

    # --- grouped cross-validation ------------------------------------------
    n_groups = len(np.unique(clusters))
    n_splits = min(5, n_groups)
    cv_scores = []
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, y, groups=clusters):
            m = _fit(X[tr], y[tr], SEED, params)
            cv_scores.append(regression_metrics(y[te], m.predict(X[te])))
        report["cross_validation"] = {
            "scheme": f"GroupKFold(n_splits={n_splits}) grouped by identity cluster",
            "fold_metrics": cv_scores,
            "mae_mean": float(np.mean([s["mae"] for s in cv_scores])),
            "mae_std": float(np.std([s["mae"] for s in cv_scores])),
            "rmse_mean": float(np.mean([s["rmse"] for s in cv_scores])),
            "rmse_std": float(np.std([s["rmse"] for s in cv_scores])),
            "spearman_mean": float(
                np.mean([s["spearman_rho"] for s in cv_scores
                         if s.get("spearman_rho") is not None])
            ),
        }
    else:
        report["cross_validation"] = {"status": "not evaluated: too few clusters"}

    # --- final model: refit on train+val of the cluster split --------------
    # The test split is held out and never used for fitting or model selection.
    fit_idx = np.concatenate([cs.train_idx, cs.val_idx])
    final = _fit(X[fit_idx], y[fit_idx], SEED, params)
    report["final_model"] = {
        "fitted_on": "cluster-split train + validation",
        "n_fit": int(len(fit_idx)),
        "test_held_out": True,
        "metrics_heldout_test": regression_metrics(
            y[cs.test_idx], final.predict(X[cs.test_idx])
        ),
    }

    # --- bootstrap ensemble for prediction uncertainty ---------------------
    rng = np.random.default_rng(SEED)
    boosters = []
    for b in range(N_BOOTSTRAP):
        draw = rng.choice(fit_idx, size=len(fit_idx), replace=True)
        boosters.append(_fit(X[draw], y[draw], SEED + b + 1, params))
    ens_test = np.stack([bst.predict(X[cs.test_idx]) for bst in boosters])
    report["bootstrap_ensemble"] = {
        "n_models": N_BOOTSTRAP,
        "purpose": "model/prediction uncertainty -- NOT experimental error",
        "mean_predictive_std_on_test": float(ens_test.std(axis=0).mean()),
        "metrics_ensemble_mean_test": regression_metrics(
            y[cs.test_idx], ens_test.mean(axis=0)
        ),
    }

    # --- normalization, frozen from training predictions -------------------
    norm = fit_normalization_from_predictions(final, X[fit_idx], spec.name)
    report["normalization"] = norm.as_dict()

    # feature importance, top 20
    imp = final.feature_importances_
    from backend.features.featurize import feature_names
    names = feature_names(config)
    top = sorted(zip(names, imp), key=lambda t: -t[1])[:20]
    report["top_features"] = [{"feature": n, "gain_importance": float(v)} for n, v in top]

    pm = PropertyModel(
        name=spec.name,
        booster=final,
        config=config,
        normalization=norm,
        metadata={
            "target_definition": spec.description,
            "direction": spec.direction,
            "n_train": int(len(fit_idx)),
            "identity_threshold": identity_threshold,
            "heldout_test_mae": report["final_model"]["metrics_heldout_test"]["mae"],
            "heldout_test_spearman": report["final_model"]["metrics_heldout_test"].get(
                "spearman_rho"
            ),
            "seed": SEED,
        },
    )
    # persist the bootstrap ensemble alongside the point model
    ens_dir = MODEL_DIR / f"{spec.name}_bootstrap"
    ens_dir.mkdir(parents=True, exist_ok=True)
    for i, bst in enumerate(boosters):
        bst.save_model(str(ens_dir / f"boot_{i}.json"))

    return pm, report


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    for spec in TARGETS:
        print(f"\n=== training {spec.name} ===")
        model, report = train_target(spec)
        model.save()
        all_reports[spec.name] = report

        m = report["final_model"]["metrics_heldout_test"]
        print(f"  sequences          : {report['n_sequences']}")
        print(f"  clusters (70% id)  : {report['redundancy']['n_clusters']}")
        print(f"  held-out test n    : {m['n']}")
        print(f"  MAE / RMSE         : {m['mae']:.3f} / {m['rmse']:.3f}")
        print(f"  RMSE mean-baseline : {m['rmse_mean_baseline']:.3f}")
        print(f"  R2 / Spearman      : {m['r2']:.3f} / {m['spearman_rho']:.3f}"
              if m.get("r2") is not None else "  R2 / Spearman      : n/a")
        og = report["optimism_gap_rmse"]
        print(f"  optimism gap (RMSE): cluster {og['cluster_split_test_rmse']:.3f} vs "
              f"random {og['random_split_test_rmse']:.3f}")

    out = RESULTS_DIR / "ml_training_report.json"
    out.write_text(json.dumps(all_reports, indent=2))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
