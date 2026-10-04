"""Model-selection sweep over feature blocks and XGBoost capacity.

Selection is done on the VALIDATION split only. The test split is never consulted here;
it is scored once, afterwards, by backend.models.train. This separation is the whole
point of the module (spec section 7: the test set must not influence model design).

Run:  python -m backend.models.select_config
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.data.splits import DEFAULT_IDENTITY_THRESHOLD, add_clusters, cluster_split
from backend.features.featurize import FeatureConfig, featurize_many
from backend.models.train import SEED, TARGETS, _fit, regression_metrics

RESULTS_DIR = Path("results/experiments")

FEATURE_GRID: dict[str, FeatureConfig] = {
    "physchem_only": FeatureConfig(use_composition=False, use_dipeptide=False,
                                   use_physchem=True),
    "comp_physchem": FeatureConfig(use_composition=True, use_dipeptide=False,
                                   use_physchem=True),
    "comp_only": FeatureConfig(use_composition=True, use_dipeptide=False,
                               use_physchem=False),
    "all_with_dipeptide": FeatureConfig(use_composition=True, use_dipeptide=True,
                                        use_physchem=True),
}

PARAM_GRID: dict[str, dict] = {
    "shallow_small": {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05,
                      "min_child_weight": 10, "reg_lambda": 5.0, "colsample_bytree": 0.8},
    "shallow_reg": {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03,
                    "min_child_weight": 8, "reg_lambda": 3.0, "colsample_bytree": 0.7},
    "default": {},
    "deep": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
             "min_child_weight": 2, "reg_lambda": 1.0, "colsample_bytree": 0.6},
}


def sweep_target(spec, identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD) -> dict:
    df = pd.read_csv(Path("data/processed") / spec.csv)
    df, _ = add_clusters(df, identity_threshold)
    sequences = df["sequence"].tolist()
    y = df[spec.y_col].to_numpy(dtype=float)
    clusters = df["cluster"].to_numpy()

    cs = cluster_split(clusters, test_frac=0.2, val_frac=0.1, seed=SEED,
                       identity_threshold=identity_threshold)

    rows = []
    for fname, fcfg in FEATURE_GRID.items():
        X = featurize_many(sequences, fcfg)
        for pname, params in PARAM_GRID.items():
            model = _fit(X[cs.train_idx], y[cs.train_idx], SEED, params)
            val = regression_metrics(y[cs.val_idx], model.predict(X[cs.val_idx]))
            train = regression_metrics(y[cs.train_idx], model.predict(X[cs.train_idx]))
            rows.append({
                "features": fname,
                "n_features": int(X.shape[1]),
                "params": pname,
                "val_rmse": val["rmse"],
                "val_mae": val["mae"],
                "val_r2": val.get("r2"),
                "val_spearman": val.get("spearman_rho"),
                "val_rmse_mean_baseline": val["rmse_mean_baseline"],
                "train_rmse": train["rmse"],
                "overfit_ratio": (train["rmse"] / val["rmse"]) if val["rmse"] else None,
            })

    # rank by validation RMSE; a model that cannot beat the validation mean baseline is
    # flagged, because that is the only threshold that makes the model worth having
    rows.sort(key=lambda r: r["val_rmse"])
    best = rows[0]
    baseline = best["val_rmse_mean_baseline"]
    return {
        "target": spec.name,
        "selection_split": "validation (cluster-level); test never consulted",
        "n_val": int(len(cs.val_idx)),
        "val_mean_baseline_rmse": baseline,
        "best": best,
        "beats_mean_baseline": bool(best["val_rmse"] < baseline),
        "all_results": rows,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for spec in TARGETS:
        print(f"\n=== sweep: {spec.name} ===")
        res = sweep_target(spec)
        out[spec.name] = res
        print(f"  validation n              : {res['n_val']}")
        print(f"  val mean-baseline RMSE    : {res['val_mean_baseline_rmse']:.4f}")
        print(f"  {'features':<20} {'params':<14} {'n_feat':>6} {'val_RMSE':>9} "
              f"{'val_rho':>8} {'overfit':>8}")
        for r in res["all_results"][:8]:
            rho = f"{r['val_spearman']:.3f}" if r["val_spearman"] is not None else "  n/a"
            print(f"  {r['features']:<20} {r['params']:<14} {r['n_features']:>6} "
                  f"{r['val_rmse']:>9.4f} {rho:>8} {r['overfit_ratio']:>8.2f}")
        print(f"  -> beats mean baseline    : {res['beats_mean_baseline']}")

    path = RESULTS_DIR / "model_selection_sweep.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
