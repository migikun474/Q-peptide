"""XGBoost property models for antimicrobial activity and hemolysis, plus the scalar
biological score that combines them.

Two separate regressors are trained (see research/literature_review.md section 2.2 for
why both targets are regressions rather than classifications):

    activity   y_A = 6 - log10(MIC[uM])     higher = more active
    hemolysis  y_H = 6 - log10(D50[uM])     higher = MORE hemolytic

Normalization. The predictions are standardised before being combined, using mu/sigma
computed **once from model predictions on the training split** and then frozen. They are
taken from predictions rather than labels because it is predictions that the objective
combines, so standardising in prediction space is what makes alpha and beta comparable.
They are never recomputed per candidate -- doing so would change the objective between
candidates and invalidate every comparison (spec section 9).

Score convention, fixed in one place:

    S(P) = alpha * A_tilde(P) - beta * H_tilde(P)       LARGER IS BETTER
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import xgboost as xgb

from backend.features.featurize import (
    DEFAULT_CONFIG,
    FeatureConfig,
    feature_names,
    featurize_many,
)

MODEL_DIR = Path("models")


@dataclass(frozen=True)
class Normalization:
    """Frozen standardisation parameters for one property."""
    mu: float
    sigma: float
    source: str  # how mu/sigma were obtained, for the reproducibility record

    def apply(self, values: np.ndarray) -> np.ndarray:
        s = self.sigma if self.sigma > 1e-12 else 1.0
        return (np.asarray(values, dtype=np.float64) - self.mu) / s

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 4,
}


class PropertyModel:
    """A single trained property regressor with its feature and normalization config."""

    def __init__(
        self,
        name: str,
        booster: xgb.XGBRegressor,
        config: FeatureConfig,
        normalization: Normalization,
        metadata: dict | None = None,
    ) -> None:
        self.name = name
        self.booster = booster
        self.config = config
        self.normalization = normalization
        self.metadata = metadata or {}

    # -- prediction ---------------------------------------------------------
    def predict_raw(self, sequences: list[str]) -> np.ndarray:
        """Predictions on the native p-scale."""
        if not sequences:
            return np.zeros(0)
        X = featurize_many(sequences, self.config)
        return np.asarray(self.booster.predict(X), dtype=np.float64)

    def predict_normalized(self, sequences: list[str]) -> np.ndarray:
        """Predictions standardised with the frozen mu/sigma."""
        return self.normalization.apply(self.predict_raw(sequences))

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path | str = MODEL_DIR) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / f"{self.name}.json"))
        meta = {
            "name": self.name,
            "feature_config": self.config.as_dict(),
            "n_features": len(feature_names(self.config)),
            "normalization": self.normalization.as_dict(),
            "metadata": self.metadata,
        }
        path = directory / f"{self.name}.meta.json"
        path.write_text(json.dumps(meta, indent=2))
        return path

    @staticmethod
    def load(name: str, directory: Path | str = MODEL_DIR) -> "PropertyModel":
        directory = Path(directory)
        meta_path = directory / f"{name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Train the models first with "
                "`python -m backend.models.train`."
            )
        meta = json.loads(meta_path.read_text())
        booster = xgb.XGBRegressor()
        booster.load_model(str(directory / f"{name}.json"))
        config = FeatureConfig.from_dict(meta["feature_config"])
        norm = Normalization(**meta["normalization"])
        return PropertyModel(name, booster, config, norm, meta.get("metadata", {}))


@dataclass(frozen=True)
class ScoreWeights:
    """Objective weights. Both non-negative; larger S is better."""
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    """Per-sequence score components, so the UI and reports never re-derive them."""
    sequences: list[str]
    activity_raw: np.ndarray
    hemolysis_raw: np.ndarray
    activity_norm: np.ndarray
    hemolysis_norm: np.ndarray
    score: np.ndarray

    def row(self, i: int) -> dict:
        return {
            "sequence": self.sequences[i],
            "activity_raw": float(self.activity_raw[i]),
            "hemolysis_raw": float(self.hemolysis_raw[i]),
            "activity_norm": float(self.activity_norm[i]),
            "hemolysis_norm": float(self.hemolysis_norm[i]),
            "score": float(self.score[i]),
        }

    def as_records(self) -> list[dict]:
        return [self.row(i) for i in range(len(self.sequences))]


class BiologicalScorer:
    """Combines the activity and hemolysis models into the scalar objective S(P).

    This class is the single place the sign convention lives. Every other module --
    mutation landscape, QUBO builder, re-scoring, Pareto analysis -- calls through here,
    so there is exactly one definition of "better".
    """

    def __init__(
        self,
        activity_model: PropertyModel,
        hemolysis_model: PropertyModel,
        weights: ScoreWeights = ScoreWeights(),
    ) -> None:
        self.activity = activity_model
        self.hemolysis = hemolysis_model
        self.weights = weights
        self._cache: dict[str, tuple[float, float]] = {}

    @staticmethod
    def load(
        directory: Path | str = MODEL_DIR, weights: ScoreWeights = ScoreWeights()
    ) -> "BiologicalScorer":
        return BiologicalScorer(
            PropertyModel.load("activity", directory),
            PropertyModel.load("hemolysis", directory),
            weights,
        )

    def with_weights(self, weights: ScoreWeights) -> "BiologicalScorer":
        """A scorer sharing the same models but a different trade-off.

        The prediction cache is shared, since raw predictions do not depend on the
        weights -- only their combination does.
        """
        other = BiologicalScorer(self.activity, self.hemolysis, weights)
        other._cache = self._cache
        return other

    # -- raw predictions with memoisation -----------------------------------
    def _raw_pair(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(activity_raw, hemolysis_raw), computing only uncached sequences.

        Memoisation matters: building the mutation landscape evaluates O(N^2) mutants and
        the surrogate-validation study re-evaluates many of the same sequences.
        """
        missing = [s for s in dict.fromkeys(sequences) if s not in self._cache]
        if missing:
            a = self.activity.predict_raw(missing)
            h = self.hemolysis.predict_raw(missing)
            for seq, av, hv in zip(missing, a, h):
                self._cache[seq] = (float(av), float(hv))
        pairs = [self._cache[s] for s in sequences]
        return (
            np.array([p[0] for p in pairs], dtype=np.float64),
            np.array([p[1] for p in pairs], dtype=np.float64),
        )

    def breakdown(self, sequences: list[str]) -> ScoreBreakdown:
        """Full component breakdown for a list of sequences."""
        if not sequences:
            empty = np.zeros(0)
            return ScoreBreakdown([], empty, empty, empty, empty, empty)
        a_raw, h_raw = self._raw_pair(sequences)
        a_norm = self.activity.normalization.apply(a_raw)
        h_norm = self.hemolysis.normalization.apply(h_raw)
        score = self.weights.alpha * a_norm - self.weights.beta * h_norm
        return ScoreBreakdown(list(sequences), a_raw, h_raw, a_norm, h_norm, score)

    def score(self, sequences: list[str]) -> np.ndarray:
        """S(P) for each sequence. Larger is better."""
        return self.breakdown(sequences).score

    def score_one(self, sequence: str) -> float:
        return float(self.score([sequence])[0])

    def cache_size(self) -> int:
        return len(self._cache)


def fit_normalization_from_predictions(
    booster: xgb.XGBRegressor,
    X_train: np.ndarray,
    label: str,
) -> Normalization:
    """mu/sigma from training-split predictions, frozen thereafter."""
    preds = np.asarray(booster.predict(X_train), dtype=np.float64)
    return Normalization(
        mu=float(preds.mean()),
        sigma=float(preds.std(ddof=0)),
        source=(
            f"mean/std of {label} model predictions on the training split "
            f"(n={len(preds)}); frozen at training time"
        ),
    )
