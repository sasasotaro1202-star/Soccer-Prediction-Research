from __future__ import annotations

import math
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler



class EloLogisticClassifier:
    """Low-dimensional Elo-only candidate for cross-regime robustness.

    The estimator intentionally ignores richer features. It learns the mapping
    from prediction-time Elo context to H/D/A outcomes using only pre-kickoff
    numeric state already present in the feature frame.
    """

    _FEATURES = ("elo_diff", "comp_elo_diff", "home_elo_expected")

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.5, random_state=random_state)),
        ])
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def _select(self, X):
        missing = [c for c in self._FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Elo candidate missing required columns: {missing}")
        return X[list(self._FEATURES)]

    def fit(self, X, y):
        self.model.fit(self._select(X), np.asarray(y, dtype=int))
        self._fitted_classes = np.asarray(getattr(self.model, "classes_", self.classes_), dtype=int)
        return self

    def predict_proba(self, X):
        raw = np.asarray(self.model.predict_proba(self._select(X)), dtype=float)
        out = np.zeros((len(X), 3), dtype=float)
        for j, cls in enumerate(self._fitted_classes):
            cls = int(cls)
            if cls in (0, 1, 2):
                out[:, cls] = raw[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError("Elo candidate produced an invalid probability row")
        return out / row_sum


class RecencyLogisticClassifier:
    """Logistic candidate with monotone recency weighting inside each fit slice."""

    def __init__(self, random_state: int = 42, half_life_rows: float = 600.0):
        self.random_state = random_state
        self.half_life_rows = max(float(half_life_rows), 1.0)
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.75, random_state=random_state)),
        ])
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def fit(self, X, y):
        yv = np.asarray(y, dtype=int)
        n = len(yv)
        positions = np.arange(n, dtype=float)
        weights = np.exp((positions - max(0, n - 1)) / self.half_life_rows)
        weights = weights / max(float(weights.mean()), 1e-12)
        self.model.fit(X, yv, model__sample_weight=weights)
        self._fitted_classes = np.asarray(getattr(self.model, "classes_", self.classes_), dtype=int)
        return self

    def predict_proba(self, X):
        raw = np.asarray(self.model.predict_proba(X), dtype=float)
        out = np.zeros((len(X), 3), dtype=float)
        for j, cls in enumerate(self._fitted_classes):
            cls = int(cls)
            if cls in (0, 1, 2):
                out[:, cls] = raw[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError("Recency logistic produced an invalid probability row")
        return out / row_sum



class QuantileLogisticClassifier:
    """Rank-normalized logistic challenger for cross-competition scale robustness.

    Quantile normalization is fitted inside each chronological training slice,
    so it cannot learn the distribution of future/OOS observations. Mapping to a
    normal reference reduces sensitivity to league-specific feature scale while
    retaining a stable linear decision surface.
    """

    def __init__(self, random_state: int = 42, n_quantiles: int = 64):
        self.random_state = random_state
        self.n_quantiles = max(int(n_quantiles), 8)
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("quantile", QuantileTransformer(
                n_quantiles=self.n_quantiles,
                output_distribution="normal",
                random_state=random_state,
            )),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, C=0.35, random_state=random_state)),
        ])
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def fit(self, X, y):
        self.model.fit(X, np.asarray(y, dtype=int))
        self._fitted_classes = np.asarray(getattr(self.model, "classes_", self.classes_), dtype=int)
        return self

    def predict_proba(self, X):
        raw = np.asarray(self.model.predict_proba(X), dtype=float)
        out = np.zeros((len(X), 3), dtype=float)
        for j, cls in enumerate(self._fitted_classes):
            cls = int(cls)
            if cls in (0, 1, 2):
                out[:, cls] = raw[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0) or not np.isfinite(out).all():
            raise ValueError("Quantile logistic produced invalid probabilities")
        return out / row_sum


class DynamicEloLogisticClassifier:
    """Low-dimensional adaptive-strength candidate using PIT-safe dynamic-Elo state."""

    _FEATURES = ("dynamic_elo_diff", "dynamic_home_elo_expected", "elo_diff", "home_elo_expected")

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2200, C=0.45, random_state=random_state)),
        ])
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def _select(self, X):
        missing = [c for c in self._FEATURES if c not in X.columns]
        if not missing:
            return X[list(self._FEATURES)].copy()
        # Generic compatibility fixtures may not contain the new Dynamic Elo columns.
        # Fall back only to existing PIT-safe Elo state; real production/OOS frames
        # provide the dedicated Dynamic Elo features.
        required_fallback = ["elo_diff", "home_elo_expected"]
        if all(c in X.columns for c in required_fallback):
            out = X[required_fallback].copy()
            out.columns = ["dynamic_elo_diff", "dynamic_home_elo_expected"]
            out["elo_diff"] = X["elo_diff"]
            out["home_elo_expected"] = X["home_elo_expected"]
            return out[list(self._FEATURES)]
        raise ValueError(f"Dynamic Elo candidate missing required columns: {missing}")

    def fit(self, X, y):
        self.model.fit(self._select(X), np.asarray(y, dtype=int))
        self._fitted_classes = np.asarray(getattr(self.model, "classes_", self.classes_), dtype=int)
        return self

    def predict_proba(self, X):
        raw = np.asarray(self.model.predict_proba(self._select(X)), dtype=float)
        out = np.zeros((len(X), 3), dtype=float)
        for j, cls in enumerate(self._fitted_classes):
            if int(cls) in (0, 1, 2):
                out[:, int(cls)] = raw[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0) or not np.isfinite(out).all():
            raise ValueError("Dynamic Elo candidate produced invalid probabilities")
        return out / row_sum


class HierarchicalEloLogisticClassifier:
    """Competition-Elo model shrunk toward global strength for sparse competitions."""

    _FEATURES = ("comp_elo_shrunk_diff", "home_comp_elo_shrunk_expected", "dynamic_elo_diff")

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2200, C=0.4, random_state=random_state)),
        ])
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def _select(self, X):
        missing = [c for c in self._FEATURES if c not in X.columns]
        if not missing:
            return X[list(self._FEATURES)].copy()
        # Compatibility fixture fallback: when dedicated hierarchical features are
        # absent, derive a conservative proxy from existing PIT-safe Elo state.
        if {"comp_elo_diff", "home_elo_expected", "elo_diff"}.issubset(X.columns):
            out = X[["comp_elo_diff", "home_elo_expected", "elo_diff"]].copy()
            out.columns = list(self._FEATURES)
            return out
        raise ValueError(f"Hierarchical Elo candidate missing required columns: {missing}")

    def fit(self, X, y):
        self.model.fit(self._select(X), np.asarray(y, dtype=int))
        self._fitted_classes = np.asarray(getattr(self.model, "classes_", self.classes_), dtype=int)
        return self

    def predict_proba(self, X):
        raw = np.asarray(self.model.predict_proba(self._select(X)), dtype=float)
        out = np.zeros((len(X), 3), dtype=float)
        for j, cls in enumerate(self._fitted_classes):
            if int(cls) in (0, 1, 2):
                out[:, int(cls)] = raw[:, j]
        row_sum = out.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0) or not np.isfinite(out).all():
            raise ValueError("Hierarchical Elo candidate produced invalid probabilities")
        return out / row_sum


def candidates(random_state: int = 42):
    """Return a compact, diverse and leakage-safe candidate set.

    Feature selection is fitted inside each temporal training slice. A variance
    filter removes constant columns before univariate scoring, avoiding unstable
    F-statistics and reducing needless work. Conservative tree leaf sizes and
    regularization are intentional: robustness on unseen seasons is preferred to
    fitting historical noise.
    """
    return {
        "elo_logistic": EloLogisticClassifier(random_state=random_state),
        "dynamic_elo_logistic": DynamicEloLogisticClassifier(random_state=random_state),
        "hierarchical_elo_logistic": HierarchicalEloLogisticClassifier(random_state=random_state),
        "recency_logistic": RecencyLogisticClassifier(random_state=random_state),
        "quantile_logistic": QuantileLogisticClassifier(random_state=random_state),
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)),
        ]),
        "logistic_select": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=1e-12)),
            ("select", SelectPercentile(score_func=f_classif, percentile=30)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.5, random_state=random_state)),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=8, max_features="sqrt", n_jobs=-1, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_features="sqrt", n_jobs=-1, random_state=random_state, class_weight="balanced_subsample")),
        ]),
        "hist_gb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=random_state)),
        ]),
        "logistic_l2_strong": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.15, random_state=random_state)),
        ]),
        "hist_gb_robust": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=320,
                learning_rate=0.035,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=random_state,
            )),
        ]),
    }


def poisson_score_probs(home_lambda: float, away_lambda: float, max_goals: int = 7) -> np.ndarray:
    vals = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = np.exp(-home_lambda) * home_lambda**h / math.factorial(h)
            p *= np.exp(-away_lambda) * away_lambda**a / math.factorial(a)
            vals.append((h, a, p))
    total = sum(v[2] for v in vals) or 1.0
    return np.array([[h, a, p / total] for h, a, p in vals])
