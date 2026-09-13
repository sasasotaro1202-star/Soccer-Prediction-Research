from __future__ import annotations

import math
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def candidates(random_state: int = 42):
    """Return a compact, diverse and leakage-safe candidate set.

    Feature selection is fitted inside each temporal training slice. A variance
    filter removes constant columns before univariate scoring, avoiding unstable
    F-statistics and reducing needless work. Conservative tree leaf sizes and
    regularization are intentional: robustness on unseen seasons is preferred to
    fitting historical noise.
    """
    return {
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
