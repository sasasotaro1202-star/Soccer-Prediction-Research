import numpy as np
import pandas as pd

from src.models.baselines import candidates


def test_all_classification_candidates_fit_multiclass_data():
    X = pd.DataFrame({
        "x1": [0, 1, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "x2": [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
        "elo_diff": [-200, -100, -50, 0, 25, 50, 75, 100, 125, 150, 175, 200],
        "comp_elo_diff": [-150, -75, -25, 0, 10, 20, 35, 50, 65, 80, 95, 110],
        "home_elo_expected": [0.30, 0.40, 0.45, 0.50, 0.53, 0.57, 0.61, 0.65, 0.68, 0.71, 0.74, 0.77],
    })
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    for name, model in candidates(random_state=42).items():
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3), name
        assert np.all(np.isfinite(proba)), name
        assert np.allclose(proba.sum(axis=1), 1.0), name
