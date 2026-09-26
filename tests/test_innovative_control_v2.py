from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.innovative_control_v2 import (
    _architectures,
    _bootstrap_ci,
    _route,
    _safe_probs,
)


def test_v2_ablation_matrix_matches_spec():
    arches = _architectures()
    assert set(arches) == set("BCDEFGHIJ")
    assert arches["B"] == dict(use_disagreement=True, use_predictability=False, use_failure=False, use_drift=False)
    assert arches["C"] == dict(use_disagreement=False, use_predictability=True, use_failure=False, use_drift=False)
    assert arches["D"] == dict(use_disagreement=False, use_predictability=False, use_failure=True, use_drift=False)
    assert arches["E"] == dict(use_disagreement=False, use_predictability=False, use_failure=False, use_drift=True)
    assert arches["F"] == dict(use_disagreement=True, use_predictability=True, use_failure=False, use_drift=False)
    assert arches["G"] == dict(use_disagreement=True, use_predictability=False, use_failure=True, use_drift=False)
    assert arches["H"] == dict(use_disagreement=False, use_predictability=True, use_failure=True, use_drift=False)
    assert arches["I"] == dict(use_disagreement=False, use_predictability=False, use_failure=True, use_drift=True)
    assert arches["J"] == dict(use_disagreement=True, use_predictability=True, use_failure=True, use_drift=True)


def test_route_probability_safety_and_shape():
    n = 8
    probs = {
        name: _safe_probs(np.random.default_rng(42 + i).random((n, 3)))
        for i, name in enumerate(("logistic", "elo_logistic", "recency_logistic", "extra_trees", "hist_gb"))
    }
    state = pd.DataFrame({
        "predictability": np.linspace(0.1, 0.9, n),
        "feature_drift": np.linspace(0.0, 1.0, n),
        **{f"{name}_disagreement": np.full(n, 0.05) for name in probs},
    })
    risks = {name: np.full(n, 0.2) for name in probs}
    weights = {name: 0.2 for name in probs}
    p, confidence = _route(
        probs, state, risks, weights,
        use_disagreement=True,
        use_predictability=True,
        use_failure=True,
        use_drift=True,
    )
    assert p.shape == (n, 3)
    assert confidence.shape == (n,)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_block_bootstrap_is_fail_closed_with_too_few_blocks():
    lo, hi = _bootstrap_ci([{"logloss": -0.1}] * 4, "logloss")
    assert lo is None and hi is None
