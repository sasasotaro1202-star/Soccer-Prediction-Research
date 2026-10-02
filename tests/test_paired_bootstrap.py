import pandas as pd
import pytest

from src.evaluation.baseline_comparison import paired_bootstrap


def _pred(ids, probs):
    return pd.DataFrame({
        "match_id": ids,
        "H": [p[0] for p in probs],
        "D": [p[1] for p in probs],
        "A": [p[2] for p in probs],
    })


def test_paired_bootstrap_is_deterministic_and_preserves_pairing():
    oos = pd.DataFrame({
        "match_id": ["m1", "m2", "m3", "m4"],
        "target": [0, 1, 2, 0],
    })
    base = _pred(
        ["m1", "m2", "m3", "m4"],
        [(0.70, 0.20, 0.10), (0.20, 0.70, 0.10), (0.10, 0.20, 0.70), (0.55, 0.30, 0.15)],
    )
    cand = _pred(
        ["m1", "m2", "m3", "m4"],
        [(0.72, 0.18, 0.10), (0.15, 0.75, 0.10), (0.10, 0.25, 0.65), (0.60, 0.25, 0.15)],
    )
    first = paired_bootstrap(oos, base, cand, n_resamples=500, seed=7)
    second = paired_bootstrap(oos, base, cand, n_resamples=500, seed=7)
    assert first == second
    assert first["same_oos"] is True
    assert first["research_only"] is True
    assert first["selection_allowed"] is False
    assert first["ci_95_low"] <= first["observed_delta_candidate_minus_baseline"] <= first["ci_95_high"]


def test_paired_bootstrap_rejects_unsupported_metric():
    oos = pd.DataFrame({"match_id": ["m1"], "target": [0]})
    base = _pred(["m1"], [(0.7, 0.2, 0.1)])
    cand = _pred(["m1"], [(0.8, 0.1, 0.1)])
    with pytest.raises(ValueError, match="logloss and brier"):
        paired_bootstrap(oos, base, cand, metric="accuracy")
