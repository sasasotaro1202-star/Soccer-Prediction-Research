import pandas as pd
import pytest

from src.evaluation.baseline_comparison import compare_same_oos
from src.research.adoption_gate import independent_adoption_gate


def _pred(ids, probs):
    return pd.DataFrame({"match_id": ids, "H": [p[0] for p in probs], "D": [p[1] for p in probs], "A": [p[2] for p in probs]})


def _locked_holdout(**metrics):
    return {
        "locked": True,
        "selection_frozen": True,
        "used_for_selection": False,
        "used_for_calibration": False,
        "used_for_threshold_tuning": False,
        "development_end_utc": "2024-12-31T23:00:00Z",
        "holdout_start_utc": "2025-01-01T00:00:00Z",
        "n": 200,
        "same_oos": True,
        "baseline": metrics.get("baseline", {"logloss": 1.0, "brier": .70, "ece": .12, "accuracy": .50}),
        "candidate": metrics.get("candidate", {"logloss": .90, "brier": .60, "ece": .10, "accuracy": .52}),
    }


def test_same_oos_comparison_requires_identical_rows():
    oos = pd.DataFrame({"match_id": ["m1", "m2"], "target": [0, 2]})
    base = _pred(["m1", "m2"], [(0.6, .2, .2), (.2, .2, .6)])
    cand = _pred(["m1", "m2"], [(0.7, .15, .15), (.15, .15, .7)])
    out = compare_same_oos(oos, base, cand)
    assert out["same_oos"] is True
    assert out["n"] == 2


def test_same_oos_rejects_missing_rows():
    oos = pd.DataFrame({"match_id": ["m1", "m2"], "target": [0, 2]})
    base = _pred(["m1", "m2"], [(0.6, .2, .2), (.2, .2, .6)])
    cand = _pred(["m1"], [(0.7, .15, .15)])
    with pytest.raises(ValueError):
        compare_same_oos(oos, base, cand)


def test_independent_gate_adopts_only_with_holdout_improvement():
    development = {"selected": "candidate"}
    holdout = _locked_holdout()
    assert independent_adoption_gate(development, holdout)["status"] == "ADOPT"


def test_independent_gate_rejects_candidate_with_worse_logloss():
    holdout = _locked_holdout(candidate={"logloss": 1.01, "brier": .60, "ece": .10, "accuracy": .52})
    assert independent_adoption_gate({}, holdout)["status"] == "REJECT"
