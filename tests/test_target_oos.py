from __future__ import annotations

import pandas as pd

from src.research.target_oos import write_target_oos_artifacts


def test_target_oos_artifacts_are_separate(tmp_path):
    score = pd.DataFrame([
        {
            "oos_start": "2025-01-01T00:00:00Z",
            "oos_end": "2025-02-01T00:00:00Z",
            "n": 100,
            "over_2_5_logloss": 0.61,
            "over_2_5_brier": 0.21,
            "btts_logloss": 0.64,
            "btts_brier": 0.22,
        },
    ])
    result = write_target_oos_artifacts(score, tmp_path)
    assert result["status"] == "PASS"

    ou = pd.read_csv(tmp_path / "ou_oos_metrics.csv")
    btts = pd.read_csv(tmp_path / "btts_oos_metrics.csv")
    assert list(ou["target"]) == ["O/U_2_5"]
    assert list(btts["target"]) == ["BTTS"]
    assert float(ou.iloc[0]["logloss"]) == 0.61
    assert float(btts.iloc[0]["brier"]) == 0.22
    assert "score_logloss" not in ou.columns
    assert "score_logloss" not in btts.columns


def test_missing_binary_score_metrics_fail_closed(tmp_path):
    score = pd.DataFrame([{"oos_start": "2025-01-01", "oos_end": "2025-02-01", "n": 100}])
    try:
        write_target_oos_artifacts(score, tmp_path)
    except ValueError as exc:
        assert "missing columns" in str(exc).lower()
    else:
        raise AssertionError("missing target metrics must fail closed")
