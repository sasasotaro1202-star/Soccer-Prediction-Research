import numpy as np
import pandas as pd
import pytest

from src.research.external_forecast_eval import (
    evaluate_external_source,
    evaluate_internal_vs_external,
    evaluate_multiple_sources,
)


def _frame():
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "source": ["optA"] * 4,
            "prediction_time_utc": [
                "2026-09-01T12:00:00Z",
                "2026-09-02T12:00:00Z",
                "2026-09-03T12:00:00Z",
                "2026-09-04T12:00:00Z",
            ],
            "source_available_at_utc": [
                "2026-09-01T09:00:00Z",
                "2026-09-02T09:00:00Z",
                "2026-09-03T09:00:00Z",
                "2026-09-04T09:00:00Z",
            ],
            "p_home": [0.8, 0.2, 0.2, 0.4],
            "p_draw": [0.1, 0.6, 0.2, 0.3],
            "p_away": [0.1, 0.2, 0.6, 0.3],
            "base_p_home": [0.6, 0.3, 0.3, 0.5],
            "base_p_draw": [0.2, 0.5, 0.3, 0.3],
            "base_p_away": [0.2, 0.2, 0.4, 0.2],
            "actual_result": ["H", "D", "A", "D"],
            "provenance_url": ["https://example.test/1"] * 4,
        }
    )


def test_evaluate_external_source_returns_standard_metrics():
    result = evaluate_external_source(_frame())
    assert result["rows"] == 4
    assert result["pit_verified_rows"] == 4
    for key in ("accuracy", "logloss", "brier", "ece"):
        assert key in result
        assert np.isfinite(result[key])


def test_internal_external_and_blend_are_scored_on_same_rows():
    result = evaluate_internal_vs_external(_frame(), blend_weight=0.25)
    assert result["rows"] == 4
    assert result["external"]["n"] == 4
    assert result["internal"]["n"] == 4
    assert result["blend"]["n"] == 4
    assert np.isfinite(result["blend"]["logloss"])


def test_multiple_sources_are_evaluated_independently():
    frame = _frame()
    second = frame.copy()
    second["source"] = "market"
    all_rows = pd.concat([frame, second], ignore_index=True)
    result = evaluate_multiple_sources(all_rows)
    assert sorted(result) == ["market", "optA"]
    assert all(result[s]["pit_verified_rows"] == 4 for s in result)


def test_missing_actual_result_fails_closed():
    frame = _frame().drop(columns=["actual_result"])
    with pytest.raises(Exception):
        evaluate_external_source(frame)
