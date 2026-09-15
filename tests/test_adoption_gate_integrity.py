from src.research.adoption_gate import independent_adoption_gate


def _valid_holdout(**overrides):
    payload = {
        "n": 200,
        "same_oos": True,
        "locked": True,
        "selection_frozen": True,
        "used_for_selection": False,
        "used_for_calibration": False,
        "used_for_threshold_tuning": False,
        "development_end_utc": "2024-12-31T23:59:59Z",
        "holdout_start_utc": "2025-01-01T00:00:00Z",
        "baseline": {"logloss": 1.0, "brier": 0.25, "ece": 0.10, "accuracy": 0.50},
        "candidate": {"logloss": 0.90, "brier": 0.24, "ece": 0.09, "accuracy": 0.51},
    }
    payload.update(overrides)
    return payload


def test_valid_locked_holdout_can_adopt():
    result = independent_adoption_gate({}, _valid_holdout())
    assert result["status"] == "ADOPT"
    assert result["holdout_integrity_verified"] is True


def test_missing_lock_metadata_fails_closed():
    holdout = _valid_holdout()
    holdout.pop("locked")
    result = independent_adoption_gate({}, holdout)
    assert result == {
        "status": "HOLD",
        "reason": "holdout_not_explicitly_locked",
        "oos_claimed": False,
        "promotion_authority": "deterministic_research_engine",
    }


def test_holdout_used_for_selection_is_blocked():
    result = independent_adoption_gate({}, _valid_holdout(used_for_selection=True))
    assert result["status"] == "HOLD"
    assert result["reason"] == "holdout_was_used_for_selection"


def test_holdout_overlap_is_blocked():
    result = independent_adoption_gate(
        {},
        _valid_holdout(holdout_start_utc="2024-12-31T23:00:00Z"),
    )
    assert result["status"] == "HOLD"
    assert result["reason"] == "holdout_overlaps_development_period"
