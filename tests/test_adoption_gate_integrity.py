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
        "pit_status": "PASS",
        "pit_violations": 0,
        "baseline": {"logloss": 1.0, "brier": 0.25, "ece": 0.10, "accuracy": 0.50},
        "candidate": {"logloss": 0.90, "brier": 0.24, "ece": 0.09, "accuracy": 0.51},
    }
    payload.update(overrides)
    return payload


def _stability_folds():
    return [
        {
            "league": "EPL",
            "season": "2022",
            "oos_start_utc": "2024-01-01T00:00:00Z",
            "oos_end_utc": "2024-01-31T23:59:59Z",
            "baseline": {"logloss": 1.00, "brier": 0.25, "accuracy": 0.50},
            "candidate": {"logloss": 0.95, "brier": 0.24, "accuracy": 0.51},
        },
        {
            "league": "Bundesliga",
            "season": "2023",
            "oos_start_utc": "2024-02-01T00:00:00Z",
            "oos_end_utc": "2024-02-29T23:59:59Z",
            "baseline": {"logloss": 1.02, "brier": 0.25, "accuracy": 0.50},
            "candidate": {"logloss": 0.98, "brier": 0.24, "accuracy": 0.51},
        },
        {
            "league": "Serie A",
            "season": "2024",
            "oos_start_utc": "2024-03-01T00:00:00Z",
            "oos_end_utc": "2024-03-31T23:59:59Z",
            "baseline": {"logloss": 1.01, "brier": 0.25, "accuracy": 0.50},
            "candidate": {"logloss": 0.99, "brier": 0.24, "accuracy": 0.51},
        },
    ]


def test_valid_locked_holdout_can_adopt():
    result = independent_adoption_gate(
        {"development_oos": True},
        _valid_holdout(),
        stability_folds=_stability_folds(),
    )
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
    result = independent_adoption_gate({"development_oos": True}, _valid_holdout(used_for_selection=True), stability_folds=_stability_folds())
    assert result["status"] == "HOLD"
    assert result["reason"] == "holdout_was_used_for_selection"


def test_holdout_overlap_is_blocked():
    result = independent_adoption_gate(
        {"development_oos": True},
        _valid_holdout(holdout_start_utc="2024-12-31T23:00:00Z"),
    )
    assert result["status"] == "HOLD"
    assert result["reason"] == "holdout_overlaps_development_period"


def test_naive_holdout_boundary_fails_closed():
    result = independent_adoption_gate(
        {"development_oos": True},
        _valid_holdout(development_end_utc="2024-12-31T23:59:59"),
    )
    assert result["status"] == "HOLD"
    assert result["reason"] == "holdout_temporal_boundaries_missing_or_invalid"


def test_offset_boundaries_are_compared_in_utc():
    result = independent_adoption_gate(
        {"development_oos": True},
        _valid_holdout(
            development_end_utc="2024-12-31T21:00:00-02:00",
            holdout_start_utc="2025-01-01T00:00:00Z",
        ),
        stability_folds=_stability_folds(),
    )
    assert result["status"] == "ADOPT"
    assert result["holdout_integrity_verified"] is True


def test_adoption_requires_explicit_development_evidence():
    result = independent_adoption_gate({}, _valid_holdout(), stability_folds=_stability_folds())
    assert result["status"] == "HOLD"
    assert result["reason"] == "development_evidence_missing_or_invalid"


def test_adoption_rejects_non_mapping_development_evidence():
    for invalid in ([], [("development_oos", True)], "development"):
        result = independent_adoption_gate(invalid, _valid_holdout(), stability_folds=_stability_folds())
        assert result["status"] == "HOLD"
        assert result["reason"] == "development_evidence_missing_or_invalid"


def test_adoption_rejects_false_or_non_boolean_development_oos():
    for invalid in (False, 1, "true", None):
        result = independent_adoption_gate(
            {"development_oos": invalid},
            _valid_holdout(),
            stability_folds=_stability_folds(),
        )
        assert result["status"] == "HOLD"
        assert result["reason"] == "development_evidence_missing_or_invalid"


def test_adoption_rejects_non_numeric_holdout_metrics():
    result = independent_adoption_gate(
        {"development_oos": True},
        _valid_holdout(
            baseline={"logloss": "bad", "brier": 0.25, "ece": 0.10, "accuracy": 0.50}
        ),
        stability_folds=_stability_folds(),
    )
    assert result["status"] == "HOLD"
    assert result["reason"] == "non_numeric_holdout_metrics"


def test_adoption_requires_zero_pit_violations():
    result = independent_adoption_gate(
        {"development_oos": True},
        _valid_holdout(pit_violations=1),
        stability_folds=_stability_folds(),
    )
    assert result["status"] == "HOLD"
    assert result["reason"] == "pit_violations_present"


def test_adoption_requires_explicit_stability_evidence():
    result = independent_adoption_gate({"development_oos": True}, _valid_holdout())
    assert result["status"] == "HOLD"
    assert result["reason"] == "stability_evidence_missing"


def test_adoption_requires_reference_relative_improvement_thresholds():
    result = independent_adoption_gate(
        {"development_oos": True},
        _valid_holdout(
            baseline={"logloss": 1.0, "brier": 0.25, "ece": 0.10, "accuracy": 0.50},
            candidate={"logloss": 0.98, "brier": 0.249, "ece": 0.10, "accuracy": 0.50},
        ),
        stability_folds=_stability_folds(),
    )
    assert result["status"] == "REJECT"
    assert result["primary_logloss_relative_improvement"] < 0.03
    assert result["auxiliary_brier_relative_improvement"] < 0.01


def test_adoption_rejects_invalid_holdout_row_count():
    for invalid in (None, True, "200", -1):
        result = independent_adoption_gate(
            {"development_oos": True},
            _valid_holdout(n=invalid),
            stability_folds=_stability_folds(),
        )
        assert result["status"] == "HOLD"
        assert result["reason"] == "holdout_row_count_missing_or_invalid"


def test_adoption_rejects_non_finite_holdout_metrics():
    result = independent_adoption_gate(
        {"development_oos": True},
        _valid_holdout(
            candidate={"logloss": 0.90, "brier": 0.24, "ece": float("nan"), "accuracy": 0.51}
        ),
        stability_folds=_stability_folds(),
    )
    assert result["status"] == "HOLD"
    assert result["reason"] == "non_finite_holdout_metrics"
