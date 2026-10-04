from src.research.stability_gate import evaluate_stability


def _fold(
    league,
    season,
    ll_base,
    ll_cand,
    b_base=.25,
    b_cand=.24,
    a_base=.50,
    a_cand=.51,
    start="2024-01-01T00:00:00Z",
    end="2024-01-31T23:59:59Z",
):
    return {
        "league": league,
        "season": season,
        "oos_start_utc": start,
        "oos_end_utc": end,
        "baseline": {"logloss": ll_base, "brier": b_base, "accuracy": a_base},
        "candidate": {"logloss": ll_cand, "brier": b_cand, "accuracy": a_cand},
    }


def test_stability_requires_multiple_leagues_and_seasons():
    folds = [
        _fold("EPL", 2022, 1, .9, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("EPL", 2023, 1, .9, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        _fold("EPL", 2024, 1, .9, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["reason"] == "too_few_unique_leagues"


def test_stability_passes_when_improvement_is_repeatable():
    folds = [
        _fold("EPL", 2022, 1.00, .95, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Bundesliga", 2023, 1.02, .98, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        _fold("Serie A", 2024, 1.01, .99, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "PASS"
    assert result["worst_logloss_delta"] <= 0
    assert result["min_improved_fraction"] == .70
    assert result["chronology_verified"] is True


def test_project_70_percent_gate_rejects_two_of_three_metric_non_regression():
    folds = [
        _fold("EPL", 2022, 1.00, .95, b_base=.25, b_cand=.24, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Bundesliga", 2023, 1.02, .98, b_base=.25, b_cand=.24, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        _fold("Serie A", 2024, 1.01, .99, b_base=.25, b_cand=.26, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["brier_improved_or_equal_folds"] == 2
    assert result["min_improved_fraction"] == .70


def test_weaker_than_project_fraction_is_rejected():
    folds = [
        _fold("EPL", 2022, 1.00, .95, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Bundesliga", 2023, 1.02, .98, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        _fold("Serie A", 2024, 1.01, .99, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
    ]
    result = evaluate_stability(folds, min_improved_fraction=2 / 3)
    assert result["status"] == "HOLD"
    assert result["reason"] == "improved_fraction_below_project_minimum_or_invalid"


def test_stability_requires_explicit_chronological_boundaries():
    folds = [
        _fold("EPL", 2022, 1, .9, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Bundesliga", 2023, 1, .9, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        {
            **_fold("Serie A", 2024, 1, .9, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
            "oos_start_utc": None,
            "oos_end_utc": None,
        },
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["reason"] == "chronology_evidence_missing_or_invalid"


def test_stability_rejects_overlapping_chronological_folds():
    folds = [
        _fold("EPL", 2022, 1, .9, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Bundesliga", 2023, 1, .9, start="2024-01-15T00:00:00Z", end="2024-02-29T23:59:59Z"),
        _fold("Serie A", 2024, 1, .9, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["reason"] == "chronology_folds_overlap_or_reverse"


def test_stability_rejects_non_finite_metrics():
    folds = [
        _fold("EPL", 2022, 1, .9, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Bundesliga", 2023, 1, .9, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        _fold("Serie A", 2024, 1, float("nan"), start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["reason"] == "non_finite_fold_metrics"


def test_one_logloss_regression_blocks_stability():
    folds = [
        _fold("EPL", 2022, 1.00, .95, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Bundesliga", 2023, 1.02, .98, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        _fold("Serie A", 2024, 1.01, 1.02, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["worst_logloss_delta"] > 0


def test_pipe_delimited_fold_coverage_is_unioned():
    folds = [
        _fold("EPL|Bundesliga", "2022|2023", 1.00, .95, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Serie A", "2024", 1.02, .98, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        _fold("Ligue 1", "2025", 1.01, .99, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "PASS"
    assert result["unique_leagues"] == ["Bundesliga", "EPL", "Ligue 1", "Serie A"]
    assert result["unique_seasons"] == ["2022", "2023", "2024", "2025"]
