from src.research.stability_gate import evaluate_stability


def _fold(league, season, ll_base, ll_cand, b_base=.25, b_cand=.24, a_base=.50, a_cand=.51):
    return {
        "league": league,
        "season": season,
        "baseline": {"logloss": ll_base, "brier": b_base, "accuracy": a_base},
        "candidate": {"logloss": ll_cand, "brier": b_cand, "accuracy": a_cand},
    }


def test_stability_requires_multiple_leagues_and_seasons():
    folds = [_fold("EPL", 2022, 1, .9), _fold("EPL", 2023, 1, .9), _fold("EPL", 2024, 1, .9)]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["reason"] == "too_few_unique_leagues"


def test_stability_passes_when_improvement_is_repeatable():
    folds = [
        _fold("EPL", 2022, 1.00, .95),
        _fold("Bundesliga", 2023, 1.02, .98),
        _fold("Serie A", 2024, 1.01, .99),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "PASS"
    assert result["worst_logloss_delta"] <= 0


def test_one_logloss_regression_blocks_stability():
    folds = [
        _fold("EPL", 2022, 1.00, .95),
        _fold("Bundesliga", 2023, 1.02, .98),
        _fold("Serie A", 2024, 1.01, 1.02),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["worst_logloss_delta"] > 0


def test_pipe_delimited_fold_coverage_is_unioned():
    folds = [
        _fold("EPL|Bundesliga", "2022|2023", 1.00, .95),
        _fold("Serie A", "2024", 1.02, .98),
        _fold("Ligue 1", "2025", 1.01, .99),
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "PASS"
    assert result["unique_leagues"] == ["Bundesliga", "EPL", "Ligue 1", "Serie A"]
    assert result["unique_seasons"] == ["2022", "2023", "2024", "2025"]
