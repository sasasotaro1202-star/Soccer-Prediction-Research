from src.research.stability_gate import evaluate_stability


def _fold(league, season, ll_base, ll_cand, b_base=.25, b_cand=.24, a_base=.50, a_cand=.51,
          start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"):
    return {
        "league": league, "season": season, "oos_start_utc": start, "oos_end_utc": end,
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


def test_stability_requires_explicit_chronological_boundaries():
    folds = [
        _fold("EPL", 2022, 1, .9, start="2024-01-01T00:00:00Z", end="2024-01-31T23:59:59Z"),
        _fold("Bundesliga", 2023, 1, .9, start="2024-02-01T00:00:00Z", end="2024-02-29T23:59:59Z"),
        {**_fold("Serie A", 2024, 1, .9, start="2024-03-01T00:00:00Z", end="2024-03-31T23:59:59Z"), "oos_start_utc": None, "oos_end_utc": None},
    ]
    result = evaluate_stability(folds)
    assert result["status"] == "HOLD"
    assert result["reason"] == "chronology_evidence_missing_or_invalid"
