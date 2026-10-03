

def test_predictability_workflow_warmup_uses_frozen_holdout_key():
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "soccer-predictability-research.yml"
    text = workflow.read_text(encoding="utf-8")
    assert '"frozen_holdout_touched": False' in text
    assert '"locked_holdout_touched": False' not in text


def test_calibration_gate_requires_meaningful_stable_development():
    from src.research import predictability_calibration as mod

    development = pd.DataFrame({
        "calibrated_logloss": [0.4, 0.5, 0.7],
        "raw_logloss": [0.5, 0.6, 0.65],
    })
    assert mod._development_improvement_rate(development) == 2 / 3

    stronger = pd.DataFrame({
        "calibrated_logloss": [0.485, 0.582, 0.63],
        "raw_logloss": [0.5, 0.6, 0.65],
    })
    assert mod._development_improvement_rate(stronger) == 1.0


def test_calibration_oos_skips_block_when_prior_outcome_is_not_mature():
    df = _rows(360)
    df.loc[119, "experience_available_at_utc"] = (
        df.loc[120, "prediction_pit_cutoff_utc"] + pd.Timedelta(hours=1)
    )
    state = calibrate(df, history_rows=120, block_size=60)
    assert state["oos_blocks"]
    # The first block is skipped because filtering the immature row leaves
    # only 119 mature training rows, below the 120-row minimum.
    assert state["oos_blocks"][0]["block"] == 2
    assert state["oos_blocks"][0]["excluded_immature_training_rows"] == 0


def test_prior_mature_training_excludes_outcomes_not_mature_by_target_cutoff():
    frame = pd.DataFrame({
        "prediction_pit_cutoff_utc": [
            "2026-01-01T08:00:00Z",
            "2026-01-01T09:00:00Z",
            "2026-01-01T10:00:00Z",
        ],
        "experience_available_at_utc": [
            "2026-01-01T09:00:00Z",
            "2026-01-01T11:00:00Z",
            "2026-01-01T10:00:00Z",
        ],
    })
    eligible = filter_prior_mature_training(
        frame, pd.Timestamp("2026-01-01T10:00:00Z")
    )
    assert len(eligible) == 2
    assert eligible["prediction_pit_cutoff_utc"].tolist() == [
        "2026-01-01T08:00:00Z",
        "2026-01-01T09:00:00Z",
    ]


def test_prior_mature_training_missing_maturity_fails_closed():
    frame = pd.DataFrame({
        "prediction_pit_cutoff_utc": ["2026-01-01T08:00:00Z"],
        "experience_available_at_utc": [pd.NaT],
    })
    with pytest.raises(RuntimeError, match="maturity"):
        filter_prior_mature_training(
            frame, pd.Timestamp("2026-01-01T10:00:00Z")
        )