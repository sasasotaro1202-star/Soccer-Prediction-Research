        "meta_logloss": [0.485, 0.582, 0.63],
        "confidence_logloss": [0.5, 0.6, 0.65],
    })
    assert mod._development_improvement_rate(stronger) == 1.0


def test_predictability_oos_skips_block_when_prior_outcome_is_not_mature():
    rows = [
        _row(i, risk=0.2, wrong=(i % 5 == 0))
        for i in range(360)
    ]
    frame = pd.DataFrame(rows)
    frame["kickoff_utc"] = pd.date_range(
        "2026-01-01", periods=len(frame), freq="6h", tz="UTC"
    )
    frame["prediction_pit_cutoff_utc"] = (
        frame["kickoff_utc"] - pd.Timedelta(hours=2)
    )
    frame["experience_available_at_utc"] = (
        frame["kickoff_utc"] + pd.Timedelta(hours=2)
    )
    frame.loc[119, "experience_available_at_utc"] = (
        frame.loc[120, "prediction_pit_cutoff_utc"] + pd.Timedelta(hours=1)
    )
    state = analyze(frame)
    assert state["oos_blocks"]
    # The first block is skipped because filtering the immature row leaves
    # only 119 mature training rows, below the 120-row minimum.
    assert state["oos_blocks"][0]["block"] == 2
    assert state["oos_blocks"][0]["excluded_immature_training_rows"] == 0