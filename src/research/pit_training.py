from __future__ import annotations

import pandas as pd


def filter_prior_mature_training(frame: pd.DataFrame, target_cutoff) -> pd.DataFrame:
    """Keep prior states whose outcomes were mature by the target cutoff.

    Missing or invalid PIT timestamps fail closed. Rows whose own prediction
    cutoff is not strictly earlier than the target are excluded.
    """
    required = {"prediction_pit_cutoff_utc", "experience_available_at_utc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"PIT training frame missing required timestamp columns: {missing}"
        )

    cutoff = pd.to_datetime(target_cutoff, utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise RuntimeError("PIT training target cutoff is missing or invalid")

    d = frame.copy()
    prediction_cutoff = pd.to_datetime(
        d["prediction_pit_cutoff_utc"], utc=True, errors="coerce"
    )
    maturity = pd.to_datetime(
        d["experience_available_at_utc"], utc=True, errors="coerce"
    )
    if prediction_cutoff.isna().any() or maturity.isna().any():
        raise RuntimeError(
            "PIT training frame contains missing or invalid prediction/maturity timestamps"
        )

    eligible = prediction_cutoff.lt(cutoff) & maturity.le(cutoff)
    return d.loc[eligible].copy()
