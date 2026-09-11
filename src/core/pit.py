from __future__ import annotations

import pandas as pd

REQUIRED_TS = [
    "event_time_utc",
    "source_available_at_utc",
    "retrieved_at_utc",
    "prediction_cutoff_at_utc",
]


def utc_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def leakage_gate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_TS:
        if c in out:
            out[c] = utc_series(out[c])
    missing = [c for c in REQUIRED_TS if c not in out]
    if missing:
        out["leakage_gate_status"] = "FAIL"
        out["leakage_gate_reason"] = "missing_timestamp:" + ",".join(missing)
        return out
    ok = (
        out["source_available_at_utc"].notna()
        & out["prediction_cutoff_at_utc"].notna()
        & out["event_time_utc"].notna()
        & (out["source_available_at_utc"] <= out["prediction_cutoff_at_utc"])
        & (out["event_time_utc"] < out["prediction_cutoff_at_utc"])
    )
    out["leakage_gate_status"] = ok.map({True: "PASS", False: "FAIL"})
    out["leakage_gate_reason"] = ""
    out.loc[~ok, "leakage_gate_reason"] = "PIT timestamp rule failed"
    return out


def assert_no_leakage(df: pd.DataFrame) -> None:
    checked = leakage_gate(df)
    bad = checked[checked["leakage_gate_status"] != "PASS"]
    if not bad.empty:
        raise ValueError(f"PIT/leakage gate failed for {len(bad)} rows")
