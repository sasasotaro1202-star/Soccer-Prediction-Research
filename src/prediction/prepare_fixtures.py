from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.football_data import load_available_history
from src.features.soccer_features import build_match_features


REQUIRED_INPUT_COLUMNS = {
    "match_id", "competition", "kickoff_utc", "home_team", "away_team",
    "source_available_at_utc", "starter_status",
}


def prepare_future_fixtures(
    fixtures: pd.DataFrame,
    history: pd.DataFrame,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Convert raw future fixtures into model-ready PIT-safe feature rows."""
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(fixtures.columns))
    if missing:
        raise RuntimeError(f"Future fixture input missing required columns: {missing}")
    f = fixtures.copy()
    f["match_id"] = f["match_id"].astype("string").str.strip()
    if f["match_id"].isna().any() or f["match_id"].eq("").any():
        raise RuntimeError("Future fixture input contains missing/empty match_id values")
    f["kickoff_utc"] = pd.to_datetime(f["kickoff_utc"], utc=True, errors="coerce")
    f["source_available_at_utc"] = pd.to_datetime(f["source_available_at_utc"], utc=True, errors="coerce")
    if f["kickoff_utc"].isna().any():
        raise RuntimeError("Future fixture input contains invalid kickoff_utc values")
    f["starter_status"] = f["starter_status"].astype("string").str.upper().str.strip()
    if f["match_id"].duplicated().any():
        dupes = f.loc[f["match_id"].duplicated(keep=False), "match_id"].astype(str).unique().tolist()[:10]
        raise RuntimeError(f"Future fixture input contains duplicate match_id values: {dupes}")

    # Empty future snapshots are a valid fail-closed state (for example when
    # every external source is temporarily unavailable). Do not pass an empty
    # frame into the feature builder, which may legitimately return no columns.
    if f.empty:
        out = f.copy()
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
        return out

    feature_rows = build_match_features(history, f)
    if feature_rows["match_id"].duplicated().any():
        raise RuntimeError("Feature builder returned duplicate future fixture identities")
    matchday_columns = [c for c in f.columns if c.startswith("matchday_")]
    passthrough_columns = ["match_id", "source_available_at_utc", "starter_status", *matchday_columns]
    passthrough = f[passthrough_columns].copy()
    # Matchday intelligence is an evidence layer, not part of historical feature fitting.
    # Preserve its raw PIT metadata/signals verbatim so the production runner can validate
    # and apply them later without guessing or silently dropping the evidence.
    overlap = [c for c in matchday_columns if c in feature_rows.columns]
    if overlap:
        feature_rows = feature_rows.drop(columns=overlap)
    out = feature_rows.merge(passthrough, on="match_id", how="left", validate="one_to_one")
    if len(out) != len(f):
        raise RuntimeError("Future fixture preparation changed fixture row count")
    out = out.sort_values(["kickoff_utc", "match_id"], kind="mergesort").reset_index(drop=True)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out


def prepare_from_files(fixtures_path: str, output_path: str, history_path: str | None = None) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_path)
    # Empty upstream snapshots are a valid fail-closed state. Return immediately
    # instead of loading the full historical dataset just to produce zero rows.
    if fixtures.empty:
        return prepare_future_fixtures(fixtures, pd.DataFrame(), output_path)
    if history_path:
        history = pd.read_csv(history_path)
        for c in ("kickoff_utc", "source_available_at_utc"):
            if c in history.columns:
                history[c] = pd.to_datetime(history[c], utc=True, errors="coerce")
    else:
        history, _ = load_available_history()
    if history.empty:
        raise RuntimeError("No historical data is available for future PIT feature preparation")
    return prepare_future_fixtures(fixtures, history, output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--history", default=None)
    parser.add_argument("--output", default="artifacts/future_fixtures.csv")
    args = parser.parse_args()
    result = prepare_from_files(args.fixtures, args.output, args.history)
    print({"status": "PREPARED", "rows": int(len(result)), "output": args.output})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
