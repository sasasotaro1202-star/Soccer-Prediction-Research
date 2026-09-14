from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.football_data import load_available_history
from src.features.soccer_features import build_match_features


REQUIRED_INPUT_COLUMNS = {
    "match_id",
    "competition",
    "kickoff_utc",
    "home_team",
    "away_team",
    "source_available_at_utc",
    "starter_status",
}


def prepare_future_fixtures(
    fixtures: pd.DataFrame,
    history: pd.DataFrame,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Convert raw future fixtures into model-ready PIT-safe feature rows.

    The same chronological feature builder used by historical OOS evaluation is
    reused here. No future result is supplied to the builder and no feature is
    imputed from the future fixture itself. Unknown starter/PIT state is retained
    as ineligible rather than guessed.
    """
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

    # Duplicate identities are an integrity error, not something to silently dedupe.
    if f["match_id"].duplicated().any():
        dupes = f.loc[f["match_id"].duplicated(keep=False), "match_id"].astype(str).unique().tolist()[:10]
        raise RuntimeError(f"Future fixture input contains duplicate match_id values: {dupes}")

    feature_rows = build_match_features(history, f)
    if feature_rows["match_id"].duplicated().any():
        raise RuntimeError("Feature builder returned duplicate future fixture identities")

    keep = [
        "match_id", "competition", "season", "season_start", "kickoff_utc",
        "home_team", "away_team", "prediction_cutoff_at_utc", "pit_verified",
        "feature_source_max_available_at_utc",
    ]
    feature_rows = feature_rows[[c for c in keep if c in feature_rows.columns]].copy()
    passthrough = f[[c for c in ["match_id", "source_available_at_utc", "starter_status"] if c in f.columns]].copy()
    out = feature_rows.merge(passthrough, on="match_id", how="left", validate="one_to_one")
    if len(out) != len(f):
        raise RuntimeError("Future fixture preparation changed fixture row count")

    out = out.sort_values(["kickoff_utc", "match_id"], kind="mergesort").reset_index(drop=True)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out


def prepare_from_files(fixtures_path: str, output_path: str) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_path)
    history, _ = load_available_history()
    if history.empty:
        raise RuntimeError("No historical data is available for future PIT feature preparation")
    return prepare_future_fixtures(fixtures, history, output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", default="artifacts/future_fixtures.csv")
    args = parser.parse_args()
    result = prepare_from_files(args.fixtures, args.output)
    print({"status": "PREPARED", "rows": int(len(result)), "output": args.output})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
