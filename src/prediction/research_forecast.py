from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.competition_sources import TARGET_COMPETITIONS
from src.prediction.prepare_fixtures import prepare_from_files
from src.prediction.runner import run as run_production_prediction
from src.data.matchday_intelligence_fetch import ESPN_LEAGUES


FORECAST_COLUMNS = (
    "match_id",
    "kickoff_utc",
    "competition",
    "home_team",
    "away_team",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "result_prediction",
    "result_prediction_probability",
    "score_1",
    "score_1_probability",
    "score_2",
    "score_2_probability",
    "score_3",
    "score_3_probability",
    "over_2_5_probability",
    "under_2_5_probability",
    "btts_yes_probability",
    "btts_no_probability",
    "mom_status",
    "mom_method",
    "mom_1_player",
    "mom_1_probability",
    "mom_2_player",
    "mom_2_probability",
    "mom_3_player",
    "mom_3_probability",
    "mom_4_player",
    "mom_4_probability",
)


POSITION_PRIOR = {
    "F": 1.00, "FW": 1.00, "ST": 1.00, "CF": 1.00,
    "AM": 0.92, "W": 0.90, "M": 0.82, "MF": 0.82,
    "D": 0.52, "DF": 0.52, "FB": 0.52, "CB": 0.48,
    "GK": 0.22,
}



def _now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _empty_forecast(output_path: str) -> pd.DataFrame:
    result = pd.DataFrame(columns=FORECAST_COLUMNS)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def _adopted_production_ready() -> tuple[bool, str]:
    root = Path("models/current")
    required = (
        root / "production_model.pkl",
        root / "production_model.json",
        root / "model_registry.json",
        root / "production_provenance.json",
    )
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size <= 0]
    if missing:
        return False, "missing_required_adopted_production_artifacts"
    try:
        registry = json.loads((root / "model_registry.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable_production_registry:{type(exc).__name__}"
    if registry.get("adoption_status") != "ADOPT":
        return False, "production_registry_is_not_ADOPT"
    return True, "ADOPTED_PRODUCTION_READY"


def _convert_production_output(
    production: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    if production.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)

    required = {
        "match_id",
        "kickoff_utc",
        "home_team",
        "away_team",
        "prediction",
        "p_home",
        "p_draw",
        "p_away",
        "score_1",
        "score_1_probability",
        "score_2",
        "score_2_probability",
        "score_3",
        "score_3_probability",
    }
    missing = sorted(required - set(production.columns))
    if missing:
        raise RuntimeError(f"Production prediction output missing fields: {missing}")

    fixture_cols = ["match_id", "competition"]
    fixture_map = fixtures[fixture_cols].drop_duplicates("match_id")
    d = production.merge(fixture_map, on="match_id", how="left", validate="one_to_one")
    if d["competition"].isna().any():
        raise RuntimeError("Production prediction output could not be mapped to fixture competition")

    labels = []
    for row in d.itertuples(index=False):
        pred = str(row.prediction)
        if pred == "H":
            labels.append(str(row.home_team))
        elif pred == "A":
            labels.append(str(row.away_team))
        elif pred == "D":
            labels.append("引き分け")
        else:
            raise RuntimeError(f"Unknown production prediction label: {pred}")

    result = pd.DataFrame({
        "match_id": d["match_id"].astype(str),
        "kickoff_utc": pd.to_datetime(d["kickoff_utc"], utc=True).astype(str),
        "competition": d["competition"].astype(str),
        "home_team": d["home_team"].astype(str),
        "away_team": d["away_team"].astype(str),
        "home_win_probability": pd.to_numeric(d["p_home"], errors="raise"),
        "draw_probability": pd.to_numeric(d["p_draw"], errors="raise"),
        "away_win_probability": pd.to_numeric(d["p_away"], errors="raise"),
        "result_prediction": labels,
        "result_prediction_probability": pd.to_numeric(
            d[["p_home", "p_draw", "p_away"]].max(axis=1), errors="raise"
        ),
        "score_1": d["score_1"].astype(str),
        "score_1_probability": pd.to_numeric(d["score_1_probability"], errors="raise"),
        "score_2": d["score_2"].astype(str),
        "score_2_probability": pd.to_numeric(d["score_2_probability"], errors="raise"),
        "score_3": d["score_3"].astype(str),
        "score_3_probability": pd.to_numeric(d["score_3_probability"], errors="raise"),
        "over_2_5_probability": pd.to_numeric(d["market_over_2_5"], errors="raise"),
        "under_2_5_probability": pd.to_numeric(d["market_under_2_5"], errors="raise"),
        "btts_yes_probability": pd.to_numeric(d["market_btts_yes"], errors="raise"),
        "btts_no_probability": pd.to_numeric(d["market_btts_no"], errors="raise"),
        "mom_status": "BLOCKED_UPSTREAM_PLAYER_MODEL",
        "mom_method": "production_runner_player_model_not_available_in_daily_forecast_contract",
        "mom_1_player": "",
        "mom_1_probability": np.nan,
        "mom_2_player": "",
        "mom_2_probability": np.nan,
        "mom_3_player": "",
        "mom_3_probability": np.nan,
        "mom_4_player": "",
        "mom_4_probability": np.nan,
    })
    if result["match_id"].duplicated().any():
        raise RuntimeError("Converted production forecast contains duplicate match_id values")
    return result[FORECAST_COLUMNS]


def run(
    fixtures_path: str,
    output_path: str,
    status_path: str,
    prediction_time: str | None = None,
) -> dict[str, Any]:
    now = pd.Timestamp(prediction_time) if prediction_time else _now_utc()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    fixtures = pd.read_csv(fixtures_path)
    required = {"match_id", "kickoff_utc", "home_team", "away_team", "competition"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise RuntimeError(f"fixture snapshot missing columns: {missing}")

    fixtures["kickoff_utc"] = pd.to_datetime(fixtures["kickoff_utc"], utc=True, errors="coerce")
    fixtures["competition"] = fixtures["competition"].astype(str).str.strip().str.upper()
    target = fixtures[
        fixtures["competition"].isin(TARGET_COMPETITIONS)
        & fixtures["kickoff_utc"].notna()
        & (fixtures["kickoff_utc"] > now)
    ].copy()
    target = target.sort_values(["kickoff_utc", "match_id"], kind="mergesort").reset_index(drop=True)

    if target.empty:
        _empty_forecast(output_path)
        status = {
            "status": "NO_TARGET_FIXTURES",
            "prediction_time_utc": now.isoformat(),
            "rows": 0,
            "production_model_used": False,
            "reason": "No target fixture exists after prediction_time.",
        }
        Path(status_path).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    ready, reason = _adopted_production_ready()
    if not ready:
        _empty_forecast(output_path)
        status = {
            "status": "DEFERRED_NO_ADOPTED_MODEL",
            "prediction_time_utc": now.isoformat(),
            "rows": int(len(target)),
            "prediction_rows": 0,
            "production_model_used": False,
            "research_heuristic_disabled": True,
            "reason": reason,
        }
        Path(status_path).parent.mkdir(parents=True, exist_ok=True)
        Path(status_path).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    base_dir = Path(output_path).parent
    prepared_path = base_dir / ".daily_research_prepared.csv"
    production_output = base_dir / ".daily_research_production_predictions.csv"
    production_status = base_dir / ".daily_research_production_status.json"
    target_snapshot = base_dir / ".daily_research_target.csv"
    target.to_csv(target_snapshot, index=False)

    prepare_from_files(str(target_snapshot), str(prepared_path))
    runner_status = run_production_prediction(
        fixtures_path=str(prepared_path),
        bundle_path="artifacts/production_model.pkl",
        output_path=str(production_output),
        status_path=str(production_status),
        prediction_time=now.isoformat(),
        registry_path="artifacts/model_registry.json",
        model_policy="production",
    )

    if not production_output.is_file() or production_output.stat().st_size <= 0:
        _empty_forecast(output_path)
        status = {
            "status": "DEFERRED_NO_PIT_ELIGIBLE_FIXTURES",
            "prediction_time_utc": now.isoformat(),
            "rows": int(len(target)),
            "prediction_rows": 0,
            "production_model_used": True,
            "runner_status": runner_status,
        }
        Path(status_path).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        for path in (prepared_path, production_output, production_status, target_snapshot):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return status

    production = pd.read_csv(production_output)
    result = _convert_production_output(production, target)
    result.to_csv(output_path, index=False)
    status = {
        "status": "PREDICTED_PRODUCTION_ADOPTED" if not result.empty else "DEFERRED_NO_PIT_ELIGIBLE_FIXTURES",
        "prediction_time_utc": now.isoformat(),
        "rows": int(len(target)),
        "prediction_rows": int(len(result)),
        "production_model_used": True,
        "research_heuristic_disabled": True,
        "runner_status": runner_status,
    }
    Path(status_path).parent.mkdir(parents=True, exist_ok=True)
    Path(status_path).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    for path in (prepared_path, production_output, production_status, target_snapshot):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return status


def verify(path: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    required = set(FORECAST_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"forecast output missing required columns: {missing}")
    errors: list[str] = []
    if df.empty:
        return {"status": "VERIFIED", "rows": 0, "empty_target_set": True}
    for idx, row in df.iterrows():
        probs = np.asarray([
            float(row["home_win_probability"]),
            float(row["draw_probability"]),
            float(row["away_win_probability"]),
        ])
        if not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1.0, atol=1e-6):
            errors.append(f"{idx}: invalid 1X2 probabilities")
        for rank in (1, 2, 3):
            score = str(row.get(f"score_{rank}", ""))
            p = float(row.get(f"score_{rank}_probability", np.nan))
            if "-" not in score or not np.isfinite(p) or not 0.0 <= p <= 1.0:
                errors.append(f"{idx}: invalid score top{rank}")
        ou = np.asarray([
            float(row.get("over_2_5_probability", np.nan)),
            float(row.get("under_2_5_probability", np.nan)),
        ])
        if not np.isfinite(ou).all() or np.any((ou < 0) | (ou > 1)) or not np.isclose(ou.sum(), 1.0, atol=1e-6):
            errors.append(f"{idx}: invalid O/U 2.5 probabilities")
        btts = np.asarray([
            float(row.get("btts_yes_probability", np.nan)),
            float(row.get("btts_no_probability", np.nan)),
        ])
        if not np.isfinite(btts).all() or np.any((btts < 0) | (btts > 1)) or not np.isclose(btts.sum(), 1.0, atol=1e-6):
            errors.append(f"{idx}: invalid BTTS probabilities")
        if str(row.get("mom_status", "")).startswith("PREDICTED"):
            mom_names = [str(row.get(f"mom_{r}_player", "")).strip() for r in (1,2,3,4)]
            mom_probs = np.asarray([float(row[f"mom_{r}_probability"]) for r in (1,2,3,4)])
            if any(not name for name in mom_names):
                errors.append(f"{idx}: incomplete MOM top4")
            if len(set(mom_names)) != 4:
                errors.append(f"{idx}: duplicate MOM candidates")
            if not np.isfinite(mom_probs).all() or np.any(mom_probs < 0) or not np.isclose(mom_probs.sum(), 1.0, atol=1e-6):
                errors.append(f"{idx}: invalid MOM probabilities")
    if errors:
        raise RuntimeError("; ".join(errors[:20]))
    return {"status": "VERIFIED", "rows": int(len(df))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", default="artifacts/daily_research_predictions.csv")
    parser.add_argument("--status", default="artifacts/daily_research_prediction_status.json")
    parser.add_argument("--prediction-time", default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    status = run(args.fixtures, args.output, args.status, args.prediction_time)
    if args.verify:
        status.update(verify(args.output))
        Path(args.status).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
