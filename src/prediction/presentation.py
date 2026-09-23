from __future__ import annotations

from typing import Any

MARKETS = (
    "over_0_5", "under_0_5",
    "over_1_5", "under_1_5",
    "over_2_5", "under_2_5",
    "over_3_5", "under_3_5",
    "over_4_5", "under_4_5",
    "btts_yes", "btts_no",
)


def _pct(value: Any) -> str:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"Probability outside [0,1]: {number}")
    return f"{number * 100:.1f}%"


def format_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    """Format one production prediction row with a fixed human-readable contract."""
    required = {"p_home", "p_draw", "p_away"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Prediction row missing required 1X2 fields: {missing}")
    scores = []
    for rank in range(1, 4):
        score = row.get(f"score_{rank}")
        probability = row.get(f"score_{rank}_probability")
        if score is None or probability is None:
            raise ValueError(f"Prediction row missing score TOP3 rank {rank}")
        scores.append({"rank": rank, "score": str(score), "probability": _pct(probability)})

    moms = []
    for rank in range(1, 5):
        player = row.get(f"mom_{rank}_player_id")
        probability = row.get(f"mom_{rank}_probability")
        if player is None or probability is None:
            raise ValueError(f"Prediction row missing MOM TOP4 rank {rank}")
        moms.append({"rank": rank, "player_id": str(player), "probability": _pct(probability)})

    markets = {}
    for key in MARKETS:
        value = row.get(f"market_{key}")
        if value is None:
            raise ValueError(f"Prediction row missing market probability: {key}")
        markets[key] = _pct(value)

    return {
        "match_id": str(row.get("match_id", "")),
        "kickoff_utc": str(row.get("kickoff_utc", "")),
        "home_team": str(row.get("home_team", "")),
        "away_team": str(row.get("away_team", "")),
        "model_version": str(row.get("model_version", "")),
        "1x2": {
            "home": _pct(row["p_home"]),
            "draw": _pct(row["p_draw"]),
            "away": _pct(row["p_away"]),
        },
        "score_top3": scores,
        "markets": markets,
        "mom_top4": moms,
    }
