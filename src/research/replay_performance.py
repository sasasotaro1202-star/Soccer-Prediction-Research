from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.engine import _model_features, _oos_temporal_integrity
from src.evaluation.walk_forward import run_walk_forward
from src.evaluation.score_walk_forward import run_score_walk_forward
from src.research.target_oos import write_target_oos_artifacts


REQUIRED = {
    "match_id",
    "kickoff_utc",
    "prediction_cutoff_at_utc",
    "feature_source_max_available_at_utc",
    "pit_verified",
    "home_goals",
    "away_goals",
    "target",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_replay_input(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = Path(path)
    if not p.is_file() or p.stat().st_size <= 0:
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"PIT replay input missing required columns: {missing}")

    out = df.copy()
    for col in ("kickoff_utc", "prediction_cutoff_at_utc", "feature_source_max_available_at_utc"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    if out["match_id"].isna().any() or out["match_id"].astype(str).str.strip().eq("").any():
        raise ValueError("PIT replay input contains missing/empty match_id")
    if out["match_id"].duplicated().any():
        raise ValueError("PIT replay input contains duplicate match_id")
    try:
        verified = out["pit_verified"].astype("boolean")
    except (TypeError, ValueError) as exc:
        raise ValueError("PIT replay input has an ambiguous pit_verified encoding") from exc
    ambiguous_rows = int(verified.isna().sum())
    total_input_rows = int(len(out))
    out = out.loc[verified.fillna(False)].copy()
    out["pit_verified"] = True
    if out["prediction_cutoff_at_utc"].isna().any() or out["kickoff_utc"].isna().any():
        raise ValueError("PIT replay input has missing kickoff/cutoff timestamps")
    if (out["prediction_cutoff_at_utc"] > out["kickoff_utc"]).any():
        raise ValueError("PIT replay input contains prediction cutoff after kickoff")
    if out["feature_source_max_available_at_utc"].isna().any():
        raise ValueError("PIT replay input has missing feature availability timestamps")
    if (out["feature_source_max_available_at_utc"] > out["prediction_cutoff_at_utc"]).any():
        raise ValueError("PIT replay input contains feature availability after prediction cutoff")
    if out[["home_goals", "away_goals", "target"]].isna().any().any():
        raise ValueError("PIT replay input contains missing outcome/target labels")

    if len(out) < 3000:
        raise ValueError(f"Insufficient PIT replay rows for OOS: {len(out)}; need >= 3000")

    report = {
        "status": "PASS",
        "evidence_scope": "PIT_VERIFIED_REPLAY_INPUT",
        "input_rows": total_input_rows,
        "rows": int(len(out)),
        "pit_verified_rows": int(len(out)),
        "pit_verified_rate": float(len(out) / total_input_rows) if total_input_rows else 0.0,
        "pit_ambiguous_rows_excluded": ambiguous_rows,
        "non_verified_rows_excluded": int(total_input_rows - len(out) - ambiguous_rows),
        "unique_match_ids": int(out["match_id"].nunique()),
        "prediction_cutoff_after_kickoff_rows": 0,
        "feature_available_after_cutoff_rows": 0,
        "input_sha256": _sha256(p),
    }
    return out, report


def _weighted(rows: pd.DataFrame, field: str) -> float | None:
    if field not in rows:
        return None
    values = pd.to_numeric(rows[field], errors="coerce")
    n = pd.to_numeric(rows["n"], errors="coerce").fillna(0.0)
    mask = values.notna() & n.gt(0)
    if not mask.any():
        return None
    return float(np.average(values[mask], weights=n[mask]))


def build_snapshot(
    *,
    input_gate: dict[str, Any],
    one_x_two: pd.DataFrame,
    score: pd.DataFrame,
    ou: pd.DataFrame,
    btts: pd.DataFrame,
    source: dict[str, Any],
) -> dict[str, Any]:
    one_integrity = _oos_temporal_integrity(one_x_two, locked_blocks=2)
    score_integrity = _oos_temporal_integrity(score, locked_blocks=2)

    def metrics(df: pd.DataFrame, fields: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for field in fields:
            value = _weighted(df, field)
            if value is not None:
                out[field] = value
        return out

    targets = [
        {
            "task": "1X2",
            "status": "EVALUATED" if one_integrity["status"] == "PASS" else "UNVERIFIED",
            "n": int(pd.to_numeric(one_x_two.get("n", pd.Series(dtype=float)), errors="coerce").sum()),
            "blocks": int(len(one_x_two)),
            "metrics": metrics(one_x_two, ["logloss", "accuracy", "brier", "rps", "ece"]),
            "oos_integrity": one_integrity,
        },
        {
            "task": "Score",
            "status": "EVALUATED" if score_integrity["status"] == "PASS" else "UNVERIFIED",
            "n": int(pd.to_numeric(score.get("n", pd.Series(dtype=float)), errors="coerce").sum()),
            "blocks": int(len(score)),
            "metrics": metrics(
                score,
                [
                    "score_logloss",
                    "exact_score_hit_rate",
                    "top3_score_hit_rate",
                    "top4_score_hit_rate",
                    "home_goals_mae",
                    "away_goals_mae",
                    "total_goals_mae",
                ],
            ),
            "oos_integrity": score_integrity,
        },
        {
            "task": "O/U_2_5",
            "status": "EVALUATED" if score_integrity["status"] == "PASS" and not ou.empty else "UNVERIFIED",
            "n": int(pd.to_numeric(ou.get("n", pd.Series(dtype=float)), errors="coerce").sum()) if not ou.empty else 0,
            "blocks": int(len(ou)),
            "metrics": metrics(ou, ["logloss", "brier"]),
            "evidence_source": "chronological_score_distribution_oos",
        },
        {
            "task": "BTTS",
            "status": "EVALUATED" if score_integrity["status"] == "PASS" and not btts.empty else "UNVERIFIED",
            "n": int(pd.to_numeric(btts.get("n", pd.Series(dtype=float)), errors="coerce").sum()) if not btts.empty else 0,
            "blocks": int(len(btts)),
            "metrics": metrics(btts, ["logloss", "brier"]),
            "evidence_source": "chronological_score_distribution_oos",
        },
        {
            "task": "MOM",
            "status": "UNAVAILABLE",
            "n": 0,
            "blocks": 0,
            "metrics": {},
            "reason": "No target-specific PIT-safe MOM OOS artifact exists in this replay lane.",
        },
    ]

    overall = (
        "E4_OOS_EVALUATED"
        if all(t["status"] in {"EVALUATED", "UNAVAILABLE"} for t in targets[:4])
        else "UNVERIFIED"
    )
    return {
        "schema_version": 1,
        "status": overall,
        "evidence_level": "E4",
        "evidence_scope": "PIT_VERIFIED_REPLAY_WITH_CURRENT_CODE",
        "production_claim": False,
        "adoption_claim": False,
        "pit_input_gate": input_gate,
        "source": source,
        "targets": targets,
        "one_x_two_oos_integrity": one_integrity,
        "score_oos_integrity": score_integrity,
        "notes": [
            "This lane replays an immutable PIT-verified feature artifact from an earlier research run.",
            "It is chronological OOS evidence only; it does not relax the main completion/audit/adoption gates.",
            "MOM is not inferred from unrelated targets.",
        ],
    }


def write_markdown(snapshot: dict[str, Any], path: Path) -> None:
    lines = [
        "# Soccer Replayed OOS Performance",
        "",
        f"- Status: **{snapshot['status']}**",
        f"- Evidence level: **{snapshot['evidence_level']}**",
        f"- Evidence scope: {snapshot['evidence_scope']}",
        f"- Production claim: **{snapshot['production_claim']}**",
        "",
        "| Target | Status | N | Blocks | Metrics |",
        "|---|---|---:|---:|---|",
    ]
    for t in snapshot["targets"]:
        compact = ", ".join(f"{k}={v:.6f}" for k, v in t["metrics"].items()) or "-"
        lines.append(f"| {t['task']} | {t['status']} | {t['n']} | {t['blocks']} | {compact} |")
    lines += [
        "",
        "This report must not be interpreted as Production approval.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="artifacts/replayed_oos")
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    df, input_gate = validate_replay_input(args.input)
    feature_cols = _model_features(df)

    one, selected = run_walk_forward(
        df,
        feature_cols,
        min_train=1000,
        oos_block=2000,
        case_output_path=str(root / "one_x_two_oos_cases.csv"),
    )
    score = run_score_walk_forward(df, min_train=1000, oos_block=2000)
    score.to_csv(root / "score_oos_metrics.csv", index=False)
    write_target_oos_artifacts(score, root)

    one.to_csv(root / "oos_metrics.csv", index=False)
    selected.to_csv(root / "model_selection.csv", index=False)

    source = {
        "input_path": str(Path(args.input)),
        "input_sha256": input_gate["input_sha256"],
        "rows": input_gate["rows"],
    }
    snapshot = build_snapshot(
        input_gate=input_gate,
        one_x_two=one,
        score=score,
        ou=pd.read_csv(root / "ou_oos_metrics.csv"),
        btts=pd.read_csv(root / "btts_oos_metrics.csv"),
        source=source,
    )

    (root / "replay_performance_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_markdown(snapshot, root / "replay_performance_snapshot.md")
    print(json.dumps({
        "status": snapshot["status"],
        "evidence_level": snapshot["evidence_level"],
        "targets": {t["task"]: t["status"] for t in snapshot["targets"]},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
