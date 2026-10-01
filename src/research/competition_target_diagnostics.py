"""Research-only competition x target diagnostics for target-specific OOS cases."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


MIN_SPECIALIST_ROWS = 120
MIN_SPECIALIST_BLOCKS = 3
MIN_BLOCK_ROWS = 30


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def _metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    yy = pd.to_numeric(y, errors="coerce").to_numpy(dtype=int)
    pp = np.clip(pd.to_numeric(p, errors="coerce").to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    return {
        "accuracy": float(((pp >= 0.5).astype(int) == yy).mean()),
        "logloss": float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))),
        "brier": float(np.mean((pp - yy) ** 2)),
        "ece": _ece(yy, pp),
    }


def _validate(cases: pd.DataFrame) -> pd.DataFrame:
    required = {
        "target",
        "block",
        "match_id",
        "competition",
        "kickoff_utc",
        "prediction_probability",
        "baseline_score_probability",
        "actual",
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Target-specific case diagnostics missing columns: {missing}")

    d = cases.copy()
    d["target"] = d["target"].astype("string").str.strip().str.upper()
    d["competition"] = d["competition"].astype("string").str.strip().str.upper()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["block"] = pd.to_numeric(d["block"], errors="coerce")
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["prediction_probability"] = pd.to_numeric(d["prediction_probability"], errors="coerce")
    d["baseline_score_probability"] = pd.to_numeric(d["baseline_score_probability"], errors="coerce")
    d["actual"] = pd.to_numeric(d["actual"], errors="coerce")

    valid = (
        d["target"].isin({"O/U", "BTTS"})
        & d["competition"].notna()
        & d["competition"].ne("")
        & d["match_id"].notna()
        & d["match_id"].ne("")
        & d["block"].notna()
        & d["kickoff_utc"].notna()
        & d["prediction_probability"].between(0.0, 1.0)
        & d["baseline_score_probability"].between(0.0, 1.0)
        & d["actual"].isin({0, 1})
    )
    if not bool(valid.all()):
        raise ValueError("Target-specific case diagnostics contain invalid PIT-OOS rows")

    duplicate = d.duplicated(subset=["target", "match_id"], keep=False)
    if bool(duplicate.any()):
        raise ValueError("Target-specific OOS cases contain duplicate fixture rows within a target")
    return d.sort_values(
        ["target", "kickoff_utc", "competition", "match_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_competition_target_diagnostics(cases: pd.DataFrame) -> dict:
    d = _validate(cases)
    rows: list[dict] = []

    for (target, competition), frame in d.groupby(["target", "competition"], sort=True):
        candidate = _metrics(frame["actual"], frame["prediction_probability"])
        baseline = _metrics(frame["actual"], frame["baseline_score_probability"])
        blocks = sorted(pd.to_numeric(frame["block"], errors="coerce").astype(int).unique().tolist())
        block_sizes = frame.groupby("block", sort=True).size()

        rows.append({
            "target": str(target),
            "competition": str(competition),
            "n": int(len(frame)),
            "blocks": int(len(blocks)),
            "min_block_rows": int(block_sizes.min()) if len(block_sizes) else 0,
            "candidate_accuracy": candidate["accuracy"],
            "baseline_accuracy": baseline["accuracy"],
            "candidate_logloss": candidate["logloss"],
            "baseline_logloss": baseline["logloss"],
            "candidate_brier": candidate["brier"],
            "baseline_brier": baseline["brier"],
            "candidate_ece": candidate["ece"],
            "baseline_ece": baseline["ece"],
            "logloss_delta": float(candidate["logloss"] - baseline["logloss"]),
            "brier_delta": float(candidate["brier"] - baseline["brier"]),
            "accuracy_delta": float(candidate["accuracy"] - baseline["accuracy"]),
            "ece_delta": float(candidate["ece"] - baseline["ece"]),
            "both_classes_present": bool(frame["actual"].nunique() == 2),
            "evidence_ready_for_specialist_research": bool(
                len(frame) >= MIN_SPECIALIST_ROWS
                and len(blocks) >= MIN_SPECIALIST_BLOCKS
                and int(block_sizes.min()) >= MIN_BLOCK_ROWS
                and frame["actual"].nunique() == 2
            ),
        })

    result = {
        "schema_version": 1,
        "status": "READY" if rows else "NO_VALID_CASES",
        "rows": int(len(d)),
        "target_competition_rows": rows,
        "rules": {
            "min_specialist_rows": MIN_SPECIALIST_ROWS,
            "min_specialist_blocks": MIN_SPECIALIST_BLOCKS,
            "min_block_rows": MIN_BLOCK_ROWS,
            "production_usable": False,
            "locked_oos_tuning": False,
        },
        "production_usable": False,
    }
    return result


def write_competition_target_diagnostics(
    case_path: str = "artifacts/target_specific/target_specific_cases.csv",
    out_dir: str = "artifacts/target_specific",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(case_path)
    state = build_competition_target_diagnostics(cases)
    pd.DataFrame(state["target_competition_rows"]).to_csv(
        out / "target_specific_competition_metrics.csv", index=False
    )
    (out / "target_specific_competition_status.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return state


if __name__ == "__main__":
    print(json.dumps(write_competition_target_diagnostics(), indent=2, ensure_ascii=False))
