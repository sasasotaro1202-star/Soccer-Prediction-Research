"""Chronological calibration of the outcome-free predictability score.

Research-only: each calibration model uses only earlier matured outcomes.
The calibrated probability estimates P(1X2 correct | predictability context)
without changing production prediction probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.predictability_map import TELEMETRY, WEIGHTS

MIN_HISTORY = 120
BLOCK_SIZE = 60
MIN_BLOCKS = 3


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), 1e-9, 1 - 1e-9)
    if len(y) == 0:
        return {"n": 0}
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {
        "n": int(len(y)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
        "accuracy_at_0_5": float(((p >= 0.5).astype(int) == y).mean()),
    }


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "kickoff_utc",
        "prediction_pit_cutoff_utc",
        "experience_available_at_utc",
        "prediction_pit_gate",
        "actual_result",
        "p_home",
        "p_draw",
        "p_away",
    }
    if frame.empty:
        return frame.copy()
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"predictability calibration ledger missing columns: {missing}")
    d = frame.copy()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["prediction_pit_cutoff_utc"] = pd.to_datetime(
        d["prediction_pit_cutoff_utc"], utc=True, errors="coerce"
    )
    d["experience_available_at_utc"] = pd.to_datetime(
        d["experience_available_at_utc"], utc=True, errors="coerce"
    )
    d["actual_result"] = d["actual_result"].astype("string").str.strip().str.upper()
    for c in ("p_home", "p_draw", "p_away"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "prediction_state_id" in d.columns:
        ids = d["prediction_state_id"].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any():
            raise RuntimeError("predictability calibration ledger contains missing prediction_state_id")
        if ids.duplicated().any():
            raise RuntimeError("predictability calibration ledger contains duplicate prediction_state_id")
    elif d["match_id"].duplicated().any():
        raise RuntimeError(
            "predictability calibration ledger contains duplicate match_id without prediction_state_id"
        )
    for name in TELEMETRY:
        source = name if name in d.columns else f"shadow_{name}"
        if source in d.columns:
            d[name] = pd.to_numeric(d[source], errors="coerce")
    valid = d["match_id"].notna() & d["match_id"].ne("")
    valid &= d["prediction_pit_gate"].astype("string").eq("PASS")
    valid &= d["prediction_pit_cutoff_utc"].notna()
    valid &= d["experience_available_at_utc"].notna()
    valid &= d["prediction_pit_cutoff_utc"] < d["kickoff_utc"]
    valid &= d["experience_available_at_utc"] > d["kickoff_utc"]
    valid &= d["experience_available_at_utc"] > d["prediction_pit_cutoff_utc"]
    valid &= d["actual_result"].isin({"H", "D", "A"})
    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    valid &= np.isfinite(p).all(axis=1) & (p >= 0.0).all(axis=1)
    valid &= p.sum(axis=1) > 0.0
    d = d.loc[valid].copy()
    if d.empty:
        return d

    telemetry_weight = np.zeros(len(d), dtype=float)
    telemetry_risk = np.zeros(len(d), dtype=float)
    for name in TELEMETRY:
        if name not in d.columns:
            continue
        value = pd.to_numeric(d[name], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(value)
        weight = float(WEIGHTS[name])
        telemetry_risk += np.where(finite, np.clip(value, 0.0, 1.0) * weight, 0.0)
        telemetry_weight += np.where(finite, weight, 0.0)
    d["predictability_score"] = np.where(
        telemetry_weight > 0.70,
        np.clip(
            1.0 - telemetry_risk / np.maximum(telemetry_weight, 1e-9),
            0.0,
            1.0,
        ),
        np.nan,
    )
    d["predictability_score"] = pd.to_numeric(
        d["predictability_score"], errors="coerce"
    )
    d = d.loc[d["predictability_score"].notna()].copy()
    if d.empty:
        return d
    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    p /= p.sum(axis=1, keepdims=True)
    d[["p_home", "p_draw", "p_away"]] = p
    d["correct"] = (
        p.argmax(axis=1) == d["actual_result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
    ).astype(int)
    return (
        d.sort_values(
            ["match_id", "prediction_pit_cutoff_utc"], kind="mergesort"
        )
        .drop_duplicates("match_id", keep="last")
        .sort_values(["prediction_pit_cutoff_utc", "match_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _calibrator() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=0.5, max_iter=2000, random_state=13013)),
    ])


def calibrate(
    ledger: pd.DataFrame,
    *,
    history_rows: int = MIN_HISTORY,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    d = _prepare(ledger)
    if d.empty:
        return {
            "schema_version": 1,
            "status": "WARMUP",
            "rows": 0,
            "production_usable": False,
            "safety_contract": {
                "research_only": True,
                "production_changed": False,
                "production_probabilities_changed": False,
                "locked_holdout_touched": False,
                "outcome_data_used_only_after_maturity": True,
            },
            "oos_blocks": [],
        }

    if len(d) < int(history_rows) + int(block_size):
        return {
            "schema_version": 1,
            "status": "INSUFFICIENT_OOS",
            "rows": int(len(d)),
            "production_usable": False,
            "reason": "Need chronological prior history plus one OOS block.",
            "safety_contract": {
                "research_only": True,
                "production_changed": False,
                "production_probabilities_changed": False,
                "locked_holdout_touched": False,
                "outcome_data_used_only_after_maturity": True,
            },
            "oos_blocks": [],
        }

    blocks: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    start = int(history_rows)
    block_id = 0
    while start < len(d):
        end = min(start + int(block_size), len(d))
        oos = d.iloc[start:end].copy()
        if len(oos) < int(block_size):
            break
        train = d.iloc[:start].copy()
        y_train = train["correct"].to_numpy(dtype=int)
        if len(train) < int(history_rows) or np.unique(y_train).size < 2:
            start = end
            block_id += 1
            continue

        x_train = train["predictability_score"].to_numpy(dtype=float).reshape(-1, 1)
        x_oos = oos["predictability_score"].to_numpy(dtype=float).reshape(-1, 1)
        model = _calibrator()
        model.fit(x_train, y_train)
        calibrated = np.clip(model.predict_proba(x_oos)[:, 1], 0.01, 0.99)
        y_oos = oos["correct"].to_numpy(dtype=int)
        raw = np.clip(x_oos[:, 0], 0.01, 0.99)

        cal_m = _metrics(y_oos, calibrated)
        raw_m = _metrics(y_oos, raw)
        blocks.append({
            "block": int(block_id),
            "oos_start": oos["prediction_pit_cutoff_utc"].min().isoformat(),
            "oos_end": oos["prediction_pit_cutoff_utc"].max().isoformat(),
            "n": int(len(oos)),
            "training_rows": int(len(train)),
            "calibrated_logloss": cal_m["logloss"],
            "raw_logloss": raw_m["logloss"],
            "delta_logloss": cal_m["logloss"] - raw_m["logloss"],
            "calibrated_brier": cal_m["brier"],
            "raw_brier": raw_m["brier"],
            "delta_brier": cal_m["brier"] - raw_m["brier"],
            "calibrated_ece": cal_m["ece"],
            "raw_ece": raw_m["ece"],
            "delta_ece": cal_m["ece"] - raw_m["ece"],
            "calibrated_accuracy_at_0_5": cal_m["accuracy_at_0_5"],
            "raw_accuracy_at_0_5": raw_m["accuracy_at_0_5"],
        })
        for row, p in zip(oos.itertuples(index=False), calibrated):
            cases.append({
                "block": int(block_id),
                "match_id": str(row.match_id),
                "prediction_time_utc": row.prediction_pit_cutoff_utc.isoformat(),
                "raw_predictability": float(row.predictability_score),
                "calibrated_predictability": float(p),
                "correct": int(row.correct),
            })
        start = end
        block_id += 1

    oos_df = pd.DataFrame(blocks)
    if len(oos_df) < MIN_BLOCKS:
        status = "INSUFFICIENT_OOS"
        gate = {"status": "HOLD", "blocks": int(len(oos_df))}
    else:
        locked = oos_df.tail(2)
        development = oos_df.iloc[:-2]
        development_improvement = bool(
            (development["calibrated_logloss"] < development["raw_logloss"]).any()
            or (development["calibrated_brier"] < development["raw_brier"]).any()
        )
        locked_non_regression = bool(
            (locked["calibrated_logloss"] <= locked["raw_logloss"]).all()
            and (locked["calibrated_brier"] <= locked["raw_brier"]).all()
            and (locked["calibrated_ece"] <= locked["raw_ece"]).all()
        )
        gate = {
            "status": "PROMOTION_CANDIDATE" if development_improvement and locked_non_regression else "HOLD",
            "blocks": int(len(oos_df)),
            "development_blocks": int(len(development)),
            "locked_blocks": int(len(locked)),
            "development_improvement": development_improvement,
            "locked_non_regression": locked_non_regression,
        }
        status = gate["status"]

    return {
        "schema_version": 1,
        "status": status,
        "rows": int(len(d)),
        "oos_blocks": oos_df.to_dict(orient="records"),
        "oos_cases": cases,
        "gate": gate,
        "production_usable": False,
        "safety_contract": {
            "research_only": True,
            "production_changed": False,
            "production_probabilities_changed": False,
            "locked_holdout_touched": False,
            "outcome_data_used_only_after_maturity": True,
            "calibration_training_is_prior_history_only": True,
            "locked_suffix_outcomes_never_train_calibrator": True,
        },
    }


def write(
    ledger_path: str | Path = "data/experience/prediction_ledger.csv",
    out_dir: str | Path = "artifacts/predictability_calibration",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(ledger_path)
    ledger = pd.read_csv(path) if path.is_file() and path.stat().st_size else pd.DataFrame()
    state = calibrate(ledger)
    pd.DataFrame(state.get("oos_blocks", [])).to_csv(out / "predictability_calibration_oos.csv", index=False)
    pd.DataFrame(state.get("oos_cases", [])).to_csv(out / "predictability_calibration_cases.csv", index=False)
    status = {k: v for k, v in state.items() if k != "oos_cases"}
    (out / "predictability_calibration_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return state


if __name__ == "__main__":
    print(json.dumps(write(), indent=2, ensure_ascii=False, default=str))
