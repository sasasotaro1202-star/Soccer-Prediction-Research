"""PIT-safe research layer for case-level predictability and future failure risk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS = {"H": 0, "D": 1, "A": 2}
MIN_TRAIN_ROWS = 120
MIN_OOS_ROWS = 60
MIN_OOS_BLOCKS = 3

TELEMETRY = (
    "predictive_entropy",
    "model_disagreement",
    "covariate_drift",
    "history_support_risk",
    "routing_risk",
)

WEIGHTS = {
    "predictive_entropy": 0.25,
    "model_disagreement": 0.20,
    "covariate_drift": 0.20,
    "history_support_risk": 0.15,
    "routing_risk": 0.20,
}

META_FEATURES = ("confidence", "margin", *TELEMETRY)


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _binary_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), 0.0, 1.0)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), 1e-9, 1 - 1e-9)
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": int(len(y)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": _binary_ece(y, p),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
    }


def _validate_and_reduce(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id", "kickoff_utc", "prediction_pit_cutoff_utc",
        "prediction_pit_gate", "experience_available_at_utc",
        "p_home", "p_draw", "p_away", "actual_result",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"predictability ledger missing required columns: {missing}")

    d = frame.copy()
    for col in ("kickoff_utc", "prediction_pit_cutoff_utc", "experience_available_at_utc"):
        d[col] = pd.to_datetime(d[col], utc=True, errors="coerce")
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["actual_result"] = d["actual_result"].astype("string").str.strip().str.upper()
    for col in ("p_home", "p_draw", "p_away"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in TELEMETRY:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    if "prediction_state_id" in d.columns:
        d["prediction_state_id"] = d["prediction_state_id"].astype("string").str.strip()

    valid = d["prediction_pit_gate"].astype("string").eq("PASS")
    valid &= d["match_id"].notna() & d["match_id"].ne("")
    valid &= d["kickoff_utc"].notna() & d["prediction_pit_cutoff_utc"].notna()
    valid &= d["experience_available_at_utc"].notna()
    valid &= d["prediction_pit_cutoff_utc"] < d["kickoff_utc"]
    valid &= d["experience_available_at_utc"] > d["kickoff_utc"]
    valid &= d["experience_available_at_utc"] > d["prediction_pit_cutoff_utc"]
    valid &= d["actual_result"].isin(LABELS)

    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    valid &= np.isfinite(p).all(axis=1)
    valid &= (p >= 0.0).all(axis=1)
    valid &= np.sum(p, axis=1) > 0.0
    d = d.loc[valid].copy()
    if d.empty:
        return d

    if "prediction_state_id" in d.columns:
        ids = d["prediction_state_id"]
        if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
            raise RuntimeError("predictability ledger contains invalid prediction_state_id")

    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    p /= p.sum(axis=1, keepdims=True)
    d[["p_home", "p_draw", "p_away"]] = p
    d["confidence"] = p.max(axis=1)
    ranked = np.sort(p, axis=1)
    d["margin"] = ranked[:, -1] - ranked[:, -2]
    d["error_label"] = (
        p.argmax(axis=1)
        != d["actual_result"].map(LABELS).to_numpy(dtype=int)
    ).astype(int)

    sort_cols = ["match_id", "prediction_pit_cutoff_utc"]
    if "prediction_recorded_at_utc" in d.columns:
        d["prediction_recorded_at_utc"] = pd.to_datetime(
            d["prediction_recorded_at_utc"], utc=True, errors="coerce"
        )
        sort_cols.append("prediction_recorded_at_utc")

    return (
        d.sort_values(sort_cols, kind="mergesort")
        .drop_duplicates("match_id", keep="last")
        .sort_values(
            ["prediction_pit_cutoff_utc", "kickoff_utc", "match_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _predictability_score(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    if "confidence" not in d.columns or "margin" not in d.columns:
        p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or np.any(p.sum(axis=1) <= 0):
            raise RuntimeError("predictability score requires valid probability rows")
        p /= p.sum(axis=1, keepdims=True)
        if "confidence" not in d.columns:
            d["confidence"] = p.max(axis=1)
        if "margin" not in d.columns:
            ranked = np.sort(p, axis=1)
            d["margin"] = ranked[:, -1] - ranked[:, -2]

    used_weight = np.zeros(len(d), dtype=float)
    risk = np.zeros(len(d), dtype=float)
    for name in TELEMETRY:
        if name not in d.columns:
            continue
        value = pd.to_numeric(d[name], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(value)
        bounded = np.clip(value, 0.0, 1.0)
        weight = float(WEIGHTS[name])
        risk += np.where(finite, bounded * weight, 0.0)
        used_weight += np.where(finite, weight, 0.0)

    d["predictability_telemetry_coverage"] = np.clip(used_weight, 0.0, 1.0)
    complete = used_weight > 0.70
    d["predictability_score"] = np.where(
        complete,
        np.clip(1.0 - risk / np.maximum(used_weight, 1e-9), 0.0, 1.0),
        np.nan,
    )
    d["predictability_band"] = pd.Series(
        np.select(
            [
                d["predictability_score"] >= 0.70,
                d["predictability_score"] >= 0.45,
            ],
            ["HIGH", "MEDIUM"],
            default="LOW",
        ),
        index=d.index,
        dtype="string",
    )
    d.loc[d["predictability_score"].isna(), "predictability_band"] = "UNKNOWN"
    d["high_confidence_low_predictability"] = (
        (d["confidence"] >= 0.75)
        & d["predictability_score"].notna()
        & (d["predictability_score"] < 0.40)
    )
    return d


def _chronological_blocks(frame: pd.DataFrame, min_train: int, block_size: int):
    start = int(min_train)
    block = 0
    while start < len(frame):
        end = min(start + int(block_size), len(frame))
        if end - start < int(block_size) and end - start < MIN_OOS_ROWS:
            break
        yield block, frame.iloc[:start].copy(), frame.iloc[start:end].copy()
        block += 1
        start = end


def _meta_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                C=0.5,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def _oos_evaluate(
    data: pd.DataFrame, block_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for block_id, train, oos in _chronological_blocks(
        data, MIN_TRAIN_ROWS, block_size
    ):
        y_train = train["error_label"].to_numpy(dtype=int)
        y_oos = oos["error_label"].to_numpy(dtype=int)
        if np.unique(y_train).size < 2:
            continue

        model = _meta_model()
        model.fit(train[list(META_FEATURES)], y_train)
        meta_risk = model.predict_proba(oos[list(META_FEATURES)])[:, 1]

        confidence_risk = np.clip(
            1.0 - oos["confidence"].to_numpy(dtype=float),
            1e-9,
            1 - 1e-9,
        )
        heuristic_score = oos["predictability_score"].to_numpy(dtype=float)
        finite = np.isfinite(heuristic_score)
        if finite.any():
            fill = float(np.median(heuristic_score[finite]))
            heuristic_risk = 1.0 - np.where(finite, heuristic_score, fill)
        else:
            heuristic_risk = confidence_risk.copy()
        heuristic_risk = np.clip(heuristic_risk, 1e-9, 1 - 1e-9)

        meta_m = _binary_metrics(y_oos, meta_risk)
        conf_m = _binary_metrics(y_oos, confidence_risk)
        heur_m = _binary_metrics(y_oos, heuristic_risk)
        rows.append({
            "block": int(block_id),
            "oos_start": oos["prediction_pit_cutoff_utc"].min().isoformat(),
            "oos_end": oos["prediction_pit_cutoff_utc"].max().isoformat(),
            "n": int(len(oos)),
            "meta_logloss": meta_m["logloss"],
            "meta_brier": meta_m["brier"],
            "meta_ece": meta_m["ece"],
            "meta_accuracy": meta_m["accuracy"],
            "confidence_logloss": conf_m["logloss"],
            "confidence_brier": conf_m["brier"],
            "confidence_ece": conf_m["ece"],
            "confidence_accuracy": conf_m["accuracy"],
            "heuristic_logloss": heur_m["logloss"],
            "heuristic_brier": heur_m["brier"],
            "heuristic_ece": heur_m["ece"],
            "delta_meta_vs_confidence_logloss": meta_m["logloss"] - conf_m["logloss"],
            "delta_meta_vs_confidence_brier": meta_m["brier"] - conf_m["brier"],
            "delta_meta_vs_heuristic_logloss": meta_m["logloss"] - heur_m["logloss"],
            "delta_meta_vs_heuristic_brier": meta_m["brier"] - heur_m["brier"],
        })
        for row, risk in zip(oos.itertuples(index=False), meta_risk):
            case_rows.append({
                "block": int(block_id),
                "match_id": str(row.match_id),
                "prediction_time_utc": row.prediction_pit_cutoff_utc.isoformat(),
                "actual_result": str(row.actual_result),
                "error_label": int(row.error_label),
                "confidence": float(row.confidence),
                "predictability_score": (
                    float(row.predictability_score)
                    if np.isfinite(row.predictability_score)
                    else np.nan
                ),
                "meta_failure_risk": float(risk),
                "meta_predictability_score": float(1.0 - risk),
            })
    return pd.DataFrame(rows), pd.DataFrame(case_rows)


def _band_map(data: pd.DataFrame) -> pd.DataFrame:
    d = data[data["predictability_score"].notna()].copy()
    columns = [
        "predictability_band", "n", "error_rate", "mean_logloss",
        "mean_confidence", "coverage",
    ]
    if d.empty:
        return pd.DataFrame(columns=columns)

    d["actual_idx"] = d["actual_result"].map(LABELS).astype(int)
    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    d["case_logloss"] = -np.log(
        np.clip(
            p[np.arange(len(d)), d["actual_idx"].to_numpy(dtype=int)],
            1e-9,
            1.0,
        )
    )
    return (
        d.groupby("predictability_band", observed=True, sort=True)
        .agg(
            n=("match_id", "size"),
            error_rate=("error_label", "mean"),
            mean_logloss=("case_logloss", "mean"),
            mean_confidence=("confidence", "mean"),
            coverage=("predictability_telemetry_coverage", "mean"),
        )
        .reset_index()
    )


def _digest(data: pd.DataFrame) -> str:
    cols = [
        c for c in [
            "match_id", "prediction_state_id",
            "prediction_pit_cutoff_utc", "experience_available_at_utc",
            "p_home", "p_draw", "p_away", "actual_result", *TELEMETRY,
        ] if c in data.columns
    ]
    stable = data[cols].copy()
    return hashlib.sha256(
        stable.to_json(orient="records", date_format="iso").encode("utf-8")
    ).hexdigest()


def analyze(
    ledger: pd.DataFrame,
    *,
    block_size: int = MIN_OOS_ROWS,
) -> dict[str, Any]:
    data = _validate_and_reduce(ledger)
    data = _predictability_score(data)
    input_digest = (
        _digest(data) if not data.empty else hashlib.sha256(b"EMPTY").hexdigest()
    )
    bands = _band_map(data)

    if len(data) >= MIN_TRAIN_ROWS + MIN_OOS_ROWS:
        oos, case_oos = _oos_evaluate(data, int(block_size))
    else:
        oos, case_oos = pd.DataFrame(), pd.DataFrame()

    status = "WARMUP"
    reason = (
        "At least 120 training rows and 3 chronological OOS blocks are required "
        "for a failure-risk challenger."
    )
    gate: dict[str, Any] = {"status": "HOLD", "blocks": int(len(oos))}

    if len(oos) >= MIN_OOS_BLOCKS:
        locked = oos.tail(2)
        development = oos.iloc[:-2]
        dev_improvement = bool(
            (development["meta_logloss"] < development["confidence_logloss"]).any()
            or (development["meta_brier"] < development["confidence_brier"]).any()
        )
        locked_non_regression = bool(
            (locked["meta_logloss"] <= locked["confidence_logloss"]).all()
            and (locked["meta_brier"] <= locked["confidence_brier"]).all()
            and (locked["meta_ece"] <= locked["confidence_ece"]).all()
        )
        gate = {
            "status": (
                "PROMOTION_CANDIDATE"
                if dev_improvement and locked_non_regression
                else "HOLD"
            ),
            "blocks": int(len(oos)),
            "development_blocks": int(len(development)),
            "locked_blocks": int(len(locked)),
            "development_improvement": dev_improvement,
            "locked_non_regression": locked_non_regression,
        }
        status = gate["status"]
        reason = (
            "Chronological meta-label OOS evidence available; "
            "production use remains prohibited."
        )
    elif len(data) >= MIN_TRAIN_ROWS + MIN_OOS_ROWS:
        status = "INSUFFICIENT_OOS"
        reason = (
            "Matured data exists but does not yet yield three chronological "
            "OOS blocks."
        )

    latest = data.tail(min(len(data), 250)).copy()
    latest_columns = [
        "match_id", "kickoff_utc", "prediction_pit_cutoff_utc",
        "confidence", "margin", "predictability_score", "predictability_band",
        "predictability_telemetry_coverage", "high_confidence_low_predictability",
        "error_label", *TELEMETRY,
    ]
    latest = latest[[c for c in latest_columns if c in latest.columns]]

    return {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "production_usable": False,
        "input_digest": input_digest,
        "rows": int(len(data)),
        "predictability_scored_rows": (
            int(data["predictability_score"].notna().sum()) if not data.empty else 0
        ),
        "high_confidence_low_predictability_rows": (
            int(data["high_confidence_low_predictability"].sum())
            if not data.empty else 0
        ),
        "band_map": bands.to_dict(orient="records"),
        "oos_blocks": oos.to_dict(orient="records"),
        "gate": gate,
        "safety_contract": {
            "research_only": True,
            "production_changed": False,
            "production_probabilities_changed": False,
            "production_registry_changed": False,
            "frozen_holdout_touched": False,
            "outcome_data_used_only_after_maturity": True,
            "prediction_time_features_only": True,
            "chronological_oos_required_for_candidate": True,
        },
        "meta_features": list(META_FEATURES),
        "telemetry_weights": WEIGHTS,
        "latest_cases": latest.to_dict(orient="records"),
        "oos_case_rows": case_oos.to_dict(orient="records"),
    }


def write(
    ledger_path: str | Path = "data/experience/prediction_ledger.csv",
    out_dir: str | Path = "artifacts/predictability",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = analyze(_read(Path(ledger_path)))
    pd.DataFrame(state["band_map"]).to_csv(out / "predictability_map.csv", index=False)
    pd.DataFrame(state["oos_blocks"]).to_csv(out / "predictability_oos.csv", index=False)
    pd.DataFrame(state["latest_cases"]).to_csv(
        out / "predictability_latest_cases.csv", index=False
    )
    pd.DataFrame(state["oos_case_rows"]).to_csv(
        out / "predictability_oos_cases.csv", index=False
    )
    status = {
        "schema_version": state["schema_version"],
        "status": state["status"],
        "reason": state["reason"],
        "production_usable": False,
        "input_digest": state["input_digest"],
        "rows": state["rows"],
        "predictability_scored_rows": state["predictability_scored_rows"],
        "high_confidence_low_predictability_rows": (
            state["high_confidence_low_predictability_rows"]
        ),
        "gate": state["gate"],
        "safety_contract": state["safety_contract"],
    }
    (out / "predictability_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return state


if __name__ == "__main__":
    print(json.dumps(write(), indent=2, ensure_ascii=False))
