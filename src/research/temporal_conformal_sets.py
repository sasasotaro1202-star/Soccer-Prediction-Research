"""Research-only temporal conformal prediction sets for soccer 1X2.

Calibration at prediction time t uses only prior prediction states whose realized
outcomes were confirmed at or before t. The module never changes production
probabilities; it evaluates a selective set/abstention policy separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS = ("H", "D", "A")
MIN_CALIBRATION = 60
DEFAULT_MAX_CALIBRATION = 240


def _probabilities(frame: pd.DataFrame) -> np.ndarray:
    required = {"p_home", "p_draw", "p_away"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"conformal input missing probability columns: {missing}")
    p = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
        raise RuntimeError("conformal probabilities are invalid")
    p /= p.sum(axis=1, keepdims=True)
    return p


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
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
        raise RuntimeError(f"conformal ledger missing required columns: {missing}")
    d = frame.copy()
    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["kickoff_utc"] = pd.to_datetime(
        d["kickoff_utc"], utc=True, errors="coerce"
    )
    d["prediction_pit_cutoff_utc"] = pd.to_datetime(
        d["prediction_pit_cutoff_utc"], utc=True, errors="coerce"
    )
    d["experience_available_at_utc"] = pd.to_datetime(
        d["experience_available_at_utc"], utc=True, errors="coerce"
    )
    d["actual_result"] = d["actual_result"].astype("string").str.strip().str.upper()
    p = _probabilities(d)
    valid = d["match_id"].notna() & d["match_id"].ne("")
    valid &= d["prediction_pit_gate"].astype("string").eq("PASS")
    valid &= d["prediction_pit_cutoff_utc"].notna()
    valid &= d["experience_available_at_utc"].notna()
    valid &= d["prediction_pit_cutoff_utc"] < d["kickoff_utc"]
    valid &= d["experience_available_at_utc"] > d["kickoff_utc"]
    valid &= d["experience_available_at_utc"] > d["prediction_pit_cutoff_utc"]
    valid &= d["actual_result"].isin(LABELS)
    valid &= np.isfinite(p).all(axis=1)
    pit_rows = d["prediction_pit_gate"].astype("string").eq("PASS")
    if bool((pit_rows & ~valid).any()):
        raise RuntimeError("conformal maturity/PIT validation failed for a PASS ledger row")
    d = d.loc[valid].copy()
    if d.empty:
        return d
    if "prediction_state_id" in d.columns:
        ids = d["prediction_state_id"].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
            raise RuntimeError("conformal ledger contains invalid prediction_state_id")
    elif d["match_id"].duplicated().any():
        raise RuntimeError(
            "conformal ledger contains duplicate match_id without prediction_state_id"
        )
    d[["p_home", "p_draw", "p_away"]] = p[valid.to_numpy()]
    # Keep one state per realized fixture for outcome evaluation. The latest
    # state has the most recent PIT cutoff and is deterministic.
    d = (
        d.sort_values(
            ["match_id", "prediction_pit_cutoff_utc", "experience_available_at_utc"],
            kind="mergesort",
        )
        .drop_duplicates("match_id", keep="last")
        .sort_values(
            ["prediction_pit_cutoff_utc", "match_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return d


def _pvalues(calibration_nonconformity: np.ndarray, row_probability: np.ndarray) -> np.ndarray:
    sorted_scores = np.sort(np.clip(calibration_nonconformity, 0.0, 1.0))
    n = len(sorted_scores)
    thresholds = 1.0 - np.asarray(row_probability, dtype=float)
    idx = np.searchsorted(sorted_scores, thresholds, side="left")
    return (1.0 + n - idx) / (n + 1.0)


def temporal_prediction_sets(
    ledger: pd.DataFrame,
    *,
    alpha: float = 0.10,
    min_calibration: int = MIN_CALIBRATION,
    max_calibration: int | None = DEFAULT_MAX_CALIBRATION,
) -> dict[str, Any]:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if int(min_calibration) < 1:
        raise ValueError("min_calibration must be >= 1")
    if max_calibration is not None and int(max_calibration) < int(min_calibration):
        raise ValueError("max_calibration must be >= min_calibration")

    d = _validate(ledger)
    if d.empty:
        return {
            "schema_version": 1,
            "status": "WARMUP",
            "rows": 0,
            "eligible_rows": 0,
            "production_usable": False,
            "prediction_set_rows": [],
            "safety_contract": {
                "research_only": True,
                "production_changed": False,
                "production_probabilities_changed": False,
                "outcome_data_used_only_after_maturity": True,
                "locked_holdout_touched": False,
                "same_prediction_time_excluded": True,
            },
        }

    p = d[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    y = d["actual_result"].map(dict(zip(LABELS, range(3)))).to_numpy(dtype=int)
    pt = d["prediction_pit_cutoff_utc"].reset_index(drop=True)
    mature = d["experience_available_at_utc"].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    calibration_counts: list[int] = []
    for i in range(len(d)):
        prior = np.arange(i)
        eligible = prior[
            mature.iloc[:i].le(pt.iloc[i]).to_numpy()
            & pt.iloc[:i].lt(pt.iloc[i]).to_numpy()
        ]
        if max_calibration is not None and len(eligible) > int(max_calibration):
            eligible = eligible[-int(max_calibration):]
        n = int(len(eligible))
        calibration_counts.append(n)

        if n < int(min_calibration):
            rows.append({
                "match_id": str(d.iloc[i]["match_id"]),
                "prediction_time_utc": pt.iloc[i].isoformat(),
                "actual_result": str(d.iloc[i]["actual_result"]),
                "calibration_rows": n,
                "set": "",
                "set_size": 0,
                "action": "ABSTAIN",
                "coverage": np.nan,
                "prediction_set_contains_actual": np.nan,
            })
            continue

        nonconformity = 1.0 - p[eligible, y[eligible]]
        pvalues = _pvalues(nonconformity, p[i])
        included = pvalues > alpha
        label_set = [LABELS[j] for j in np.flatnonzero(included)]
        size = int(included.sum())
        contains = bool(y[i] in np.flatnonzero(included))
        action = "ABSTAIN" if size == 0 else ("SINGLE" if size == 1 else "SET")
        rows.append({
            "match_id": str(d.iloc[i]["match_id"]),
            "prediction_time_utc": pt.iloc[i].isoformat(),
            "actual_result": str(d.iloc[i]["actual_result"]),
            "calibration_rows": n,
            "set": "|".join(label_set),
            "set_size": size,
            "action": action,
            "coverage": float(contains),
            "prediction_set_contains_actual": float(contains),
        })

    result = pd.DataFrame(rows)
    eligible_mask = result["calibration_rows"].astype(int) >= int(min_calibration)
    eligible = result.loc[eligible_mask].copy()
    if eligible.empty:
        status = "INSUFFICIENT_CALIBRATION"
        metrics = {
            "total_rows": int(len(result)),
            "eligible_rows": 0,
            "coverage": np.nan,
            "mean_set_size": np.nan,
            "singleton_rate": np.nan,
            "singleton_accuracy": np.nan,
            "abstain_rate": np.nan,
        }
    else:
        single = eligible["set_size"].eq(1)
        single_rows = eligible.loc[single]
        metrics = {
            "total_rows": int(len(result)),
            "eligible_rows": int(len(eligible)),
            "coverage": float(eligible["prediction_set_contains_actual"].mean()),
            "mean_set_size": float(eligible["set_size"].mean()),
            "singleton_rate": float(single.mean()),
            "singleton_accuracy": (
                float(
                    sum(
                        str(row.set).split("|")[0] == str(row.actual_result)
                        for row in single_rows.itertuples(index=False)
                    ) / len(single_rows)
                )
                if not single_rows.empty else np.nan
            ),
            "abstain_rate": float(eligible["set_size"].eq(0).mean()),
        }
        status = "READY"

    digest_cols = [
        c for c in (
            "match_id",
            "prediction_state_id",
            "prediction_pit_cutoff_utc",
            "kickoff_utc",
            "experience_available_at_utc",
            "p_home",
            "p_draw",
            "p_away",
            "actual_result",
        )
        if c in d.columns
    ]
    digest = hashlib.sha256(
        d[digest_cols].to_json(orient="records", date_format="iso").encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": status,
        "rows": int(len(d)),
        "eligible_rows": int(metrics["eligible_rows"]),
        "metrics": metrics,
        "config": {
            "alpha": alpha,
            "min_calibration": int(min_calibration),
            "max_calibration": int(max_calibration) if max_calibration is not None else None,
        },
        "input_digest": digest,
        "production_usable": False,
        "prediction_set_rows": rows,
        "safety_contract": {
            "research_only": True,
            "production_changed": False,
            "production_probabilities_changed": False,
            "outcome_data_used_only_after_maturity": True,
            "locked_holdout_touched": False,
            "same_prediction_time_excluded": True,
            "calibration_rule": "prior prediction cutoff < target cutoff AND outcome maturity <= target cutoff",
        },
    }


def write(
    ledger_path: str | Path = "data/experience/prediction_ledger.csv",
    out_dir: str | Path = "artifacts/conformal",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ledger_path) if Path(ledger_path).is_file() and Path(ledger_path).stat().st_size else pd.DataFrame()
    state = temporal_prediction_sets(ledger)
    rows = pd.DataFrame(state.get("prediction_set_rows", []))
    rows.to_csv(out / "prediction_sets.csv", index=False)
    status = {k: v for k, v in state.items() if k != "prediction_set_rows"}
    (out / "conformal_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return state


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=2))
