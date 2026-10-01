"""PIT-safe target-specific learning from matured prediction experience."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


TARGETS = {
    "O/U": ("over_2_5", lambda h, a: int(h + a >= 3)),
    "BTTS": ("btts_yes", lambda h, a: int(h >= 1 and a >= 1)),
}
MIN_TEACHERS = 60
MIN_BLOCKS = 3
MIN_ROWS_PER_BLOCK = 30


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() and path.stat().st_size else pd.DataFrame()


def _validate(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id", "kickoff_utc", "prediction_pit_cutoff_utc",
        "prediction_pit_gate", "experience_available_at_utc",
        "actual_home_goals", "actual_away_goals",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise RuntimeError(f"target experience ledger missing columns: {missing}")
    d = ledger.copy()
    for c in ("kickoff_utc", "prediction_pit_cutoff_utc", "experience_available_at_utc"):
        d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
    d["actual_home_goals"] = pd.to_numeric(d["actual_home_goals"], errors="coerce")
    d["actual_away_goals"] = pd.to_numeric(d["actual_away_goals"], errors="coerce")
    d = d[d["prediction_pit_gate"].astype("string").eq("PASS")].copy()
    d = d.dropna(subset=[
        "kickoff_utc", "prediction_pit_cutoff_utc",
        "experience_available_at_utc", "actual_home_goals", "actual_away_goals",
    ])
    d = d[
        (d["prediction_pit_cutoff_utc"] < d["kickoff_utc"])
        & (d["experience_available_at_utc"] > d["kickoff_utc"])
        & (d["experience_available_at_utc"] > d["prediction_pit_cutoff_utc"])
    ]
    if d["match_id"].isna().any() or d["match_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError("target experience ledger contains invalid match_id")
    return d.sort_values(
        ["prediction_pit_cutoff_utc", "kickoff_utc", "match_id"], kind="mergesort"
    ).reset_index(drop=True)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _binary_ece(y, p, bins=10):
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

def _score(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return {
        "n": int(len(y)),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "logloss": float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
        "brier": float(np.mean((p-y) ** 2)),
        "ece": _binary_ece(y, p),
    }


def _blocks(frame: pd.DataFrame):
    if len(frame) < MIN_BLOCKS * MIN_ROWS_PER_BLOCK:
        return []
    k = min(5, len(frame) // MIN_ROWS_PER_BLOCK)
    edges = np.linspace(0, len(frame), k + 1, dtype=int)
    return [
        frame.iloc[edges[i]:edges[i + 1]].copy()
        for i in range(k)
        if edges[i] < edges[i + 1]
    ]


def learn_target_specific_experience(ledger: pd.DataFrame) -> dict:
    data = _validate(ledger)
    result = {
        "schema_version": 1,
        "status": "INSUFFICIENT_EXPERIENCE",
        "production_usable": False,
        "targets": {},
        "mom": {
            "status": "UPSTREAM_PLAYER_MODEL_REQUIRED",
            "rows": 0,
            "production_usable": False,
        },
        "oos_rows": [],
    }
    if data.empty:
        return result

    for target, (prob_col, label_fn) in TARGETS.items():
        if prob_col not in data.columns:
            continue
        td = data.copy()
        td["p"] = pd.to_numeric(td[prob_col], errors="coerce")
        td = td[td["p"].notna()].copy()
        td["y"] = [
            label_fn(int(h), int(a))
            for h, a in td[["actual_home_goals", "actual_away_goals"]].to_numpy()
        ]
        replay = []
        for idx, row in td.iterrows():
            cutoff = row["prediction_pit_cutoff_utc"]
            teachers = td.loc[
                (td.index < idx)
                & (td["experience_available_at_utc"] <= cutoff)
                & (td["prediction_pit_cutoff_utc"] < cutoff)
            ]
            p = float(row["p"])
            adjusted = p
            used = False
            teacher_rows = int(len(teachers))
            if teacher_rows >= MIN_TEACHERS and teachers["y"].nunique() >= 2:
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                model.fit(_logit(teachers["p"]).reshape(-1, 1), teachers["y"].astype(int))
                adjusted = float(model.predict_proba([[_logit([p])[0]]])[0, 1])
                used = True
            replay.append({
                "target": target,
                "match_id": str(row["match_id"]),
                "prediction_cutoff_utc": row["prediction_pit_cutoff_utc"].isoformat(),
                "actual": int(row["y"]),
                "baseline_probability": p,
                "experience_probability": adjusted,
                "teacher_rows": teacher_rows,
                "experience_used": used,
            })

        replay_df = pd.DataFrame(replay)
        blocks = _blocks(replay_df)
        if not blocks:
            result["targets"][target] = {
                "status": "INSUFFICIENT_EXPERIENCE",
                "rows": int(len(replay_df)),
                "production_usable": False,
            }
            continue

        block_rows = []
        for block_id, frame in enumerate(blocks):
            base = _score(frame["actual"], frame["baseline_probability"])
            exp = _score(frame["actual"], frame["experience_probability"])
            block_rows.append({
                "target": target,
                "block": int(block_id),
                "n": int(len(frame)),
                "baseline_accuracy": base["accuracy"],
                "experience_accuracy": exp["accuracy"],
                "baseline_logloss": base["logloss"],
                "experience_logloss": exp["logloss"],
                "baseline_brier": base["brier"],
                "experience_brier": exp["brier"],
                "baseline_ece": base["ece"],
                "experience_ece": exp["ece"],
            })
        b = pd.DataFrame(block_rows)
        locked = b.tail(2)
        development = b.iloc[:-2]
        development_improvement = bool(
            (development["experience_logloss"] < development["baseline_logloss"]).any()
            or (development["experience_brier"] < development["baseline_brier"]).any()
        )
        locked_non_regression = bool(
            (locked["experience_logloss"] <= locked["baseline_logloss"]).all()
            and (locked["experience_brier"] <= locked["baseline_brier"]).all()
            and (locked["experience_ece"] <= locked["baseline_ece"]).all()
            and (locked["experience_accuracy"] >= locked["baseline_accuracy"]).all()
        )
        candidate = bool(development_improvement and locked_non_regression)
        result["targets"][target] = {
            "status": "PROMOTION_CANDIDATE" if candidate else "HOLD",
            "rows": int(len(replay_df)),
            "blocks": int(len(b)),
            "production_usable": False,
            "development_improvement": development_improvement,
            "locked_non_regression": locked_non_regression,
        }
        result["oos_rows"].extend(block_rows)

    if {"mom_top1_hit", "mom_top4_hit"}.issubset(data.columns):
        mom_rows = data[["mom_top1_hit", "mom_top4_hit"]].notna().all(axis=1)
        result["mom"] = {
            "status": "MONITOR_ONLY_UPSTREAM_PLAYER_MODEL_REQUIRED",
            "rows": int(mom_rows.sum()),
            "production_usable": False,
        }
    if result["targets"]:
        result["status"] = "READY"
    return result


def write_target_specific_experience(
    ledger_path="data/experience/prediction_ledger.csv",
    out_dir="artifacts/target_experience",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = learn_target_specific_experience(_read(Path(ledger_path)))
    pd.DataFrame(state["oos_rows"]).to_csv(out / "target_experience_oos.csv", index=False)
    (out / "target_experience_status.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return state


if __name__ == "__main__":
    print(json.dumps(write_target_specific_experience(), indent=2, ensure_ascii=False, default=str))