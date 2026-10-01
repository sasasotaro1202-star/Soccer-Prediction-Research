"""Target-specific historical binary learning for O/U and BTTS.

Research-only: each binary target gets its own chronological OOS model
selection. No target shares the selected model or calibration parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import classification_metrics
from src.prediction.secondary_outputs import fit_score_rate_model, predict_score_markets

TARGETS = {
    "O/U": lambda h, a: int(h + a >= 3),
    "BTTS": lambda h, a: int(h >= 1 and a >= 1),
}
META_COLUMNS = {
    "match_id","competition","season","season_start","kickoff_utc","home_team","away_team",
    "prediction_cutoff_at_utc","home_goals","away_goals","target","pit_verified",
    "feature_source_max_available_at_utc","source_available_at_utc",
}

def _models(random_state: int = 42):
    return {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, C=1.0, random_state=random_state)),
        ]),
        "logistic_strong": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, C=0.2, random_state=random_state)),
        ]),
        "hist_gb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=260, learning_rate=0.04, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=1.5, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)),
        ]),
    }

def _feature_cols(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=["number","bool"]).columns.tolist()
    cols = [c for c in numeric if c not in META_COLUMNS and not c.startswith("actual_") and not c.startswith("baseline_")]
    if not cols:
        raise ValueError("No numeric target-specific features available")
    return cols

def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=float).reshape(-1)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=int).reshape(-1)
    m = classification_metrics(y, np.column_stack([1.0-p, p]))
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "logloss": float(m["logloss"]),
        "brier": float(np.mean((p-y) ** 2)),
        "ece": float(m["ece"]),
    }

def _oos_blocks(df: pd.DataFrame, min_train: int, block_size: int):
    start = int(min_train)
    while start < len(df):
        end = min(start + int(block_size), len(df))
        yield df.iloc[:start].copy(), df.iloc[start:end].copy()
        start = end

def run_target_specific_learning(df: pd.DataFrame, *, min_train: int = 1000, block_size: int = 2000, min_blocks: int = 3) -> dict:
    required = {"kickoff_utc","home_goals","away_goals","pit_verified"}
    missing = sorted(required-set(df.columns))
    if missing: raise ValueError(f"Target-specific data missing columns: {missing}")
    d=df.copy()
    d["kickoff_utc"]=pd.to_datetime(d["kickoff_utc"],utc=True,errors="coerce")
    d["home_goals"]=pd.to_numeric(d["home_goals"],errors="coerce")
    d["away_goals"]=pd.to_numeric(d["away_goals"],errors="coerce")
    d=d[d["pit_verified"].astype("boolean").eq(True)].dropna(subset=["kickoff_utc","home_goals","away_goals"]).sort_values("kickoff_utc",kind="mergesort").reset_index(drop=True)
    if len(d) < min_train + block_size: raise ValueError(f"Not enough PIT-verified rows: {len(d)}")
    features=_feature_cols(d)
    all_rows=[]
    selections={}
    for target, label_fn in TARGETS.items():
        td=d.copy()
        td["_target_specific"]= [label_fn(int(h),int(a)) for h,a in td[["home_goals","away_goals"]].to_numpy()]
        target_blocks=[]
        candidates=_models()
        for block_id,(train,oos) in enumerate(_oos_blocks(td,min_train,block_size)):
            if len(train) < 120:
                raise ValueError(f"{target}: each OOS block requires at least 120 pre-block training rows")
            val_n = max(60, min(200, int(len(train) * 0.2)))
            val_n = min(val_n, len(train) - 60)
            fit = train.iloc[:-val_n]
            val = train.iloc[-val_n:]
            if len(fit) < 60 or len(val) < 60:
                raise ValueError(f"{target}: chronological validation split requires at least 60 fit and 60 validation rows")
            scores={}
            for name in candidates:
                model=_models()[name]
                model.fit(fit[features],fit["_target_specific"].astype(int))
                pv=model.predict_proba(val[features])[:,1]
                scores[name]=_metrics(val["_target_specific"],pv)
            score_model=fit_score_rate_model(fit)
            market_key="over_2_5" if target == "O/U" else "btts_yes"
            val_market=[predict_score_markets(score_model,row.home_team,row.away_team,row.competition).get(market_key,float("nan")) for row in val.itertuples(index=False)]
            score_metrics=_metrics(val["_target_specific"],np.asarray(val_market,dtype=float))
            scores["score_distribution"]=score_metrics
            selected=min(scores,key=lambda k:scores[k]["logloss"])
            if selected == "score_distribution":
                final_score_model=fit_score_rate_model(train)
                oos_market=[predict_score_markets(final_score_model,row.home_team,row.away_team,row.competition).get(market_key,float("nan")) for row in oos.itertuples(index=False)]
                po=np.asarray(oos_market,dtype=float)
            else:
                model=_models()[selected]
                model.fit(train[features],train["_target_specific"].astype(int))
                po=model.predict_proba(oos[features])[:,1]
            mm=_metrics(oos["_target_specific"],po)
            oos_score_model=fit_score_rate_model(train)
            oos_baseline=[predict_score_markets(oos_score_model,row.home_team,row.away_team,row.competition).get(market_key,float("nan")) for row in oos.itertuples(index=False)]
            baseline_mm=_metrics(oos["_target_specific"],np.asarray(oos_baseline,dtype=float))
            target_blocks.append({
                "target":target,
                "block":block_id,
                "oos_start":str(oos["kickoff_utc"].min()),
                "oos_end":str(oos["kickoff_utc"].max()),
                "selected_model":selected,
                "validation_selected_logloss":float(scores[selected]["logloss"]),
                "validation_score_distribution_logloss":float(score_metrics["logloss"]),
                "baseline_score_distribution_logloss":float(baseline_mm["logloss"]),
                "baseline_score_distribution_accuracy":float(baseline_mm["accuracy"]),
                "baseline_score_distribution_brier":float(baseline_mm["brier"]),
                "baseline_score_distribution_ece":float(baseline_mm["ece"]),
                **mm,
            })
        blocks=pd.DataFrame(target_blocks)
        if len(blocks)<min_blocks: raise ValueError(f"{target}: insufficient OOS blocks")
        locked=blocks.tail(2); development=blocks.iloc[:-2]
        development_improvement=bool(
            (development["logloss"].mean() < development["baseline_score_distribution_logloss"].mean())
            or (development["brier"].mean() < development["baseline_score_distribution_brier"].mean())
        )
        locked_non_regression=bool(
            (locked["logloss"] <= locked["baseline_score_distribution_logloss"]).all()
            and (locked["brier"] <= locked["baseline_score_distribution_brier"]).all()
            and (locked["ece"] <= locked["baseline_score_distribution_ece"]).all()
            and (locked["accuracy"] >= locked["baseline_score_distribution_accuracy"]).all()
        )
        selections[target]={
            "selected_by_block": target_blocks,
            "development_logloss": float(development["logloss"].mean()),
            "development_baseline_logloss": float(development["baseline_score_distribution_logloss"].mean()),
            "development_brier": float(development["brier"].mean()),
            "development_baseline_brier": float(development["baseline_score_distribution_brier"].mean()),
            "locked_logloss": float(locked["logloss"].mean()),
            "locked_baseline_logloss": float(locked["baseline_score_distribution_logloss"].mean()),
            "locked_brier": float(locked["brier"].mean()),
            "locked_baseline_brier": float(locked["baseline_score_distribution_brier"].mean()),
            "locked_baseline_accuracy": float(locked["baseline_score_distribution_accuracy"].mean()),
            "locked_ece": float(locked["ece"].mean()),
            "locked_baseline_ece": float(locked["baseline_score_distribution_ece"].mean()),
            "development_improvement": development_improvement,
            "locked_non_regression": locked_non_regression,
            "candidate_gate": bool(development_improvement and locked_non_regression),
            "locked_blocks_finite": bool(np.isfinite(locked[["logloss","brier","ece","accuracy"]].to_numpy(dtype=float)).all()),
            "training_source": "PIT_verified_historical_matches",
            "production_usable": False,
        }
        all_rows.extend(target_blocks)
    return {"schema_version":1,"status":"READY","targets":selections,"oos_rows":all_rows,"production_usable":False}

def write_target_specific_learning(df: pd.DataFrame, out_dir: str = "artifacts/target_specific") -> dict:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    state=run_target_specific_learning(df)
    pd.DataFrame(state["oos_rows"]).to_csv(out/"target_specific_oos.csv",index=False)
    (out/"target_specific_selection.json").write_text(json.dumps(state,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    (out/"target_specific_status.json").write_text(json.dumps({"status":state["status"],"targets":list(state["targets"]),"production_usable":False},indent=2),encoding="utf-8")
    return state
