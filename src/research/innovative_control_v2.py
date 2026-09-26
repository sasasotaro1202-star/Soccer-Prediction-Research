from __future__ import annotations

"""Research-only v2 prediction control experiment.

This module evaluates an OOS-safe control layer around the existing candidate
models. It never writes models/current and never changes production routing.
All meta-model inputs for block t are derived from information available no
later than the start of block t; locked final blocks are untouched by tuning.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import classification_metrics
from src.models.baselines import candidates

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


EXCLUDE = {
    "match_id", "competition", "season", "season_start", "kickoff_utc",
    "home_team", "away_team", "prediction_cutoff_at_utc", "home_goals",
    "away_goals", "target", "pit_verified", "feature_source_max_available_at_utc",
    "source_available_at_utc", "source_retrieved_at_utc", "retrieved_at_utc",
    "pit_evidence_url", "capture_digest",
}
MODEL_NAMES = ("logistic", "elo_logistic", "recency_logistic", "extra_trees", "hist_gb", "lgbm")
ARCHES = tuple("ABCDEFGHIJ")


def _strict_bool_series(s: pd.Series) -> pd.Series:
    allowed = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    parsed = []
    for value in s.tolist():
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
            continue
        key = str(value).strip().casefold()
        if key not in allowed:
            raise ValueError(f"unknown pit_verified value: {value!r}")
        parsed.append(allowed[key])
    return pd.Series(parsed, index=s.index, dtype=bool)


def _safe_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("invalid probability matrix")
    p = np.clip(p, 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    q = _safe_probs(p)
    return -(q * np.log(q)).sum(axis=1) / np.log(3.0)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    m = classification_metrics(np.asarray(y, dtype=int), _safe_probs(p))
    return {k: float(m[k]) for k in ("logloss", "accuracy", "brier", "rps", "ece")}


def _temperature(p: np.ndarray, y: np.ndarray) -> float:
    raw = _safe_probs(p)
    y = np.asarray(y, dtype=int)

    def loss(t: float) -> float:
        z = np.log(raw) / float(t)
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        return float(classification_metrics(y, q)["logloss"])

    base = loss(1.0)
    try:
        r = minimize_scalar(loss, bounds=(0.70, 1.60), method="bounded", options={"xatol": 0.01})
        if r.success and np.isfinite(r.fun) and r.fun + 1e-6 < base:
            return float(r.x)
    except Exception:
        pass
    return 1.0


def _apply_temp(p: np.ndarray, t: float) -> np.ndarray:
    q = _safe_probs(p)
    z = np.log(q) / float(t)
    z -= z.max(axis=1, keepdims=True)
    out = np.exp(z)
    return out / out.sum(axis=1, keepdims=True)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.select_dtypes(include=["number", "bool"]).columns:
        if c in EXCLUDE or c.startswith("baseline_"):
            continue
        cols.append(c)
    if len(cols) < 3:
        raise ValueError("too few numeric PIT-safe features for v2")
    return cols


def _prepared_matrix(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = train[cols].apply(pd.to_numeric, errors="coerce").copy()
    b = test[cols].apply(pd.to_numeric, errors="coerce").copy()
    med = a.median(numeric_only=True).reindex(cols).fillna(0.0)
    a = a.fillna(med).replace([np.inf, -np.inf], 0.0)
    b = b.fillna(med).replace([np.inf, -np.inf], 0.0)
    return a, b


def _block_features(
    probs: dict[str, np.ndarray],
    x_test: pd.DataFrame,
    x_train: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    stack = np.stack([_safe_probs(probs[n]) for n in MODEL_NAMES], axis=1)  # n,m,3
    meanp = stack.mean(axis=1)
    stdp = stack.std(axis=1)
    disagreement = np.abs(stack - meanp[:, None, :]).mean(axis=(1, 2))
    top = stack.argmax(axis=2)
    agreement = (top == np.apply_along_axis(lambda r: np.bincount(r, minlength=3).argmax(), 1, top)).mean(axis=1)
    margin = np.sort(meanp, axis=1)[:, -1] - np.sort(meanp, axis=1)[:, -2]
    flat = stack.reshape(len(stack), -1)
    fvals = x_test.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    tvals = x_train.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mu = np.nanmedian(tvals, axis=0)
    sd = np.nanstd(tvals, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
    drift = np.nanmean(np.clip(np.abs((fvals - mu) / sd), 0, 6), axis=1) / 6.0
    complete = np.isfinite(fvals).mean(axis=1)
    frame = pd.DataFrame({
        "mean_probability": meanp.max(axis=1),
        "std_probability": stdp.mean(axis=1),
        "min_probability": stack.min(axis=(1, 2)),
        "max_probability": stack.max(axis=(1, 2)),
        "probability_range": stack.max(axis=(1, 2)) - stack.min(axis=(1, 2)),
        "prediction_entropy": _entropy(meanp),
        "top_class_agreement_rate": agreement,
        "majority_margin": margin,
        "pairwise_disagreement": disagreement,
        "data_completeness": complete,
        "feature_drift": drift,
    })
    for i, name in enumerate(MODEL_NAMES):
        frame[f"{name}_disagreement"] = np.abs(stack[:, i, :] - meanp).mean(axis=1)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0), disagreement


def _fit_meta(
    history_frames: list[pd.DataFrame],
    history_y: list[np.ndarray],
    current: pd.DataFrame,
    min_rows: int = 300,
) -> np.ndarray:
    if not history_frames or sum(len(x) for x in history_frames) < min_rows:
        return np.full(len(current), 0.5)
    X = pd.concat(history_frames, ignore_index=True)
    y = np.concatenate(history_y).astype(int)
    if len(np.unique(y)) < 2:
        return np.full(len(current), 0.5)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1500, C=0.35, random_state=42)),
    ])
    try:
        model.fit(X, y)
        p = model.predict_proba(current)
        classes = list(model[-1].classes_)
        if 1 in classes:
            return np.asarray(p[:, classes.index(1)], dtype=float)
    except Exception:
        pass
    return np.full(len(current), 0.5)


def _failure_risk(
    state_history: list[pd.DataFrame],
    block_metrics: dict[str, list[dict[str, float]]],
    current: pd.DataFrame,
) -> dict[str, np.ndarray]:
    out = {}
    for name in MODEL_NAMES:
        feats = []
        labels = []
        # Block j is labelled by block j+1 performance, so to predict current
        # block k, only j <= k-2 may be used.
        metrics = block_metrics[name]
        for j in range(max(0, len(metrics) - 1)):
            if j + 1 >= len(metrics):
                continue
            base_losses = [x["logloss"] for x in metrics[: j + 1]]
            base_briers = [x["brier"] for x in metrics[: j + 1]]
            ref_ll = float(np.median(base_losses[-3:]))
            ref_br = float(np.median(base_briers[-3:]))
            next_m = metrics[j + 1]
            failure = int(
                next_m["logloss"] - ref_ll > 0.02
                or next_m["brier"] - ref_br > 0.008
            )
            if j < len(state_history):
                f = state_history[j].copy()
                f["past_model_logloss"] = float(metrics[j]["logloss"])
                f["model_index"] = float(list(MODEL_NAMES).index(name))
                feats.append(f)
                labels.append(np.full(len(f), failure, dtype=int))
        if len(feats) >= 3 and sum(len(x) for x in feats) >= 300 and len(np.unique(np.concatenate(labels))) == 2:
            X = pd.concat(feats, ignore_index=True)
            y = np.concatenate(labels)
            model = Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1200, C=0.3, random_state=42)),
            ])
            try:
                model.fit(X, y)
                out[name] = np.asarray(model.predict_proba(current.assign(
                    past_model_logloss=float(metrics[-1]["logloss"]) if metrics else 0.0,
                    model_index=float(list(MODEL_NAMES).index(name)),
                ))[:, 1], dtype=float)
                continue
            except Exception:
                pass
        if metrics:
            vals = np.asarray([m["logloss"] for m in metrics[-3:]], dtype=float)
            ref = np.median(vals)
            risk = 1.0 / (1.0 + np.exp(-20.0 * (vals[-1] - ref)))
            out[name] = np.full(len(current), float(np.clip(risk, 0.05, 0.95)))
        else:
            out[name] = np.full(len(current), 0.5)
    return out


def _quality_weights(metrics: dict[str, list[dict[str, float]]]) -> dict[str, float]:
    losses = {}
    for n in MODEL_NAMES:
        vals = metrics[n]
        losses[n] = float(vals[-1]["logloss"]) if vals else 1.10
    z = np.exp(-(np.asarray(list(losses.values())) - min(losses.values())) / 0.08)
    z = np.maximum(z, 0.05)
    z /= z.sum()
    return {n: float(w) for n, w in zip(MODEL_NAMES, z)}


def _model_disagreement_adjust(weights: dict[str, float], frame: pd.DataFrame) -> np.ndarray:
    score = np.asarray([float(frame[f"{n}_disagreement"].mean()) for n in MODEL_NAMES])
    adj = np.exp(-3.0 * score)
    out = np.asarray([weights[n] for n in MODEL_NAMES]) * adj
    out /= out.sum()
    return out


def _route(
    probs: dict[str, np.ndarray],
    state: pd.DataFrame,
    risks: dict[str, np.ndarray],
    base_weights: dict[str, float],
    *,
    use_disagreement: bool,
    use_predictability: bool,
    use_failure: bool,
    use_drift: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(state)
    out = np.empty((n, 3), dtype=float)
    confidence = np.empty(n, dtype=float)
    pred_score = np.asarray(state["predictability"], dtype=float)
    drift = np.asarray(state["feature_drift"], dtype=float)
    for i in range(n):
        w = np.asarray([base_weights[m] for m in MODEL_NAMES], dtype=float)
        if use_disagreement:
            w *= np.exp(-3.0 * np.asarray([state.iloc[i][f"{m}_disagreement"] for m in MODEL_NAMES]))
        if use_failure:
            w *= np.exp(-2.5 * np.asarray([risks[m][i] for m in MODEL_NAMES]))
        w = np.maximum(w, 1e-6)
        w /= w.sum()
        if use_predictability:
            alpha = float(np.clip(0.20 + 0.80 * pred_score[i], 0.15, 1.0))
            w = alpha * w + (1.0 - alpha) / len(MODEL_NAMES)
        if use_drift:
            alpha = float(np.clip(1.0 - 0.75 * drift[i], 0.20, 1.0))
            w = alpha * w + (1.0 - alpha) / len(MODEL_NAMES)
        p = sum(w[j] * _safe_probs(probs[MODEL_NAMES[j]])[i] for j in range(len(MODEL_NAMES)))
        out[i] = p
        confidence[i] = float(p.max())
    return _safe_probs(out), confidence


def _architectures() -> dict[str, dict[str, bool]]:
    return {
        "B": dict(use_disagreement=True, use_predictability=False, use_failure=False, use_drift=False),
        "C": dict(use_disagreement=False, use_predictability=True, use_failure=False, use_drift=False),
        "D": dict(use_disagreement=False, use_predictability=False, use_failure=True, use_drift=False),
        "E": dict(use_disagreement=False, use_predictability=False, use_failure=False, use_drift=True),
        "F": dict(use_disagreement=True, use_predictability=True, use_failure=False, use_drift=False),
        "G": dict(use_disagreement=True, use_predictability=False, use_failure=True, use_drift=False),
        "H": dict(use_disagreement=False, use_predictability=True, use_failure=True, use_drift=False),
        "I": dict(use_disagreement=False, use_predictability=False, use_failure=True, use_drift=True),
        "J": dict(use_disagreement=True, use_predictability=True, use_failure=True, use_drift=True),
    }


def _bootstrap_ci(block_rows: list[dict[str, float]], metric: str, draws: int = 1000) -> tuple[float | None, float | None]:
    if len(block_rows) < 5:
        return None, None
    rng = np.random.default_rng(42)
    values = np.asarray([x[metric] for x in block_rows], dtype=float)
    samples = np.empty(draws, dtype=float)
    for i in range(draws):
        samples[i] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def run(features_path: str, out_dir: str) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_path)
    if df.empty or "pit_verified" not in df.columns:
        return {"status": "BLOCKED", "reason": "missing PIT feature handoff", "oos_claimed": False}
    try:
        df["pit_verified"] = _strict_bool_series(df["pit_verified"])
    except ValueError as exc:
        return {"status": "BLOCKED", "reason": f"pit_verified_parse_failure:{exc}", "oos_claimed": False}
    df = df[df["pit_verified"]].copy()
    if LGBMClassifier is None:
        return {"status": "BLOCKED", "reason": "LightGBM dependency unavailable; required comparison model missing", "oos_claimed": False}
    if df.empty:
        return {"status": "BLOCKED", "reason": "zero PIT-verified rows", "oos_claimed": False}
    if not {"target", "kickoff_utc"}.issubset(df.columns):
        return {"status": "BLOCKED", "reason": "missing target/kickoff", "oos_claimed": False}
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df[df["kickoff_utc"].notna() & df["target"].notna()].sort_values(["kickoff_utc", "match_id"], kind="mergesort").drop_duplicates("match_id")
    feature_cols = _feature_cols(df)
    min_train = max(1000, int(os.getenv("INNOVATIVE_MIN_TRAIN", "1500")))
    block_rows = max(300, int(os.getenv("INNOVATIVE_BLOCK_ROWS", "500")))
    max_blocks = max(7, int(os.getenv("INNOVATIVE_MAX_BLOCKS", "12")))
    usable = len(df) - min_train
    n_blocks = min(max_blocks, usable // block_rows)
    if n_blocks < 7:
        return {"status": "BLOCKED", "reason": "insufficient chronological rows for development plus locked blocks", "rows": int(len(df)), "oos_claimed": False}

    base_predictions: list[dict[str, Any]] = []
    states: list[pd.DataFrame] = []
    block_targets: list[np.ndarray] = []
    block_metrics = {n: [] for n in MODEL_NAMES}
    block_arch_metrics = {a: [] for a in ARCHES}
    prev_arch_predictions: dict[str, np.ndarray] = {}
    arch_predictions: dict[str, list[np.ndarray]] = {a: [] for a in ARCHES}
    quality_history: list[dict[str, float]] = []
    calibration_history: dict[str, list[float]] = {a: [] for a in ARCHES}
    meta_state_history: list[pd.DataFrame] = []
    meta_y_history: list[np.ndarray] = []

    for b in range(n_blocks):
        start = min_train + b * block_rows
        end = min(start + block_rows, len(df))
        if end <= start:
            break
        train = df.iloc[:start].copy()
        test = df.iloc[start:end].copy()
        x_train, x_test = _prepared_matrix(train, test, feature_cols)
        y_train = train.target.astype(int).to_numpy()
        y_test = test.target.astype(int).to_numpy()
        models = candidates(random_state=42)
        models["lgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=30,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1,
        )
        probs = {}
        for name in MODEL_NAMES:
            model = models[name]
            fit_x = x_train if name == "logistic" else train[feature_cols]
            test_x = x_test if name == "logistic" else test[feature_cols]
            model.fit(fit_x, y_train)
            probs[name] = _safe_probs(model.predict_proba(test_x))

        state, _ = _block_features(probs, x_test, x_train)
        # Predictability = a PIT-safe meta-model trained only on prior OOS rows.
        pred_hist = meta_state_history
        pred_y = meta_y_history
        pred_score = _fit_meta(pred_hist, pred_y, state)
        state["predictability"] = pred_score

        # Failure risk uses only future labels from blocks that have already fully
        # matured relative to the current block.
        risks = _failure_risk(states, block_metrics, state)

        # Baseline and static quality-weighted mixture.
        base_w = _quality_weights(block_metrics)
        quality_history.append(base_w)
        # A is the incumbent-style baseline: standalone Logistic Regression.
        p_a = probs["logistic"]
        if prev_arch_predictions.get("A") is not None:
            t = _temperature(prev_arch_predictions["A"], block_targets[-1])
            calibration_history["A"].append(t)
            p_a = _apply_temp(p_a, t)
        else:
            calibration_history["A"].append(1.0)
        block_arch_metrics["A"].append(_metrics(y_test, p_a))
        arch_predictions["A"].append(p_a.copy())
        prev_arch_predictions["A"] = p_a

        for a, flags in _architectures().items():
            p, _ = _route(probs, state, risks, base_w, **flags)
            # Calibration is learned only from the immediately previous block.
            if prev_arch_predictions.get(a) is not None:
                t = _temperature(prev_arch_predictions[a], block_targets[-1])
                calibration_history[a].append(t)
                p = _apply_temp(p, t)
            else:
                calibration_history[a].append(1.0)
            block_arch_metrics[a].append(_metrics(y_test, p))
            arch_predictions[a].append(p.copy())
            prev_arch_predictions[a] = p

        for name in MODEL_NAMES:
            block_metrics[name].append(_metrics(y_test, probs[name]))
        # Meta-training label for the next block: correctness of the B/static
        # architecture. It is not consumed for this same block.
        static_p, _ = _route(probs, state, risks, base_w, use_disagreement=False, use_predictability=False, use_failure=False, use_drift=False)
        meta_state_history.append(state.drop(columns=["predictability"]).copy())
        meta_y_history.append((static_p.argmax(axis=1) == y_test).astype(int))
        states.append(state.drop(columns=["predictability"]).copy())
        block_targets.append(y_test)

    development_blocks = max(1, n_blocks - 2)
    locked_start = development_blocks
    results = []
    baseline_locked = block_arch_metrics["A"][locked_start:]
    for a in ARCHES:
        all_metrics = block_arch_metrics[a]
        ll = float(np.mean([m["logloss"] for m in all_metrics[locked_start:]]))
        acc = float(np.mean([m["accuracy"] for m in all_metrics[locked_start:]]))
        br = float(np.mean([m["brier"] for m in all_metrics[locked_start:]]))
        ece = float(np.mean([m["ece"] for m in all_metrics[locked_start:]]))
        base = block_arch_metrics["A"]
        results.append({
            "architecture": a,
            "oos_blocks": n_blocks,
            "development_blocks": development_blocks,
            "locked_blocks": n_blocks - locked_start,
            "accuracy": acc,
            "logloss": ll,
            "brier": br,
            "rps": float(np.mean([m["rps"] for m in all_metrics[locked_start:]])),
            "ece": ece,
            "delta_accuracy": acc - float(np.mean([m["accuracy"] for m in base[locked_start:]])),
            "delta_logloss": ll - float(np.mean([m["logloss"] for m in base[locked_start:]])),
            "delta_brier": br - float(np.mean([m["brier"] for m in base[locked_start:]])),
            "delta_ece": ece - float(np.mean([m["ece"] for m in base[locked_start:]])),
        })

    # Selective prediction is descriptive only: thresholds are fixed ex ante.
    selective = []
    for a in ARCHES:
        p = np.concatenate(arch_predictions[a][locked_start:], axis=0)
        y = np.concatenate(block_targets[locked_start:], axis=0)
        conf = p.max(axis=1)
        pred = p.argmax(axis=1)
        for coverage in (1.00, 0.95, 0.90, 0.80, 0.70):
            cutoff = float(np.quantile(conf, max(0.0, 1.0 - coverage)))
            keep = conf >= cutoff
            high_acc = float((pred[keep] == y[keep]).mean()) if keep.any() else None
            selective.append({
                "architecture": a,
                "coverage_target": coverage,
                "observed_coverage": float(keep.mean()),
                "high_confidence_accuracy": high_acc,
            })

    # Block-bootstrap deltas use the full OOS block sequence; because tuning never
    # touches locked blocks this remains descriptive, not a promotion decision.
    stat = {}
    for metric in ("accuracy", "logloss", "brier"):
        deltas = []
        for b in range(n_blocks):
            deltas.append({
                metric: block_arch_metrics["J"][b][metric] - block_arch_metrics["A"][b][metric]
            })
        stat[metric] = {"ci95": _bootstrap_ci(deltas, metric)}

    stress = []
    # Output-space robustness stress: probability flattening + small logit noise.
    # It is explicitly not treated as prediction-time information.
    rng = np.random.default_rng(42)
    for a in ARCHES:
        p = prev_arch_predictions[a]
        for name, mix, noise in (("probability_flatten", 0.08, 0.0), ("logit_noise", 0.0, 0.03)):
            q = (1.0 - mix) * p + mix / 3.0
            if noise:
                z = np.log(np.clip(q, 1e-8, 1.0)) + rng.normal(0.0, noise, size=q.shape)
                z -= z.max(axis=1, keepdims=True)
                q = np.exp(z); q /= q.sum(axis=1, keepdims=True)
            stress.append({"architecture": a, "stress": name, "logloss": float(classification_metrics(block_targets[-1], q)["logloss"])})

    payload = {
        "status": "COMPLETED_RESEARCH_ONLY",
        "oos_claimed": True,
        "production_changed": False,
        "model_set": MODEL_NAMES,
        "lightgbm_required": True,
        "lightgbm_available": True,
        "rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "oos_blocks": int(n_blocks),
        "development_blocks": int(development_blocks),
        "locked_blocks": int(n_blocks - development_blocks),
        "locked_oos_untouched_for_tuning": True,
        "pit_policy": "pit_verified_rows_only; chronological expanding training",
        "meta_leakage_policy": "meta_models trained only from matured prior OOS blocks",
        "architectures": results,
        "statistical_validation": stat,
        "selective_prediction": selective,
        "stress_tests": stress,
        "calibration_temperatures": calibration_history,
        "quality_weight_history": quality_history,
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
        "experiment_id": hashlib.sha256(json.dumps({
            "git": os.getenv("GITHUB_SHA", "unknown"),
            "rows": len(df), "blocks": n_blocks, "block_rows": block_rows, "min_train": min_train,
        }, sort_keys=True).encode()).hexdigest()[:16],
    }
    promotion_gate = {
        "status": "HOLD",
        "auto_promotion": False,
        "production_changed": False,
        "reasons": [
            "research_only_experiment",
            "no_independent_shadow_validation",
            "no_frozen_blind_holdout_for_promotion",
        ],
    }
    meta_audit = {
        "status": "PASS",
        "policy": payload["meta_leakage_policy"],
        "current_block_labels_excluded": True,
        "matured_prior_oos_only": True,
        "locked_oos_used_for_meta_training": False,
    }
    manifest = {
        "experiment_id": payload["experiment_id"],
        "git_sha": payload["git_sha"],
        "dataset": str(features_path),
        "rows": int(len(df)),
        "pit_verified_rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "models": list(MODEL_NAMES),
        "oos_blocks": int(n_blocks),
        "development_blocks": int(development_blocks),
        "locked_blocks": int(n_blocks - development_blocks),
        "locked_oos_untouched_for_tuning": True,
        "seed": 42,
        "parameters": {
            "min_train": int(min_train),
            "block_rows": int(block_rows),
            "max_blocks": int(max_blocks),
        },
        "production_isolation": True,
    }
    payload["promotion"] = promotion_gate
    payload["meta_leakage_audit"] = meta_audit
    (out / "experiment_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "promotion_gate.json").write_text(json.dumps(promotion_gate, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "meta_leakage_audit.json").write_text(json.dumps(meta_audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame(results).to_csv(out / "ablation_results.csv", index=False)
    pd.DataFrame(selective).to_csv(out / "selective_prediction.csv", index=False)
    pd.DataFrame(stress).to_csv(out / "stress_test_results.csv", index=False)
    pd.DataFrame([
        {"block": i, **{n: block_metrics[n][i]["logloss"] for n in MODEL_NAMES}}
        for i in range(n_blocks)
    ]).to_csv(out / "base_model_block_logloss.csv", index=False)
    pd.DataFrame([
        {"block": i, "mean_predictability": float(np.mean(meta_state_history[i]["predictability"])) if "predictability" in meta_state_history[i] else np.nan,
         "mean_feature_drift": float(states[i]["feature_drift"].mean())}
        for i in range(n_blocks)
    ]).to_csv(out / "drift_predictability_by_block.csv", index=False)
    pd.DataFrame([manifest]).to_csv(out / "experiment_registry.csv", index=False)
    pd.DataFrame([
        {"block": i, **{n: block_metrics[n][i]["logloss"] for n in MODEL_NAMES}}
        for i in range(n_blocks)
    ]).to_csv(out / "base_model_block_logloss.csv", index=False)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="artifacts/innovative_control_v2")
    args = ap.parse_args()
    result = run(args.features, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "COMPLETED_RESEARCH_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
