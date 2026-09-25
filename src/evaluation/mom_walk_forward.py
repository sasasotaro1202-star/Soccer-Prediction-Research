"""Chronological walk-forward evaluation for PIT-safe MOM ranking candidates.

Research-only: this module never creates or treats missing historical player data as
zero performance. It requires explicit PIT verification and exactly one MOM label
per match. Locked holdout blocks are evaluated only after all development choices
are fixed by the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.mom_model import MOM_FEATURE_COLUMNS, fit_mom_model, predict_mom_distribution


@dataclass(frozen=True)
class MOMBlockMetrics:
    block: int
    train_matches: int
    test_matches: int
    test_rows: int
    top1_hit_rate: float
    top4_hit_rate: float
    mrr: float
    ndcg_at_4: float
    logloss: float
    brier: float
    ece: float
    mean_candidate_count: float
    min_candidate_count: int


def _validate_history(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "player_id",
        "kickoff_utc",
        "feature_available_at_utc",
        "pit_verified",
        "is_motm",
        *MOM_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MOM WFO data missing required columns: {missing}")

    d = frame.copy()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    d["feature_available_at_utc"] = pd.to_datetime(
        d["feature_available_at_utc"], utc=True, errors="coerce"
    )
    if d["kickoff_utc"].isna().any() or d["feature_available_at_utc"].isna().any():
        raise RuntimeError("MOM WFO timestamps are invalid or missing")
    d["pit_verified"] = d["pit_verified"].astype("boolean")
    if d["pit_verified"].isna().any():
        raise RuntimeError("MOM WFO pit_verified contains unknown values")
    d = d.loc[d["pit_verified"]].copy()
    if d.empty:
        raise RuntimeError("MOM WFO has no PIT-verified rows")

    d["match_id"] = d["match_id"].astype("string").str.strip()
    d["player_id"] = d["player_id"].astype("string").str.strip()
    if d["match_id"].eq("").any() or d["player_id"].eq("").any():
        raise RuntimeError("MOM WFO contains empty match_id/player_id")
    if d.duplicated(["match_id", "player_id"]).any():
        raise RuntimeError("MOM WFO contains duplicate match_id/player_id rows")
    if bool((d["feature_available_at_utc"] > d["kickoff_utc"]).any()):
        raise RuntimeError("MOM WFO contains a feature timestamp after kickoff")

    d["is_motm"] = pd.to_numeric(d["is_motm"], errors="coerce")
    if d["is_motm"].isna().any() or ~d["is_motm"].isin([0, 1]).all():
        raise RuntimeError("MOM WFO is_motm must be 0/1")
    counts = d.groupby("match_id", sort=False)["is_motm"].sum()
    if bool((counts != 1).any()):
        raise RuntimeError("MOM WFO requires exactly one positive MOTM label per match")

    for column in MOM_FEATURE_COLUMNS:
        raw = d[column]
        values = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & values.isna()
        if bool(invalid.any()):
            raise RuntimeError(f"MOM WFO feature column {column!r} contains non-numeric values")
        if bool(np.isinf(values.to_numpy(dtype=float)).any()):
            raise RuntimeError(f"MOM WFO feature column {column!r} contains infinite values")
        d[column] = values
    # Missing feature values are retained explicitly. The MOM estimator applies
    # fit-slice-only median imputation plus missingness indicators.


    return d.sort_values(["kickoff_utc", "match_id", "player_id"], kind="mergesort").reset_index(drop=True)


def _split_match_blocks(match_ids: list[str], n_blocks: int) -> list[list[str]]:
    if n_blocks < 3:
        raise ValueError("MOM WFO requires at least 3 blocks")
    if len(match_ids) < n_blocks:
        raise RuntimeError("Insufficient chronological matches for requested MOM WFO blocks")
    chunks = np.array_split(np.asarray(match_ids, dtype=object), n_blocks)
    blocks = [[str(x) for x in chunk.tolist()] for chunk in chunks if len(chunk)]
    if len(blocks) < 3:
        raise RuntimeError("MOM WFO produced fewer than 3 non-empty blocks")
    return blocks


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < bins - 1 else confidence <= hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(total)


def _evaluate_predictions(test: pd.DataFrame, distribution: pd.DataFrame) -> dict[str, float]:
    rows = []
    for match_id, group in distribution.groupby("match_id", sort=False):
        truth = test.loc[test["match_id"] == match_id]
        if truth.empty:
            raise RuntimeError(f"MOM WFO prediction has no truth rows for {match_id}")
        actual = str(truth.loc[truth["is_motm"] == 1, "player_id"].iloc[0])
        g = group.sort_values(["probability", "player_id"], ascending=[False, True], kind="mergesort")
        ids = g["player_id"].astype(str).tolist()
        probs = g["probability"].to_numpy(dtype=float)
        if len(ids) < 4:
            raise RuntimeError(f"MOM WFO match {match_id} has fewer than 4 candidates")
        if not np.isfinite(probs).all() or abs(float(probs.sum()) - 1.0) > 1e-9:
            raise RuntimeError(f"MOM WFO probabilities invalid for {match_id}")

        rank = ids.index(actual) + 1 if actual in ids else len(ids) + 1
        top1 = float(rank == 1)
        top4 = float(rank <= 4)
        reciprocal = float(1.0 / rank) if rank <= len(ids) else 0.0
        ndcg = float(1.0 / np.log2(rank + 1)) if rank <= 4 else 0.0
        true_index = ids.index(actual) if actual in ids else None
        true_prob = float(probs[true_index]) if true_index is not None else 1e-9
        y = np.zeros(len(probs), dtype=float)
        if true_index is not None:
            y[true_index] = 1.0
        brier = float(np.sum((probs - y) ** 2))
        rows.append({
            "top1": top1,
            "top4": top4,
            "mrr": reciprocal,
            "ndcg": ndcg,
            "logloss": -float(np.log(np.clip(true_prob, 1e-9, 1.0))),
            "brier": brier,
            "confidence": float(probs[0]),
            "correct": top1,
            "candidate_count": len(ids),
        })

    m = pd.DataFrame(rows)
    return {
        "top1_hit_rate": float(m["top1"].mean()),
        "top4_hit_rate": float(m["top4"].mean()),
        "mrr": float(m["mrr"].mean()),
        "ndcg_at_4": float(m["ndcg"].mean()),
        "logloss": float(m["logloss"].mean()),
        "brier": float(m["brier"].mean()),
        "ece": _ece(m["confidence"].to_numpy(), m["correct"].to_numpy()),
        "mean_candidate_count": float(m["candidate_count"].mean()),
        "min_candidate_count": int(m["candidate_count"].min()),
    }


def run_mom_walk_forward(
    history: pd.DataFrame,
    *,
    n_blocks: int = 6,
    locked_blocks: int = 2,
    min_train_matches: int = 10,
    regularization_c: float = 0.30,
    temperature: float = 1.0,
    random_state: int = 42,
    method: str = "binary_logit",
) -> pd.DataFrame:
    """Run expanding-window chronological OOS evaluation by complete match blocks."""
    d = _validate_history(history)
    match_order = (
        d[["match_id", "kickoff_utc"]]
        .drop_duplicates("match_id")
        .sort_values(["kickoff_utc", "match_id"], kind="mergesort")
        ["match_id"].astype(str).tolist()
    )
    blocks = _split_match_blocks(match_order, n_blocks)

    metrics: list[MOMBlockMetrics] = []
    for block_idx in range(1, len(blocks)):
        train_ids = [m for block in blocks[:block_idx] for m in block]
        test_ids = blocks[block_idx]
        if len(train_ids) < int(min_train_matches):
            continue

        train = d[d["match_id"].isin(train_ids)].copy()
        test = d[d["match_id"].isin(test_ids)].copy()

        if method == "soft_ensemble":
            # Fixed, pre-registered equal-weight ensemble. No OOS/locked data
            # is inspected to tune the weights, so this is a variance-reduction
            # challenger rather than another selection parameter.
            component_methods = ("binary_logit", "conditional_logit", "hist_gbdt")
            component_models = [
                fit_mom_model(
                    train,
                    regularization_c=regularization_c,
                    temperature=temperature,
                    min_matches=min_train_matches,
                    random_state=random_state,
                    method=component,
                )
                for component in component_methods
            ]
            distributions = []
            for match_id, group in test.groupby("match_id", sort=False):
                kickoff = group["kickoff_utc"].iloc[0]
                prediction_time = kickoff - pd.Timedelta(seconds=1)
                pred_input = group.drop(columns=["is_motm"]).copy()
                component_dists = [
                    predict_mom_distribution(
                        model_bundle,
                        pred_input,
                        prediction_time=prediction_time,
                    )
                    for model_bundle in component_models
                ]
                merged = component_dists[0][["match_id", "player_id", "kickoff_utc", "probability"]].copy()
                merged = merged.rename(columns={"probability": "p0"})
                for idx, component_dist in enumerate(component_dists[1:], start=1):
                    part = component_dist[["match_id", "player_id", "probability"]].copy()
                    if part.duplicated(["match_id", "player_id"]).any():
                        raise RuntimeError("MOM soft ensemble component contains duplicate candidates")
                    merged = merged.merge(
                        part.rename(columns={"probability": f"p{idx}"}),
                        on=["match_id", "player_id"],
                        how="outer",
                        validate="one_to_one",
                    )
                probability_cols = [f"p{i}" for i in range(len(component_methods))]
                if merged[probability_cols].isna().any().any():
                    raise RuntimeError(f"MOM soft ensemble candidate sets differ for match_id={match_id!r}")
                merged["probability"] = merged[probability_cols].mean(axis=1)
                merged = merged[["match_id", "player_id", "kickoff_utc", "probability"]]
                total = float(merged["probability"].sum())
                if not np.isfinite(total) or total <= 0:
                    raise RuntimeError(f"MOM soft ensemble produced invalid probabilities for match_id={match_id!r}")
                merged["probability"] = merged["probability"] / total
                distributions.append(merged)
        elif method == "rank_consensus":
            # Fixed Borda-style rank consensus. Each component contributes only
            # its within-match ordering; no OOS/locked data is used to tune weights.
            component_methods = ("binary_logit", "conditional_logit", "hist_gbdt")
            component_models = [
                fit_mom_model(
                    train,
                    regularization_c=regularization_c,
                    temperature=temperature,
                    min_matches=min_train_matches,
                    random_state=random_state,
                    method=component,
                )
                for component in component_methods
            ]
            distributions = []
            for match_id, group in test.groupby("match_id", sort=False):
                kickoff = group["kickoff_utc"].iloc[0]
                prediction_time = kickoff - pd.Timedelta(seconds=1)
                pred_input = group.drop(columns=["is_motm"]).copy()
                component_dists = [
                    predict_mom_distribution(
                        model_bundle,
                        pred_input,
                        prediction_time=prediction_time,
                    )
                    for model_bundle in component_models
                ]
                merged = component_dists[0][["match_id", "player_id", "kickoff_utc", "probability"]].copy()
                merged = merged.rename(columns={"probability": "p0"})
                for idx, component_dist in enumerate(component_dists[1:], start=1):
                    part = component_dist[["match_id", "player_id", "probability"]].copy()
                    part = part.rename(columns={"probability": f"p{idx}"})
                    merged = merged.merge(
                        part,
                        on=["match_id", "player_id"],
                        how="outer",
                        validate="one_to_one",
                    )
                probability_cols = [f"p{i}" for i in range(len(component_methods))]
                if merged[probability_cols].isna().any().any():
                    raise RuntimeError(f"MOM rank consensus candidate sets differ for match_id={match_id!r}")
                rank_scores = []
                for col in probability_cols:
                    ordered = merged.sort_values(
                        [col, "player_id"],
                        ascending=[False, True],
                        kind="mergesort",
                    )
                    ranks = pd.Series(
                        np.arange(len(ordered), 0, -1, dtype=float) / float(max(len(ordered), 1)),
                        index=ordered.index,
                    )
                    rank_scores.append(ranks.reindex(merged.index).to_numpy(dtype=float))
                merged["probability"] = np.mean(np.stack(rank_scores, axis=0), axis=0)
                total = float(merged["probability"].sum())
                if not np.isfinite(total) or total <= 0:
                    raise RuntimeError(f"MOM rank consensus produced invalid probabilities for match_id={match_id!r}")
                merged["probability"] /= total
                distributions.append(merged[["match_id", "player_id", "kickoff_utc", "probability"]])
        else:
            model = fit_mom_model(
                train,
                regularization_c=regularization_c,
                temperature=temperature,
                min_matches=min_train_matches,
                random_state=random_state,
                method=method,
            )
            distributions = []
            for match_id, group in test.groupby("match_id", sort=False):
                kickoff = group["kickoff_utc"].iloc[0]
                prediction_time = kickoff - pd.Timedelta(seconds=1)
                pred_input = group.drop(columns=["is_motm"]).copy()
                dist = predict_mom_distribution(
                    model,
                    pred_input,
                    prediction_time=prediction_time,
                )
                distributions.append(dist)

        if not distributions:
            continue
        dist = pd.concat(distributions, ignore_index=True)
        evaluated = _evaluate_predictions(test, dist)
        metrics.append(MOMBlockMetrics(
            block=block_idx,
            train_matches=len(train_ids),
            test_matches=len(test_ids),
            test_rows=len(test),
            **evaluated,
        ))

    if not metrics:
        raise RuntimeError("MOM WFO produced no evaluable OOS blocks")
    return pd.DataFrame([asdict(x) for x in metrics])



def _apply_matchwise_temperature(distribution: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Apply a positive temperature to each complete match distribution."""
    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    d = distribution.copy()
    p = np.clip(pd.to_numeric(d["probability"], errors="coerce").to_numpy(dtype=float), 1e-12, 1.0)
    if not np.isfinite(p).all():
        raise RuntimeError("MOM calibration probabilities are invalid")
    transformed = np.empty_like(p)
    for positions in d.groupby("match_id", sort=False).indices.values():
        q = np.power(p[np.asarray(positions, dtype=int)], 1.0 / float(temperature))
        total = float(q.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("MOM calibration produced an invalid match probability sum")
        transformed[np.asarray(positions, dtype=int)] = q / total
    d["probability"] = transformed
    return d


def _fit_matchwise_temperature(
    calibration: pd.DataFrame,
    distribution: pd.DataFrame,
    *,
    min_matches: int = 30,
    prior_strength: float = 120.0,
) -> tuple[float, bool]:
    """Fit low-dimensional temperature on a chronological calibration slice only."""
    truth = (
        calibration.loc[calibration["is_motm"] == 1, ["match_id", "player_id"]]
        .drop_duplicates("match_id")
    )
    groups = list(distribution.groupby("match_id", sort=False))
    if len(groups) < int(min_matches):
        return 1.0, False

    truth_map = {str(row.match_id): str(row.player_id) for row in truth.itertuples(index=False)}
    valid_groups = []
    for match_id, group in groups:
        actual = truth_map.get(str(match_id))
        if actual is None or actual not in set(group["player_id"].astype(str)):
            continue
        probs = pd.to_numeric(group["probability"], errors="coerce").to_numpy(dtype=float)
        ids = group["player_id"].astype(str).tolist()
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            raise RuntimeError(f"MOM calibration distribution invalid for match_id={match_id!r}")
        valid_groups.append((ids, probs, actual))
    if len(valid_groups) < int(min_matches):
        return 1.0, False

    def loss(temperature: float) -> float:
        total = 0.0
        for ids, probs, actual in valid_groups:
            q = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / float(temperature))
            q /= q.sum()
            index = ids.index(actual)
            total -= float(np.log(np.clip(q[index], 1e-12, 1.0)))
        return total / float(len(valid_groups))

    candidates = np.linspace(0.70, 1.60, 91)
    values = np.asarray([loss(float(t)) for t in candidates], dtype=float)
    best_index = int(np.argmin(values))
    fitted = float(candidates[best_index])
    raw_loss = float(loss(1.0))
    if not np.isfinite(values[best_index]) or values[best_index] + 1e-6 >= raw_loss:
        return 1.0, False

    alpha = float(len(valid_groups) / (len(valid_groups) + max(float(prior_strength), 1.0)))
    shrunk = float(1.0 + alpha * (fitted - 1.0))
    return float(np.clip(shrunk, 0.70, 1.60)), True


def run_mom_walk_forward_calibrated_soft_ensemble(
    history: pd.DataFrame,
    *,
    n_blocks: int = 6,
    locked_blocks: int = 2,
    min_train_matches: int = 30,
    calibration_fraction: float = 0.25,
    min_calibration_matches: int = 30,
    prior_strength: float = 120.0,
    regularization_c: float = 0.30,
    temperature: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Nested chronological WFO for an equal-weight MOM ensemble plus post-hoc temperature.

    The temperature is fitted only on the tail of each training window, then the
    ensemble is refit on the full training window before OOS prediction. Ranking is
    invariant to positive temperature scaling; only probability calibration changes.
    """
    d = _validate_history(history)
    match_order = (
        d[["match_id", "kickoff_utc"]]
        .drop_duplicates("match_id")
        .sort_values(["kickoff_utc", "match_id"], kind="mergesort")
        ["match_id"].astype(str).tolist()
    )
    blocks = _split_match_blocks(match_order, n_blocks)

    metrics: list[MOMBlockMetrics] = []
    component_methods = ("binary_logit", "conditional_logit", "hist_gbdt")
    for block_idx in range(1, len(blocks)):
        train_ids = [m for block in blocks[:block_idx] for m in block]
        test_ids = blocks[block_idx]
        if len(train_ids) < int(min_train_matches):
            continue

        calibration_count = max(
            int(min_calibration_matches),
            int(round(len(train_ids) * float(calibration_fraction))),
        )
        calibration_count = min(
            calibration_count,
            max(1, len(train_ids) - int(min_train_matches)),
        )
        fit_ids = train_ids[:-calibration_count]
        calibration_ids = train_ids[-calibration_count:]
        fit = d[d["match_id"].isin(fit_ids)].copy()
        calibration = d[d["match_id"].isin(calibration_ids)].copy()
        test = d[d["match_id"].isin(test_ids)].copy()

        calibration_models = [
            fit_mom_model(
                fit,
                regularization_c=regularization_c,
                temperature=temperature,
                min_matches=min_train_matches,
                random_state=random_state,
                method=component,
            )
            for component in component_methods
        ]
        calibration_parts = []
        for match_id, group in calibration.groupby("match_id", sort=False):
            kickoff = group["kickoff_utc"].iloc[0]
            prediction_time = kickoff - pd.Timedelta(seconds=1)
            pred_input = group.drop(columns=["is_motm"]).copy()
            component_dists = [
                predict_mom_distribution(
                    model_bundle,
                    pred_input,
                    prediction_time=prediction_time,
                )
                for model_bundle in calibration_models
            ]
            merged = component_dists[0][["match_id", "player_id", "kickoff_utc", "probability"]].rename(
                columns={"probability": "p0"}
            )
            for idx, component_dist in enumerate(component_dists[1:], start=1):
                part = component_dist[["match_id", "player_id", "probability"]].rename(
                    columns={"probability": f"p{idx}"}
                )
                merged = merged.merge(
                    part,
                    on=["match_id", "player_id"],
                    how="outer",
                    validate="one_to_one",
                )
            if merged[[f"p{i}" for i in range(len(component_methods))]].isna().any().any():
                raise RuntimeError(f"Calibrated MOM ensemble candidate sets differ for match_id={match_id!r}")
            merged["probability"] = merged[[f"p{i}" for i in range(len(component_methods))]].mean(axis=1)
            total = float(merged["probability"].sum())
            if not np.isfinite(total) or total <= 0:
                raise RuntimeError(f"Calibrated MOM ensemble produced invalid probabilities for match_id={match_id!r}")
            merged["probability"] /= total
            calibration_parts.append(merged[["match_id", "player_id", "kickoff_utc", "probability"]])

        calibration_distribution = pd.concat(calibration_parts, ignore_index=True)
        fitted_temperature, used = _fit_matchwise_temperature(
            calibration,
            calibration_distribution,
            min_matches=min_calibration_matches,
            prior_strength=prior_strength,
        )

        final_models = [
            fit_mom_model(
                train := d[d["match_id"].isin(train_ids)].copy(),
                regularization_c=regularization_c,
                temperature=temperature,
                min_matches=min_train_matches,
                random_state=random_state,
                method=component,
            )
            for component in component_methods
        ]
        oos_parts = []
        for match_id, group in test.groupby("match_id", sort=False):
            kickoff = group["kickoff_utc"].iloc[0]
            prediction_time = kickoff - pd.Timedelta(seconds=1)
            pred_input = group.drop(columns=["is_motm"]).copy()
            component_dists = [
                predict_mom_distribution(
                    model_bundle,
                    pred_input,
                    prediction_time=prediction_time,
                )
                for model_bundle in final_models
            ]
            merged = component_dists[0][["match_id", "player_id", "kickoff_utc", "probability"]].rename(
                columns={"probability": "p0"}
            )
            for idx, component_dist in enumerate(component_dists[1:], start=1):
                part = component_dist[["match_id", "player_id", "probability"]].rename(
                    columns={"probability": f"p{idx}"}
                )
                merged = merged.merge(
                    part,
                    on=["match_id", "player_id"],
                    how="outer",
                    validate="one_to_one",
                )
            probability_cols = [f"p{i}" for i in range(len(component_methods))]
            if merged[probability_cols].isna().any().any():
                raise RuntimeError(f"Calibrated MOM ensemble candidate sets differ for match_id={match_id!r}")
            merged["probability"] = merged[probability_cols].mean(axis=1)
            total = float(merged["probability"].sum())
            if not np.isfinite(total) or total <= 0:
                raise RuntimeError(f"Calibrated MOM ensemble produced invalid OOS probabilities for match_id={match_id!r}")
            merged["probability"] /= total
            oos_parts.append(merged[["match_id", "player_id", "kickoff_utc", "probability"]])

        if not oos_parts:
            continue
        distribution = pd.concat(oos_parts, ignore_index=True)
        distribution = _apply_matchwise_temperature(
            distribution,
            fitted_temperature if used else 1.0,
        )
        evaluated = _evaluate_predictions(test, distribution)
        metrics.append(MOMBlockMetrics(
            block=block_idx,
            train_matches=len(train_ids),
            test_matches=len(test_ids),
            test_rows=len(test),
            **evaluated,
        ))
    if not metrics:
        raise RuntimeError("Calibrated MOM WFO produced no evaluable OOS blocks")
    return pd.DataFrame([asdict(x) for x in metrics])


def split_mom_development_locked(
    block_metrics: pd.DataFrame,
    *,
    locked_blocks: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split already-computed chronological OOS blocks without touching metrics."""
    if block_metrics.empty:
        raise ValueError("MOM block metrics are empty")
    if len(block_metrics) <= int(locked_blocks):
        raise RuntimeError("MOM OOS requires development blocks before locked holdout")
    ordered = block_metrics.sort_values("block", kind="mergesort").reset_index(drop=True)
    development = ordered.iloc[:-int(locked_blocks)].copy()
    locked = ordered.iloc[-int(locked_blocks):].copy()
    if not bool((development["block"].max() < locked["block"].min())):
        raise RuntimeError("MOM development/locked blocks are not strictly separated")
    return development, locked
