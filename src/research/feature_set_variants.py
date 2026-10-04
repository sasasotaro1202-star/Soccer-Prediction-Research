from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Models that can consume every declared feature representation without
# requiring specialist columns (for example Elo differences removed by levels-only variants).
FEATURE_SCREEN_MODELS = (
    "logistic",
    "logistic_select",
    "extra_trees",
    "random_forest",
    "hist_gb",
)


MODEL_META_COLUMNS = {
    "match_id",
    "competition",
    "season",
    "season_start",
    "kickoff_utc",
    "home_team",
    "away_team",
    "prediction_cutoff_at_utc",
    "home_goals",
    "away_goals",
    "target",
    "pit_verified",
    "feature_source_max_available_at_utc",
    "source_available_at_utc",
}

CORE_STRENGTH = {
    "neutral_venue",
    "neutral_venue_known",
    "home_advantage",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_comp_elo",
    "away_comp_elo",
    "comp_elo_diff",
    "home_elo_expected",
    "home_dynamic_elo",
    "away_dynamic_elo",
    "dynamic_elo_diff",
    "dynamic_home_elo_expected",
    "home_comp_elo_shrunk",
    "away_comp_elo_shrunk",
    "comp_elo_shrunk_diff",
    "home_comp_elo_shrunk_expected",
    "elo_gap_abs",
}

CORE_REST = {
    "home_rest_hours",
    "away_rest_hours",
    "rest_diff_hours",
    "strength_rest_interaction",
}

BASIC_STAT_KEYS = (
    "shots",
    "shots_on_target",
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
)

ADVANCED_STAT_KEYS = (
    "xg",
    "possession",
    "offsides",
    "pass_accuracy",
    "goals_ht",
    "xg_ht",
    "shots_inside_box",
    "shots_outside_box",
    "blocked_shots",
    "penalties",
)

SUMMARY_KEYS = {
    "games",
    "gf",
    "ga",
    "points",
    "gd",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "gf_ewma",
    "ga_ewma",
    "gd_ewma",
    "points_ewma",
    "gd_std",
    "home_rate",
    "goal_total_avg",
    "clean_sheet_rate",
    "failed_to_score_rate",
}

MOMENTUM_KEYS = {
    "points_ewma",
    "gd_ewma",
    "gf_ewma",
    "ga_ewma",
    "win_rate",
}


def _is_numeric_feature(column: str) -> bool:
    return column not in MODEL_META_COLUMNS and not column.startswith("baseline_")


def _window(column: str) -> int | None:
    parts = column.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        value = int(parts[1])
    except ValueError:
        return None
    return value if value in {3, 5, 10, 20} else None


def _is_summary_feature(column: str) -> bool:
    for prefix in ("home_", "away_"):
        body = column[len(prefix):] if column.startswith(prefix) else column
        if any(body == f"{k}_{w}" for k in SUMMARY_KEYS for w in (3, 5, 10, 20)):
            return True
        if any(body == f"{k}_{w}" for k in BASIC_STAT_KEYS for w in (3, 5, 10, 20)):
            return True
        if any(body == f"{k}_{w}" for k in ADVANCED_STAT_KEYS for w in (3, 5, 10, 20)):
            return True
    return False


def _contains_stat(column: str, keys: Iterable[str]) -> bool:
    stem = column
    for prefix in ("home_", "away_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return any(stem.startswith(f"{k}_") for k in keys)


def _family(column: str) -> str:
    if column in CORE_STRENGTH:
        return "strength"
    if column in CORE_REST:
        return "rest"
    if column.startswith("h2h_"):
        return "h2h"
    if "momentum_3v10" in column:
        return "momentum"
    if column in {"attack_defense_matchup_diff_5", "attack_defense_matchup_sum_5", "draw_tension_10"}:
        return "interaction"
    if column in {"home_history_support_n", "away_history_support_n"}:
        return "history_support"
    if any(column.endswith(f"_diff_{w}") for w in (3, 5, 10, 20)):
        return "derived_difference"
    if _contains_stat(column, ADVANCED_STAT_KEYS):
        return "advanced_stats"
    if _contains_stat(column, BASIC_STAT_KEYS):
        return "basic_stats"
    if _is_summary_feature(column):
        return "form"
    return "other"


def available_feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [c for c in numeric if _is_numeric_feature(c)]


def _keep_window(column: str, windows: set[int]) -> bool:
    value = _window(column)
    return value in windows if value is not None else True


def _keep_representation(column: str, representation: str) -> bool:
    if representation == "all":
        return True
    if representation == "difference":
        if column.endswith("_diff_3") or column.endswith("_diff_5") or column.endswith("_diff_10") or column.endswith("_diff_20"):
            return True
        return column in {
            "elo_diff",
            "comp_elo_diff",
            "dynamic_elo_diff",
            "comp_elo_shrunk_diff",
            "rest_diff_hours",
            "elo_gap_abs",
            "attack_defense_matchup_diff_5",
            "draw_tension_10",
        }
    if representation == "levels":
        return not (
            column.endswith("_diff_3")
            or column.endswith("_diff_5")
            or column.endswith("_diff_10")
            or column.endswith("_diff_20")
        ) and column not in {"elo_diff", "comp_elo_diff", "dynamic_elo_diff", "comp_elo_shrunk_diff", "rest_diff_hours"}
    raise ValueError(f"Unknown representation: {representation}")


def _is_offensive(column: str) -> bool:
    tokens = (
        "_gf_", "_goal_total_avg_", "_xg_", "_shots_", "_shots_on_target_",
        "_goals_ht_", "_xg_ht_", "_failed_to_score_rate_",
    )
    return any(token in column for token in tokens)


def _is_defensive(column: str) -> bool:
    tokens = (
        "_ga_", "_gd_", "_clean_sheet_rate_", "_blocked_shots_",
        "_failed_to_score_rate_",
    )
    return any(token in column for token in tokens)


def _is_process(column: str) -> bool:
    tokens = (
        "_xg_", "_possession_", "_pass_accuracy_", "_shots_inside_box_",
        "_shots_outside_box_", "_blocked_shots_", "_offsides_",
    )
    return any(token in column for token in tokens)


def _is_discipline(column: str) -> bool:
    tokens = ("_fouls_", "_yellow_cards_", "_red_cards_", "_penalties_")
    return any(token in column for token in tokens)


def _is_goal_environment(column: str) -> bool:
    tokens = (
        "_goal_total_avg_", "_gf_", "_ga_", "_xg_", "_goals_ht_", "_xg_ht_",
        "_clean_sheet_rate_", "_failed_to_score_rate_",
    )
    return any(token in column for token in tokens)


def _is_volatility(column: str) -> bool:
    return "_std_" in column or "volatility" in column


def select_feature_set(frame: pd.DataFrame, variant: str) -> tuple[list[str], dict]:
    cols = available_feature_columns(frame)

    def allow(c: str) -> bool:
        fam = _family(c)
        w = _window(c)
        if variant == "strength_only":
            return fam == "strength"
        if variant == "strength_rest":
            return fam in {"strength", "rest"}
        if variant == "strength_rest_history":
            return fam in {"strength", "rest", "history_support"}
        if variant == "form_3_only":
            return fam in {"strength", "rest"} or (fam == "form" and w == 3)
        if variant == "form_10_only":
            return fam in {"strength", "rest"} or (fam == "form" and w == 10)
        if variant == "form_20_only":
            return fam in {"strength", "rest"} or (fam == "form" and w == 20)
        if variant == "form_3_5":
            return fam in {"strength", "rest"} or (fam == "form" and w in {3, 5})
        if variant == "form_5_10":
            return fam in {"strength", "rest"} or (fam == "form" and w in {5, 10})
        if variant == "form_3_5_10":
            return fam in {"strength", "rest"} or (fam == "form" and w in {3, 5, 10})
        if variant == "core_form_5":
            return fam in {"strength", "rest"} or (fam == "form" and w == 5)
        if variant == "form_3_10":
            return fam in {"strength", "rest"} or (fam == "form" and w in {3, 10})
        if variant == "form_all":
            return fam in {"strength", "rest", "form"}
        if variant == "basic_stats_5":
            return fam in {"strength", "rest", "form"} or (fam == "basic_stats" and w == 5)
        if variant == "basic_3_10":
            return fam in {"strength", "rest", "form"} or (fam == "basic_stats" and w in {3, 10})
        if variant == "advanced_stats_5":
            return fam in {"strength", "rest", "form", "basic_stats"} or (fam == "advanced_stats" and w == 5)
        if variant == "advanced_3_10":
            return fam in {"strength", "rest", "form", "basic_stats"} or (fam == "advanced_stats" and w in {3, 10})
        if variant == "xg_possession_dense":
            if fam in {"strength", "rest", "form", "basic_stats"}:
                return True
            if fam == "advanced_stats":
                return any(token in c for token in (
                    "_xg_", "_possession_", "_pass_accuracy_", "_shots_inside_box_",
                ))
            return False
        if variant == "advanced_all":
            return fam in {"strength", "rest", "form", "basic_stats", "advanced_stats"}
        if variant == "h2h_momentum":
            return fam in {"strength", "rest", "form", "h2h", "momentum"}
        if variant == "interactions":
            return fam in {"strength", "rest", "form", "h2h", "momentum", "interaction"}
        if variant == "difference_heavy":
            return _keep_representation(c, "difference") and fam not in {"other", "history_support"}
        if variant == "ewma_heavy":
            if fam in {"strength", "rest", "h2h", "momentum", "interaction"}:
                return True
            return ("_ewma_" in c or c.endswith("_ewma_3") or c.endswith("_ewma_5")
                    or c.endswith("_ewma_10") or c.endswith("_ewma_20"))
        if variant == "compact":
            if fam == "strength":
                return True
            if fam == "rest":
                return True
            if fam == "form" and w in {5, 10}:
                return any(token in c for token in (
                    "_points_", "_gd_", "_win_rate_", "_gf_", "_ga_",
                    "_clean_sheet_rate_", "_failed_to_score_rate_",
                ))
            if fam == "advanced_stats" and w == 10:
                return any(token in c for token in ("_xg_", "_shots_on_target_", "_possession_"))
            return fam in {"h2h", "momentum", "interaction"}
        if variant == "offense_lean":
            if fam in {"strength", "rest"}:
                return True
            return _is_offensive(c)
        if variant == "defense_lean":
            if fam in {"strength", "rest"}:
                return True
            return _is_defensive(c)
        if variant == "process_lean":
            if fam in {"strength", "rest"}:
                return True
            return _is_process(c)
        if variant == "discipline_lean":
            if fam in {"strength", "rest"}:
                return True
            return _is_discipline(c)
        if variant == "goal_environment":
            if fam in {"strength", "rest"}:
                return True
            return _is_goal_environment(c)
        if variant == "volatility_lean":
            if fam in {"strength", "rest"}:
                return True
            return _is_volatility(c)
        if variant == "short_form_3_5":
            if fam in {"strength", "rest"}:
                return True
            return fam in {"form", "basic_stats", "advanced_stats", "derived_difference"} and w in {3, 5}
        if variant == "long_form_10_20":
            if fam in {"strength", "rest"}:
                return True
            return fam in {"form", "basic_stats", "advanced_stats", "derived_difference"} and w in {10, 20}
        if variant == "ewma_difference":
            return _keep_representation(c, "difference") and ("ewma" in c or fam in {"strength", "rest"})
        if variant == "levels_form":
            if fam in {"strength", "rest", "form", "basic_stats", "advanced_stats"}:
                return _keep_representation(c, "levels")
            return False
        if variant == "levels_only":
            return _keep_representation(c, "levels")
        raise ValueError(f"Unknown feature-set variant: {variant}")

    selected = [c for c in cols if allow(c)]
    if not selected:
        raise RuntimeError(f"Feature variant {variant!r} produced zero columns")

    payload = {"variant": variant, "ordered_feature_cols": selected}
    feature_set_id = "fs-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    meta = {
        "feature_set_id": feature_set_id,
        "variant": variant,
        "feature_count": len(selected),
        "families": sorted({_family(c) for c in selected}),
    }
    return selected, meta


VARIANT_ORDER = (
    "strength_only",
    "strength_rest",
    "strength_rest_history",
    "form_3_only",
    "core_form_5",
    "form_10_only",
    "form_20_only",
    "form_3_5",
    "form_3_10",
    "form_5_10",
    "form_3_5_10",
    "form_all",
    "short_form_3_5",
    "long_form_10_20",
    "basic_stats_5",
    "basic_3_10",
    "advanced_stats_5",
    "advanced_3_10",
    "xg_possession_dense",
    "offense_lean",
    "defense_lean",
    "process_lean",
    "discipline_lean",
    "goal_environment",
    "volatility_lean",
    "advanced_all",
    "h2h_momentum",
    "interactions",
    "difference_heavy",
    "ewma_heavy",
    "ewma_difference",
    "compact",
    "levels_only",
    "levels_form",
)


def variant_catalog() -> list[dict]:
    return [{"variant": name, "order": i} for i, name in enumerate(VARIANT_ORDER)]


def aggregate_fold_metrics(wf: pd.DataFrame, *, locked_blocks: int = 2) -> dict:
    if wf.empty:
        raise RuntimeError("Cannot aggregate empty WFO result")
    if "n" not in wf.columns:
        raise RuntimeError("WFO result missing n")
    weights = pd.to_numeric(wf["n"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError("WFO result contains invalid fold weights")
    development = wf.iloc[:-locked_blocks] if len(wf) > locked_blocks else wf.iloc[0:0]
    locked = wf.tail(locked_blocks)
    def weighted(frame: pd.DataFrame, key: str) -> float | None:
        if frame.empty or key not in frame.columns:
            return None
        v = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
        w = pd.to_numeric(frame["n"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).all() or not np.isfinite(w).all() or w.sum() <= 0:
            return None
        return float(np.average(v, weights=w))
    return {
        "blocks": int(len(wf)),
        "development_blocks": int(len(development)),
        "locked_blocks": int(len(locked)),
        "development_n": int(pd.to_numeric(development["n"], errors="coerce").sum()) if not development.empty else 0,
        "locked_n": int(pd.to_numeric(locked["n"], errors="coerce").sum()) if not locked.empty else 0,
        "development_logloss": weighted(development, "logloss"),
        "development_brier": weighted(development, "brier"),
        "development_accuracy": weighted(development, "accuracy"),
        "development_ece": weighted(development, "ece"),
        "locked_logloss": weighted(locked, "logloss"),
        "locked_brier": weighted(locked, "brier"),
        "locked_accuracy": weighted(locked, "accuracy"),
        "locked_ece": weighted(locked, "ece"),
    }


def select_development_winner(results: pd.DataFrame, baseline_variant: str) -> dict:
    required = {
        "variant", "feature_set_id", "development_logloss", "development_brier",
        "development_accuracy", "development_ece", "oos_window_signature",
    }
    missing = required - set(results.columns)
    if missing:
        raise RuntimeError(f"Feature-set comparison missing columns: {sorted(missing)}")

    if results["oos_window_signature"].nunique(dropna=True) != 1:
        raise RuntimeError("Feature variants were evaluated on different OOS window signatures")

    baseline = results.loc[results["variant"] == baseline_variant]
    if len(baseline) != 1:
        raise RuntimeError(f"Baseline variant {baseline_variant!r} must exist exactly once")

    candidates = results.copy()
    candidates = candidates.sort_values(
        ["development_logloss", "development_brier", "development_ece", "variant"],
        kind="mergesort",
    )
    winner = candidates.iloc[0].to_dict()
    base_ll = float(baseline.iloc[0]["development_logloss"])
    winner_ll = float(winner["development_logloss"])
    winner["development_relative_logloss_improvement"] = (
        float((base_ll - winner_ll) / base_ll) if base_ll > 0 else None
    )
    winner["baseline_variant"] = baseline_variant
    winner["selection_basis"] = "development_chronological_oos_only"
    winner["locked_oos_used_for_selection"] = False
    return winner


def run_feature_set_research(
    input_path: str,
    output_dir: str,
    variants: list[str] | None = None,
    *,
    min_train: int = 2000,
    oos_block: int = 4000,
    random_state: int = 42,
    model_names: list[str] | None = None,
) -> dict:
    """Compare predeclared feature-set variants on identical chronological WFO windows.

    Feature-set selection is performed on development OOS blocks only. The final two
    OOS blocks are retained as locked verification and are never consulted when
    choosing the winner.
    """
    from src.evaluation.walk_forward import run_walk_forward

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path)
    required = {
        "match_id",
        "kickoff_utc",
        "target",
        "pit_verified",
        "prediction_cutoff_at_utc",
        "feature_source_max_available_at_utc",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Feature research input missing required PIT columns: {missing}")
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
    frame["prediction_cutoff_at_utc"] = pd.to_datetime(
        frame["prediction_cutoff_at_utc"], utc=True, errors="coerce"
    )
    frame["feature_source_max_available_at_utc"] = pd.to_datetime(
        frame["feature_source_max_available_at_utc"], utc=True, errors="coerce"
    )
    if frame[["kickoff_utc", "prediction_cutoff_at_utc", "feature_source_max_available_at_utc"]].isna().any().any():
        raise RuntimeError("Feature research input contains invalid PIT temporal fields")
    pit = frame["pit_verified"].astype(str).str.strip().str.lower()
    if not pit.isin({"true", "false", "1", "0", "yes", "no"}).all():
        raise RuntimeError("Feature research input contains ambiguous pit_verified values")
    frame["pit_verified"] = pit.isin({"true", "1", "yes"})
    verified = frame.loc[frame["pit_verified"]]
    if not verified.empty:
        availability_ok = (
            verified["feature_source_max_available_at_utc"] <= verified["prediction_cutoff_at_utc"]
        )
        cutoff_ok = verified["prediction_cutoff_at_utc"] <= verified["kickoff_utc"]
        if not bool((availability_ok & cutoff_ok).all()):
            bad = int((availability_ok & cutoff_ok).eq(False).sum())
            raise RuntimeError(
                f"Feature research PIT provenance validation failed for {bad} verified rows"
            )
    frame = frame.sort_values(["kickoff_utc", "match_id"], kind="mergesort").reset_index(drop=True)

    selected_variants = list(variants or (
        "strength_only",
        "strength_rest",
        "strength_rest_history",
        "form_3_only",
        "core_form_5",
        "form_10_only",
        "form_20_only",
        "form_3_5",
        "form_3_10",
        "form_5_10",
        "form_3_5_10",
        "form_all",
        "short_form_3_5",
        "long_form_10_20",
        "basic_stats_5",
        "basic_3_10",
        "advanced_stats_5",
        "advanced_3_10",
        "xg_possession_dense",
        "offense_lean",
        "defense_lean",
        "process_lean",
        "discipline_lean",
        "goal_environment",
        "volatility_lean",
        "advanced_all",
        "h2h_momentum",
        "interactions",
        "difference_heavy",
        "ewma_difference",
        "compact",
        "levels_only",
        "levels_form",
    ))
    valid_names = {x["variant"] for x in variant_catalog()}
    unknown = [x for x in selected_variants if x not in valid_names]
    if unknown:
        raise RuntimeError(f"Unknown feature-set variants: {unknown}")

    # Keep the feature-set screen model set compatible across every declared
    # representation. Elo-specialist candidates are evaluated later in the
    # dedicated model-ecology stage.
    models = list(model_names or FEATURE_SCREEN_MODELS)
    manifest_path = out / "feature_variant_catalog.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variants": variant_catalog(),
                "selected_variants": selected_variants,
                "models": models,
                "min_train": int(min_train),
                "oos_block": int(oos_block),
                "locked_blocks": 2,
                "selection_basis": "development_chronological_oos_only",
                "locked_oos_used_for_selection": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = []
    detailed: dict[str, pd.DataFrame] = {}
    signatures: set[str] = set()
    for variant in selected_variants:
        feature_cols, meta = select_feature_set(frame, variant)
        wf, _selection = run_walk_forward(
            frame,
            feature_cols,
            min_train=min_train,
            oos_block=oos_block,
            random_state=random_state,
            candidate_names=models,
        )
        agg = aggregate_fold_metrics(wf, locked_blocks=2)
        signature = str(wf["oos_window_signature"].iloc[0])
        signatures.add(signature)
        rows.append({
            **meta,
            **agg,
            "oos_window_signature": signature,
        })
        detailed[variant] = wf

    if len(signatures) != 1:
        raise RuntimeError("Feature variants were not evaluated on the same OOS window signature")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "feature_set_comparison.csv", index=False)
    winner = select_development_winner(comparison, "strength_only")
    (out / "feature_set_winner.json").write_text(
        json.dumps(winner, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    baseline = comparison.loc[comparison["variant"] == "strength_only"].iloc[0]
    candidate_rows = []
    for _, row in comparison.iterrows():
        candidate_rows.append({
            "variant": row["variant"],
            "development_logloss_delta_vs_strength_only": float(row["development_logloss"] - baseline["development_logloss"]),
            "development_relative_logloss_improvement_vs_strength_only": (
                float((baseline["development_logloss"] - row["development_logloss"]) / baseline["development_logloss"])
                if float(baseline["development_logloss"]) > 0 else None
            ),
            "locked_logloss_delta_vs_strength_only": (
                float(row["locked_logloss"] - baseline["locked_logloss"])
                if pd.notna(row["locked_logloss"]) and pd.notna(baseline["locked_logloss"]) else None
            ),
            "locked_brier_delta_vs_strength_only": (
                float(row["locked_brier"] - baseline["locked_brier"])
                if pd.notna(row["locked_brier"]) and pd.notna(baseline["locked_brier"]) else None
            ),
            "locked_accuracy_delta_vs_strength_only": (
                float(row["locked_accuracy"] - baseline["locked_accuracy"])
                if pd.notna(row["locked_accuracy"]) and pd.notna(baseline["locked_accuracy"]) else None
            ),
        })
    pd.DataFrame(candidate_rows).to_csv(out / "feature_set_deltas.csv", index=False)

    winner_variant = str(winner["variant"])
    winner_features, _ = select_feature_set(frame, winner_variant)
    case_path = out / "winner_case_diagnostics.csv"
    winner_wf, _ = run_walk_forward(
        frame,
        winner_features,
        min_train=min_train,
        oos_block=oos_block,
        random_state=random_state,
        case_output_path=str(case_path),
        candidate_names=models,
    )
    # Keep the diagnostic rerun explicitly linked to the predeclared winner identity.
    winner_wf.to_csv(out / "winner_oos_metrics.csv", index=False)

    manifest = {
        "schema_version": 1,
        "status": "RESEARCH_EXECUTED",
        "feature_variants_tested": selected_variants,
        "models_per_variant": models,
        "variant_count": len(selected_variants),
        "oos_window_signature": next(iter(signatures)),
        "winner": winner,
        "winner_case_diagnostics": "winner_case_diagnostics.csv",
        "safety_contract": {
            "research_only": True,
            "production_changed": False,
            "production_registry_changed": False,
            "frozen_holdout_used_for_selection": False,
            "locked_oos_used_for_selection": False,
            "pit_verified_input_required": True,
            "same_oos_window_required": True,
        },
    }
    (out / "feature_set_research_status.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Soccer feature-set OOS research.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-train", type=int, default=2000)
    parser.add_argument("--oos-block", type=int, default=4000)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    result = run_feature_set_research(
        args.input,
        args.output_dir,
        args.variants,
        min_train=args.min_train,
        oos_block=args.oos_block,
        model_names=args.models,
    )
    print(json.dumps(result, ensure_ascii=False, default=str))
