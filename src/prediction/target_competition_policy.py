from __future__ import annotations

"""Explicit target x competition prediction-policy metadata.

This module does not change probabilities. It makes the already-validated
prediction stack inspectable at the same granularity as the requested targets:
competition x {1X2, Score, O/U, BTTS, MOM}.
"""

from typing import Any

import pandas as pd

from src.evaluation.walk_forward import _lookup_context_weights


TARGETS = ("1X2", "Score", "O/U", "BTTS", "MOM")


def build_target_policy(
    row: pd.Series,
    bundle: dict[str, Any],
) -> dict[str, str]:
    competition = str(row.get("competition", "")).strip().upper() or "__MISSING__"
    routing = bundle.get("routing_policy")
    route = "GLOBAL"
    route_scope = "GLOBAL_FALLBACK"
    if isinstance(routing, dict):
        fallback = routing.get("fallback_weights") or bundle.get("weights") or {}
        context_weights = routing.get("context_weights") or {}
        try:
            _, route = _lookup_context_weights(
                row,
                context_weights,
                {str(k): float(v) for k, v in fallback.items()},
            )
            route_scope = (
                "COMPETITION_HIERARCHICAL"
                if route != "GLOBAL" and competition in route
                else "GLOBAL_FALLBACK"
            )
        except Exception:
            route = "GLOBAL"
            route_scope = "GLOBAL_FALLBACK"

    score_method = str(bundle.get("score_method", "primary"))
    score_model = bundle.get("score_model") or {}
    comp_rates = score_model.get("competition_rates") if isinstance(score_model, dict) else {}
    score_rate_scope = (
        "COMPETITION_RATE"
        if isinstance(comp_rates, dict) and competition in comp_rates
        else "GLOBAL_RATE"
    )

    score_policy = f"{score_method}|{score_rate_scope}"
    return {
        "competition": competition,
        "policy_version": "target-x-competition-v1",
        "competition_routing_scope": route_scope,
        "competition_route": route,
        "target_1x2_policy": f"1X2|{route_scope}|dynamic={bool(isinstance(routing, dict) and isinstance(routing.get('dynamic_routing'), dict) and routing.get('dynamic_routing', {}).get('enabled') is True)}",
        "target_score_policy": f"Score|{score_policy}",
        "target_ou_policy": f"O/U|DERIVED_SCORE_DISTRIBUTION|{score_policy}",
        "target_btts_policy": f"BTTS|DERIVED_SCORE_DISTRIBUTION|{score_policy}",
        "target_mom_policy": "MOM|UPSTREAM_PLAYER_MODEL_REQUIRED",
    }


def annotate_predictions(fixtures: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures.copy()
    rows = [build_target_policy(row, bundle) for _, row in fixtures.iterrows()]
    policies = pd.DataFrame(rows, index=fixtures.index)
    out = fixtures.copy()
    for col in policies.columns:
        out[f"policy_{col}"] = policies[col]
    return out
