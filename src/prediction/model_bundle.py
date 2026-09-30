        if not isinstance(routing, dict):
            raise RuntimeError("Production model bundle schema 3 requires routing_policy")
        if routing.get("schema_version") != 1 or routing.get("type") != "hierarchical_validation_context":
            raise RuntimeError("Production routing policy schema/type is invalid")
        fallback_weights = routing.get("fallback_weights")
        if not isinstance(fallback_weights, dict) or set(fallback_weights) != set(weights):
            raise RuntimeError("Production routing fallback/model sets mismatch")
        routing_weights = routing.get("context_weights")
        if not isinstance(routing_weights, dict):
            raise RuntimeError("Production routing context_weights are invalid")
        for route_key, mapping in routing_weights.items():
            if not isinstance(mapping, dict) or set(mapping) != set(weights):
                raise RuntimeError(f"Production routing context model set is invalid for {route_key!r}")
            values = np.asarray(list(mapping.values()), dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-8):
                raise RuntimeError(f"Production routing context weights are invalid for {route_key!r}")
        risk_temperature_modifiers = routing.get("risk_temperature_modifiers", {})
        if not isinstance(risk_temperature_modifiers, dict):
            raise RuntimeError("Production risk_temperature_modifiers are invalid")
        for bucket, value in risk_temperature_modifiers.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise RuntimeError(f"Production risk temperature modifier is invalid for {bucket!r}")
            if not np.isfinite(value) or not 0.85 <= value <= 1.20:
                raise RuntimeError(f"Production risk temperature modifier is out of bounds for {bucket!r}")
        routing_temps = routing.get("contextual_temperatures", {})
        if not isinstance(routing_temps, dict):
            raise RuntimeError("Production contextual temperatures are invalid")
        for route_key, value in routing_temps.items():
            value = float(value)
            if not np.isfinite(value) or not 0.70 <= value <= 1.60:
                raise RuntimeError(f"Production contextual temperature is invalid for {route_key!r}")
        dynamic = routing.get("dynamic_routing")
        if dynamic is not None:
            if not isinstance(dynamic, dict):
                raise RuntimeError("Production dynamic_routing is invalid")
            if dynamic.get("schema_version") != 1 or dynamic.get("type") != "drift_uncertainty_router":
                raise RuntimeError("Production dynamic_routing schema/type is invalid")
            if dynamic.get("enabled") is not True:
                raise RuntimeError("Production dynamic_routing must be enabled for schema 3")
            try:
                drift_strength = float(dynamic.get("drift_strength"))
                uncertainty_strength = float(dynamic.get("uncertainty_strength"))
                min_trust = float(dynamic.get("min_specialist_trust"))
            except (TypeError, ValueError):
                raise RuntimeError("Production dynamic_routing parameters are invalid")
            if not np.isfinite(drift_strength) or not 0.0 <= drift_strength <= 3.0:
                raise RuntimeError("Production dynamic drift_strength is invalid")
            if not np.isfinite(uncertainty_strength) or not 0.0 <= uncertainty_strength <= 3.0:
                raise RuntimeError("Production dynamic uncertainty_strength is invalid")
            if not np.isfinite(min_trust) or not 0.05 <= min_trust <= 0.95:
                raise RuntimeError("Production dynamic min_specialist_trust is invalid")
            reference = dynamic.get("reference")
            if not isinstance(reference, dict) or not isinstance(reference.get("features"), dict) or not reference.get("features"):
                raise RuntimeError("Production dynamic routing reference is missing")
    matchday_policy = bundle.get("matchday_policy", {"schema_version": 1, "status": "SHADOW_ONLY"})
    if not isinstance(matchday_policy, dict):
        raise RuntimeError("Production matchday_policy must be an object")
    if matchday_policy.get("schema_version") != 1:
        raise RuntimeError("Production matchday_policy schema_version must be 1")
    status = str(matchday_policy.get("status", "SHADOW_ONLY")).upper()
    if status not in {"PASS", "SHADOW_ONLY"}:
        raise RuntimeError(f"Unsupported production matchday_policy status: {status}")
    if status == "PASS":
        evidence = matchday_policy.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("oos_verified") is not True:
            raise RuntimeError("Production matchday policy PASS requires explicit OOS evidence")
        if evidence.get("locked_holdout_untouched") is not True:
            raise RuntimeError("Production matchday policy PASS requires untouched locked holdout evidence")

    score_method = str(bundle.get("score_method", "primary"))
    if score_method not in {"primary", "neutral_aware", "recency", "time_decay", "dixon_coles", "negative_binomial", "xg"}:
        raise RuntimeError(f"Unsupported production score method: {score_method}")
    score_models_by_competition = bundle.get("score_models_by_competition", {})
    score_method_by_competition = bundle.get("score_method_by_competition", {})
    if not isinstance(score_models_by_competition, dict) or not isinstance(score_method_by_competition, dict):
        raise RuntimeError("Competition-specific Score model metadata must be objects")
    allowed_score_methods = {"primary", "neutral_aware", "recency", "time_decay", "dixon_coles", "negative_binomial", "xg"}
    for competition, specialist in score_models_by_competition.items():
        if not isinstance(specialist, dict):
            raise RuntimeError(f"Competition Score specialist is invalid for {competition!r}")
        method = str(specialist.get("method", ""))
        if method.startswith("xg_"):
            resolved = "xg"
        elif method.startswith("dixon_coles_"):
            resolved = "dixon_coles"
        elif method.startswith("negative_binomial_"):
            resolved = "negative_binomial"
        elif method.startswith("neutral_aware_"):
            resolved = "neutral_aware"
        elif method.startswith("pit_recency_"):
            resolved = "recency"
        elif method.startswith("pit_time_decay_"):
            resolved = "time_decay"
        elif method.startswith("pit_smoothed_"):
            resolved = "primary"
        else:
            raise RuntimeError(f"Unknown competition Score specialist method for {competition!r}: {method!r}")
        if resolved not in allowed_score_methods:
            raise RuntimeError(f"Unsupported competition Score specialist method for {competition!r}")
        meta = score_method_by_competition.get(str(competition))
        if not isinstance(meta, dict) or str(meta.get("status", "")).upper() != "COMPETITION_SPECIALIST":
            raise RuntimeError(f"Competition Score specialist lacks COMPETITION_SPECIALIST metadata for {competition!r}")
        if str(meta.get("method")) != resolved:
            raise RuntimeError(f"Competition Score method metadata mismatch for {competition!r}")