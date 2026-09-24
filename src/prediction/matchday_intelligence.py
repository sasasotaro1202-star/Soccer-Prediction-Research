from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

_SIGNAL_GROUPS = {
    "injury": ("matchday_injury_impact_home", "matchday_injury_impact_away"),
    "lineup": ("matchday_lineup_impact_home", "matchday_lineup_impact_away"),
    "weather": ("matchday_weather_penalty_home", "matchday_weather_penalty_away"),
    "rest": ("matchday_rest_diff_hours",),
    "market": ("matchday_market_p_home", "matchday_market_p_draw", "matchday_market_p_away"),
    "market_odds": ("matchday_odds_home", "matchday_odds_draw", "matchday_odds_away"),
}
_REQUIRED_META = ("matchday_available_at_utc", "matchday_pit_verified", "matchday_source")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x -= np.max(x)
    out = np.exp(x)
    return out / out.sum()


def _freshness(age_hours: float) -> float:
    return float(np.clip(math.exp(-max(age_hours, 0.0) / 48.0), 0.25, 1.0))


def _value_present(row: pd.Series, name: str) -> bool:
    if name not in row.index:
        return False
    value = row[name]
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def _present_group(columns: set[str], names: tuple[str, ...]) -> bool:
    return any(name in columns for name in names)


def _present_group_row(row: pd.Series, names: tuple[str, ...]) -> bool:
    return any(_value_present(row, name) for name in names)


def _frame_has_group(frame: pd.DataFrame, names: tuple[str, ...]) -> bool:
    return any(name in frame.columns and frame[name].notna().any() for name in names)


def _validate_group(row: pd.Series, names: tuple[str, ...], group: str) -> np.ndarray:
    present = [_value_present(row, name) for name in names]
    if any(present) and not all(present):
        raise ValueError(f"{group} signal is partially specified")
    values = []
    for name in names:
        value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            raise ValueError(f"{group} signal {name!r} is invalid")
        values.append(float(value))
    result = np.asarray(values, dtype=float)
    if group in {"injury", "lineup", "weather"} and np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{group} signal is outside [0, 1]")
    if group == "rest" and not -72.0 <= float(result[0]) <= 72.0:
        raise ValueError("rest signal is outside [-72, 72] hours")
    if group == "market":
        if np.any((result < 0.0) | (result > 1.0)):
            raise ValueError("market probability is outside [0, 1]")
        if not np.isclose(result.sum(), 1.0, atol=1e-4):
            raise ValueError("market probabilities do not sum to one")
    if group == "market_odds":
        if np.any(result <= 1.0):
            raise ValueError("decimal market odds must be greater than one")
    return result


def _diagnostic(status: str, *, applied=False, age_hours=np.nan, freshness=0.0, signal_count=0, adjustment_l1=0.0, source="", signal_names=(), reason=""):
    return {
        "status": status,
        "applied": bool(applied),
        "age_hours": float(age_hours) if np.isfinite(age_hours) else np.nan,
        "freshness": float(freshness),
        "signal_count": int(signal_count),
        "adjustment_l1": float(adjustment_l1),
        "source": str(source),
        "signal_names": "|".join(signal_names),
        "reason": str(reason),
    }


def apply_matchday_intelligence(
    base_probabilities: np.ndarray,
    fixtures: pd.DataFrame,
    prediction_time: pd.Timestamp,
    *,
    max_abs_shift: float = 0.06,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply only timestamped, PIT-verified, bounded matchday evidence."""
    base = np.asarray(base_probabilities, dtype=float)
    if base.ndim != 2 or base.shape != (len(fixtures), 3) or not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("base_probabilities are invalid")
    base = np.clip(base, 1e-9, 1.0)
    base /= base.sum(axis=1, keepdims=True)
    columns = set(fixtures.columns)
    signal_present = any(_frame_has_group(fixtures, names) for names in _SIGNAL_GROUPS.values())
    if not signal_present:
        return base.copy(), pd.DataFrame([_diagnostic("ABSENT") for _ in range(len(fixtures))])
    missing_meta = [name for name in _REQUIRED_META if name not in columns]
    if missing_meta:
        diagnostics = [_diagnostic("BLOCKED_PIT_METADATA", reason=f"missing matchday metadata: {missing_meta}") for _ in range(len(fixtures))]
        return base.copy(), pd.DataFrame(diagnostics)
    now = pd.Timestamp(prediction_time)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    outputs = base.copy()
    diagnostics = []
    for i, (_, row) in enumerate(fixtures.iterrows()):
        try:
            if not any(_present_group_row(row, names) for names in _SIGNAL_GROUPS.values()):
                diagnostics.append(_diagnostic("ABSENT"))
                continue
            if not _parse_bool(row["matchday_pit_verified"]):
                raise PermissionError("matchday PIT verification is false")
            source = str(row["matchday_source"]).strip()
            if not source:
                raise PermissionError("matchday source is empty")
            available = pd.Timestamp(row["matchday_available_at_utc"])
            if available.tzinfo is None:
                available = available.tz_localize("UTC")
            else:
                available = available.tz_convert("UTC")
            if pd.isna(available) or available > now:
                raise PermissionError("matchday evidence is unavailable at prediction time")
            signal_confidence = 1.0
            if "matchday_signal_confidence" in row.index:
                signal_confidence = float(pd.to_numeric(pd.Series([row["matchday_signal_confidence"]]), errors="coerce").iloc[0])
                if not np.isfinite(signal_confidence) or not 0.0 <= signal_confidence <= 1.0:
                    raise ValueError("matchday_signal_confidence is outside [0, 1]")
            if "starter_status" in row.index:
                starter_status = str(row["starter_status"]).strip().upper()
            else:
                starter_status = "UNKNOWN"
            if "kickoff_utc" in row.index:
                kickoff = pd.Timestamp(row["kickoff_utc"])
                if kickoff.tzinfo is None:
                    kickoff = kickoff.tz_localize("UTC")
                else:
                    kickoff = kickoff.tz_convert("UTC")
                if available > kickoff:
                    raise PermissionError("matchday evidence arrives after kickoff")
            age_hours = max(0.0, float((now - available).total_seconds() / 3600.0))
            freshness = _freshness(age_hours)
            delta = 0.0
            signal_names = []
            signal_count = 0
            for group, names in _SIGNAL_GROUPS.items():
                # Market probability/odds are fused once after the directional
                # context adjustments below; never count the raw odds as an
                # additional directional signal here.
                if group == "market_odds":
                    continue
                if not _present_group_row(row, names):
                    continue
                values = _validate_group(row, names, group)
                if group == "injury":
                    delta += -0.65 * (values[0] - values[1])
                    signal_names.append("injury")
                    signal_count += 2
                elif group == "lineup":
                    if starter_status not in {"ANNOUNCED", "CONFIRMED"}:
                        continue
                    delta += -0.50 * (values[0] - values[1])
                    signal_names.append("lineup")
                    signal_count += 2
                elif group == "weather":
                    delta += -0.30 * (values[0] - values[1])
                    signal_names.append("weather")
                    signal_count += 2
                elif group == "rest":
                    delta += 0.006 * float(np.clip(values[0], -72.0, 72.0))
                    signal_names.append("rest")
                    signal_count += 1
                elif group == "market_odds":
                    implied = 1.0 / np.clip(values, 1.000001, None)
                    implied /= implied.sum()
                    market = implied
                    signal_names.append("market_odds")
                    signal_count += 3
            delta = float(np.clip(delta * signal_confidence, -1.5, 1.5))
            logits = np.log(np.clip(outputs[i], 1e-9, 1.0))
            state_prob = _softmax(logits + freshness * np.array([delta / 2.0, 0.0, -delta / 2.0]))
            state_weight = 0.20 * freshness * signal_confidence
            adjusted = (1.0 - state_weight) * outputs[i] + state_weight * state_prob
            market_reference = None
            if _present_group_row(row, _SIGNAL_GROUPS["market"]):
                market_reference = _validate_group(row, _SIGNAL_GROUPS["market"], "market")
                market_weight = 0.15 * freshness * signal_confidence
                adjusted = (1.0 - market_weight) * adjusted + market_weight * market_reference
                signal_names.append("market")
                signal_count += 3
            elif _present_group_row(row, _SIGNAL_GROUPS["market_odds"]):
                odds = _validate_group(row, _SIGNAL_GROUPS["market_odds"], "market_odds")
                market_reference = 1.0 / np.clip(odds, 1.000001, None)
                market_reference /= market_reference.sum()
                market_weight = 0.15 * freshness * signal_confidence
                adjusted = (1.0 - market_weight) * adjusted + market_weight * market_reference
                signal_names.append("market_odds")
                signal_count += 3

            # Large model-vs-market disagreement is a risk signal, not a
            # directional instruction. Apply only a small uncertainty increase
            # after bounded market fusion so one market snapshot cannot dominate.
            if market_reference is not None:
                disagreement = float(np.abs(outputs[i] - market_reference).sum() / 2.0)
                if disagreement > 0.20:
                    market_temperature = 1.0 + 0.10 * float(
                        np.clip((disagreement - 0.20) / 0.40, 0.0, 1.0)
                    ) * freshness * signal_confidence
                    adjusted = _softmax(
                        np.log(np.clip(adjusted, 1e-9, 1.0)) / market_temperature
                    )
                    signal_names.append("market_uncertainty")
                    signal_count += 1

            # Severe weather is treated as uncertainty, not as an invented
            # home/away edge. A bounded temperature increase flattens the
            # distribution and is only applied when weather evidence is
            # actually present and PIT-safe.
            if "matchday_weather_severity" in row.index and _value_present(row, "matchday_weather_severity"):
                weather_severity = float(
                    pd.to_numeric(pd.Series([row["matchday_weather_severity"]]), errors="coerce").iloc[0]
                )
                if not np.isfinite(weather_severity) or not 0.0 <= weather_severity <= 1.0:
                    raise ValueError("matchday_weather_severity is outside [0, 1]")
                if weather_severity > 0.0:
                    weather_temperature = 1.0 + 0.20 * weather_severity * freshness * signal_confidence
                    adjusted = _softmax(np.log(np.clip(adjusted, 1e-9, 1.0)) / weather_temperature)
                    signal_names.append("weather_uncertainty")
                    signal_count += 1

            adjusted = np.clip(adjusted, 1e-9, 1.0)
            adjusted /= adjusted.sum()
            shift = adjusted - base[i]
            peak = float(np.max(np.abs(shift)))
            if peak > float(max_abs_shift) > 0:
                shift *= float(max_abs_shift) / peak
                adjusted = base[i] + shift
                adjusted = np.clip(adjusted, 1e-9, 1.0)
                adjusted /= adjusted.sum()
            outputs[i] = adjusted
            diagnostics.append(_diagnostic("APPLIED", applied=True, age_hours=age_hours, freshness=freshness, signal_count=signal_count, adjustment_l1=float(np.abs(outputs[i] - base[i]).sum()), source=source, signal_names=tuple(signal_names)))
        except PermissionError as exc:
            outputs[i] = base[i]
            diagnostics.append(_diagnostic("BLOCKED_PIT_METADATA", source=str(row.get("matchday_source", "")), reason=str(exc)))
        except (ValueError, TypeError, OverflowError) as exc:
            outputs[i] = base[i]
            diagnostics.append(_diagnostic("BLOCKED_INVALID_SIGNAL", source=str(row.get("matchday_source", "")), reason=str(exc)))
    return outputs, pd.DataFrame(diagnostics)
