# Soccer Match-State / Event-Hazard Research

Research-only dynamic match-state, event-hazard and scenario layer.

This layer does not change Production, Champion, registry, frozen holdout or production probabilities.

## PIT contract

Required timestamps and fields are: match_id, kickoff_utc, prediction_cutoff_utc, event_time_utc, source_available_at_utc, pit_verified and label_available_at_utc.

The required rule is: source_available_at_utc <= prediction_cutoff_utc. Retrieval time is never historical availability evidence. A label available at or before the prediction cutoff is rejected for training.

## Snapshot schema

Required state/label fields: match_id, kickoff_utc, prediction_cutoff_utc, event_time_utc, source_available_at_utc, pit_verified, home_score, away_score, home_red_cards, away_red_cards, next_event_type, label_available_at_utc, final_home_goals, final_away_goals.

Allowed next-event classes: HOME_GOAL, AWAY_GOAL, HOME_RED, AWAY_RED, NO_EVENT.

## Models

The research candidate contains a stable logistic discrete-time hazard model and a nonlinear HistGradientBoosting challenger. Both require identical chronological OOS evaluation before any adoption decision.

## Scenario propagation

Future paths are propagated as deterministic probability mass over score and red-card states. Default resolution is 5 minutes, with explicit state pruning and recorded discarded mass. The output is P(Home), P(Draw), P(Away) and a top-score distribution.

## Accuracy strategy

The intended sequence is PIT replay -> chronological training -> identical OOS blocks -> next-event LogLoss -> scenario-derived 1X2 LogLoss/Brier/Accuracy -> calibration/ECE -> case-level slices -> robustness -> frozen holdout.

A match can generate multiple snapshots, so snapshot count is not an independent sample-size claim. Match-clustered or match-level evaluation is required for aggregate claims.

## Status

IMPLEMENTED / RESEARCH_CANDIDATE. This is not PERFORMANCE_VERIFIED, PROMOTION_CANDIDATE, ADOPTED or PRODUCTION.

A missing historical in-play dataset produces WARMUP and no performance claim.