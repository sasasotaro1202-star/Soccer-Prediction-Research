# Soccer-Prediction-Research — Feature Manifest

## Version
soccer-feature-contract-v1

## Purpose
This manifest separates the source-generated candidate feature pool from verified Production usage. Feature existence in code does not imply Production consumption.

## Feature states

- ACTIVE — explicitly consumed by a currently verified Production bundle.
- CONDITIONAL — eligible only when an explicit PIT-safe context/cutoff contract passes.
- OBSERVATION_ONLY — collected or preserved for diagnostics/evidence but not fed into Production probabilities.
- RESEARCH_CANDIDATE — eligible for research/OOS experiments but not Production-authorized.

On current main, the exact Production feature set is UNVERIFIABLE because a committed `models/current` bundle is not present. This document therefore describes the candidate pool and selection contract, not a Production claim.

## Current source-level candidate pool

The current `src/features/soccer_features.py` implementation has a nominal full-field matrix of approximately 540 model-input candidate columns when all optional basic/advanced statistics are present. The actual count depends on upstream field coverage.

### Strength / venue / Elo — RESEARCH_CANDIDATE until bundle verification

- neutral_venue
- neutral_venue_known
- home_advantage
- home_elo / away_elo / elo_diff
- home_comp_elo / away_comp_elo / comp_elo_diff
- home_elo_expected
- home_dynamic_elo / away_dynamic_elo / dynamic_elo_diff
- dynamic_home_elo_expected
- home_comp_elo_shrunk / away_comp_elo_shrunk / comp_elo_shrunk_diff
- home_comp_elo_shrunk_expected
- elo_gap_abs

### Schedule / rest — RESEARCH_CANDIDATE until bundle verification

- home_rest_hours
- away_rest_hours
- rest_diff_hours
- strength_rest_interaction

### Recent team form — RESEARCH_CANDIDATE until bundle verification

For windows 3, 5, 10 and 20, each team has 49 summary outputs:

- games
- gf / ga / points / gd
- win_rate / draw_rate / loss_rate
- gf_ewma / ga_ewma / gd_ewma / points_ewma
- gd_std
- home_rate
- goal_total_avg
- clean_sheet_rate
- failed_to_score_rate
- shots_avg / shots_ewma
- shots_on_target_avg / shots_on_target_ewma
- corners_avg / corners_ewma
- fouls_avg / fouls_ewma
- yellow_cards_avg / yellow_cards_ewma
- red_cards_avg / red_cards_ewma
- xg_avg / xg_ewma
- possession_avg / possession_ewma
- offsides_avg / offsides_ewma
- pass_accuracy_avg / pass_accuracy_ewma
- goals_ht_avg / goals_ht_ewma
- xg_ht_avg / xg_ht_ewma
- shots_inside_box_avg / shots_inside_box_ewma
- shots_outside_box_avg / shots_outside_box_ewma
- blocked_shots_avg / blocked_shots_ewma
- penalties_avg / penalties_ewma

Both home_ and away_ representations are produced. Explicit comparison/difference fields are also generated. Under the full-field schema this is 123 columns per window and 492 across all four windows.

### H2H — RESEARCH_CANDIDATE

- h2h_games_5
- h2h_home_win_rate_5
- h2h_draw_rate_5
- h2h_away_win_rate_5
- h2h_points_edge_5

H2H must remain chronologically replayed and PIT-safe.

### Momentum — RESEARCH_CANDIDATE

For points_ewma, gd_ewma, gf_ewma, ga_ewma and win_rate:
- home 3v10 momentum
- away 3v10 momentum
- home-minus-away momentum

### Matchup / interaction — RESEARCH_CANDIDATE

- attack_defense_matchup_diff_5
- attack_defense_matchup_sum_5
- draw_tension_10
- strength_rest_interaction

Interactions are independent candidate groups and must be ablated rather than assumed useful.

### History support — RESEARCH_CANDIDATE

- home_history_support_n
- away_history_support_n

These indicate prior-history support and are not observed team strength.

### PIT / provenance metadata — OBSERVATION_ONLY

- pit_verified
- feature_source_max_available_at_utc

These fields are validation metadata and are excluded from model inputs.

## Matchday context

These channels are not assumed to be core Production 1X2 inputs:

- lineup / starters / formations
- injuries
- suspensions
- market context
- weather
- late official updates

Default state is CONDITIONAL / RESEARCH_CANDIDATE. Historical OOS consumption requires independently proven availability at or before the prediction cutoff. Retrieval time is never a substitute for historical availability.

## Source-visible model consumers

Simple baseline consumers:
- EloLogisticClassifier: `elo_diff`, `comp_elo_diff`, `home_elo_expected`
- DynamicEloLogisticClassifier: `dynamic_elo_diff`, `dynamic_home_elo_expected`, `elo_diff`, `home_elo_expected`
- HierarchicalEloLogisticClassifier: `comp_elo_shrunk_diff`, `home_comp_elo_shrunk_expected`, `dynamic_elo_diff`

Richer generic candidates include:
- logistic
- logistic_select
- ExtraTrees
- RandomForest
- HistGradientBoosting
- robust HistGradientBoosting variants
- recency/quantile logistic challengers

Generic models consume a locked `feature_cols` list, so source presence does not prove Production use.

## Feature-set research patterns

- strength-only baseline
- strength + recent form
- strength + schedule/rest
- strength + basic event statistics
- strength + advanced statistics
- H2H add/remove
- interaction add/remove
- 3/5/10/20 window combinations
- average versus EWMA
- volatility/trend inclusion
- home/away levels versus difference-only representation
- group deletion / leave-one-family-out
- source deletion / source-group ablation
- missingness-aware variants
- model-specific selectors
- competition specialist feature sets
- regime/data-quality conditional feature sets
- PIT-safe matchday context variants

No pattern is Production-authorized by this manifest alone.

## Nested/prequential selection

For every outer chronological OOS block:
1. freeze the outer OOS boundary;
2. construct candidate feature sets only from prior information;
3. if feature selection is data-driven, tune it inside the training side using inner chronological WFO/validation as required;
4. lock feature_set_id + ordered feature_cols + model + training window + calibration + routing;
5. score the outer OOS block exactly once;
6. keep locked/frozen holdout untouched.

Selection evidence must be sufficiently contiguous for the claim being made. Sparse prior-fold evidence does not substitute for complete prequential evidence.

## Feature-set identity

Minimum reproducibility fields:
- feature_set_id
- ordered feature_cols
- feature_manifest_version
- feature_policy_version
- source_set_fingerprint
- data_snapshot_id/hash
- model_id/version
- training_window_definition
- calibration_id/version
- router_id/version
- target_definition_version
- OOS definition
- seed
- implementation Git SHA
- experiment fingerprint

Equivalent feature sets must be deduplicated.

## Adoption gate

A feature-set/model candidate becomes a Promotion Candidate only after same-observation chronological comparison with the incumbent and the normal gates pass:

- reference primary relative OOS LogLoss improvement >= 3%
- reference auxiliary improvement >= 1% where applicable
- no regression in >= 70% of development evaluation blocks
- no newest/frozen-holdout deterioration
- no material calibration degradation
- zero PIT violations
- no unacceptable robustness, operational, cost or reproducibility regression

These are reference gates, not automatic promotion rules.

## Integrity

- missing/unavailable/unknown/delayed/source-failed are not zero;
- same-upstream mirrors/wrappers/republishers are not independent information;
- feature source code is not Production evidence;
- feature count is not information value;
- frozen holdout is never used for feature tuning;
- evidence-affecting feature changes invalidate stale OOS evidence;
- Production must preserve exact feature order and schema hash.


## Dynamic match-state / event-hazard research candidate

The research layer `src/research/match_state_hazard.py` introduces a separate candidate state representation for in-play prediction. It is RESEARCH_CANDIDATE only.

State inputs are cutoff-safe elapsed time, remaining regulation time, current score, tied/leading/trailing regime, red-card state and optional pre-match priors. The candidate next-event target is one of HOME_GOAL, AWAY_GOAL, HOME_RED, AWAY_RED or NO_EVENT.

Future states are propagated as deterministic probability mass rather than random Monte Carlo samples. This produces final 1X2 probabilities and a top-score distribution while reporting state pruning and retained mass.

This feature family must not be fed to Production merely because the module exists. Historical snapshots require explicit source availability at or before the prediction cutoff, and the current main contains no verified historical in-play snapshot dataset for this layer. Adoption requires identical chronological OOS comparison, next-event calibration, final-outcome calibration, case-level slices, robustness and frozen-holdout evidence.
