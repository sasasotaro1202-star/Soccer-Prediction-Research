# Hierarchical Dynamic Soccer Simulator Research

## Purpose

This document converts the supplied mathematical blueprint into a staged research program for Soccer-Prediction-Research.

Target architecture:

D
→ P(Θ | D)
→ P(X,S0 | Θ,D)
→ P(E1:T,S1:T | Θ,X,D)
→ P(GH,GA | D)
→ P(Result | D)

where Θ is the latent team/player/tactical parameter state, X is lineup/formation/role state, S is dynamic match state, and E is the event sequence.

This is a research blueprint, not a claim that the complete simulator is currently Production.

## Implementation ladder

### Stage 0 — verified foundation

Reuse the existing PIT/timestamp contracts, chronological features, Score distribution models, calibration, predictability, uncertainty, dynamic routing, matchday intelligence, OOS/WFO and adoption gates.

Do not replace working foundations merely to introduce a more complex model.

### Stage 1 — latent strength state

Represent team strength as a time-varying latent state:

A(i,t) = A(i,t-1) + ε(A,i,t)
D(i,t) = D(i,t-1) + ε(D,i,t)

Use hierarchical shrinkage such as:

A(i,l) ~ N(μ(A,l), σ(A)^2)
μ(A,l) ~ N(μ(A,global), τ(A)^2)

and the analogous defensive hierarchy.

Any implementation must be compared with the incumbent Score model on identical chronological OOS observations.

### Stage 2 — dynamic match state

Represent:

S(t) = (time, score, cards, tactical state, pressure/territory state, fatigue proxies, data quality)

and estimate:

P(E(t+1) | S(t), Θ, X)

A discrete competing-risk hazard is an appropriate first implementation:

h(e,t) = exp(α(e) + β(e)^T Z(t))

P(E=e | S(t)) = h(e,t) / sum_j h(j,t).

The existing Match-State Hazard research candidate is the closest current implementation of this stage.

### Stage 3 — scenario propagation

For each state:

S(t+1) ~ P(S(t+1) | S(t), E(t), Θ, X)

Propagate a bounded probability tree and record pruned probability mass.

Terminal distributions should be internally consistent across 1X2, exact score, totals, BTTS and selected event probabilities rather than relying on unrelated downstream predictors.

### Stage 4 — lineup distribution

Model:

P(X | D)

instead of treating expected or announced XI as certain.

Separate confirmed, probable, unavailable and unknown statuses. Lineup uncertainty is a predictive uncertainty component and must remain PIT-valid.

### Stage 5 — player/role interaction

A conceptual team attack representation is:

TeamAttack = sum_i PI(i) + sum_(i<j) I(i,j)

but interactions should be sparse, regularized and role-aware. Do not introduce unrestricted O(n²) player-pair parameters without sufficient OOS evidence.

### Stage 6 — full event simulator

Final target:

E(t) ~ P(E(t) | S(t), Θ, X)
S(t+1) ~ P(S(t+1) | S(t), E(t), Θ, X)

Use posterior-predictive / Monte Carlo simulation over complete matches.

The final score and result distributions are obtained by marginalization from the simulated match distribution.

## PIT contract

Every state/event feature must preserve event_time, prediction_time, prediction_cutoff, source_available_at, published_at, retrieved_at and revision_time.

retrieved_at is never historical availability proof.

Unknown historical availability is fail-closed.

In-play research must also prevent outcome maturity leakage: a snapshot is scored only after the information required to label its next event is demonstrably available.

## Evaluation contract

Every stage requires chronological WFO/OOS, identical comparison observations, event-level clustering when multiple snapshots belong to one match, probability metrics, score-distribution metrics, case-level slices, robustness tests, frozen-holdout separation and reproducible lineage.

Accuracy alone is insufficient.

## Adoption contract

No stage becomes Production because code exists, CI is green, an artifact was generated, or an external source reports improvement.

Promotion requires the existing PIT, chronological OOS, calibration, robustness, holdout and reproducibility gates.

## Autonomous execution

The GitHub control plane runs this frontier on a recurring schedule and the 30-minute orchestrator catches up stale runs.

The lane is research-only and never modifies Champion, Production bundle, frozen holdout or target semantics.

When prerequisites are absent, the lane returns WARMUP/HOLD instead of fabricating data or performance evidence.

## Complexity rule

The complete simulator is the destination, not the starting point.

At every stage:

incremental OOS value + calibration + robustness + failure reduction

must justify:

model complexity + data dependency + compute cost + operational failure surface.

If a simpler model generalizes equally well, retain the simpler model.
