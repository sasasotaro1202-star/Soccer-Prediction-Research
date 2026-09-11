# V9/V12 → Research Engine Adapter Contract

## Purpose

Preserve the legacy V9/V12 soccer feature and model implementations while placing their execution behind the Research Engine's safety boundary.

## Ownership

- Legacy V9/V12: feature/model calculation and existing soccer-specific logic.
- Adapter: input/output translation only; no silent defaults and no timestamp invention.
- Research Engine: data acquisition, Coverage, PIT/Leakage gate, chronological OOS, metrics, candidate selection, independent validation, Adoption Gate, Registry, Production eligibility.

## Required input

Each legacy prediction must receive a `match_id`, `prediction_cutoff_at_utc`, and `pit_verified=true`. The adapter rejects anything else.

## Required output

At minimum, legacy predictions expose H/D/A probabilities summing to 1. Score candidates and metadata are preserved when available.

## Safety rules

1. Retrieval time is never substituted for source availability time.
2. Missing PIT evidence fails closed.
3. Legacy predictions are never allowed to bypass the Research Engine OOS split.
4. Legacy code is not copied or rewritten merely for migration.
5. Candidate models cannot enter production through this adapter.
6. Baseline identity/version must be retained so Old V9/V12 and New candidates are compared on the same OOS rows.

## Migration sequence

1. Introduce adapter contract and tests.
2. Wrap concrete V9/V12 prediction callable without changing its implementation.
3. Feed only PIT-verified records from the Research Engine.
4. Emit normalized baseline predictions.
5. Run Old V9/V12 and candidate models on identical chronological OOS rows.
6. Compare metrics and stability.
7. Apply independent-OOS Adoption Gate.
