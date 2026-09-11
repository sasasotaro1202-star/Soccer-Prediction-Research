# V9/V12 → Research Engine Migration Plan

## Objective

Migrate the proven V9/V12 soccer acquisition, features, models, and research logic into the new Research Engine without discarding the old system. All candidates must pass data coverage, source-availability, PIT, chronological OOS, stability, and adoption gates before entering production.

## Pipeline

V9/V12 legacy baseline
→ source-specific adapters
→ 15-competition coverage matrix
→ source availability timestamps + PIT replay
→ legacy feature/model wrappers
→ Research Engine
→ identical chronological OOS
→ legacy vs Research Engine comparison
→ weakness analysis
→ candidate generation
→ independent OOS validation
→ adoption gate
→ registry/manifest
→ adopted-model-only production prediction

## Safety rules

1. A successful GitHub Actions run is not evidence of successful OOS. `BLOCKED` is a valid result.
2. Retrieval time is never substituted for source availability time.
3. Missing, unavailable, parse failure, mapping failure, and true zero remain distinct states.
4. Random train/test splitting is prohibited.
5. The same OOS period cannot both select and finally validate a candidate.
6. Candidate models cannot enter production before adoption.
7. Legacy code is classified KEEP/MOVE/WRAP/MODIFY/NEW; no wholesale deletion.

## Fixed competition set

Premier League, Championship, Bundesliga, Serie A, La Liga, Ligue 1, Eredivisie, UEFA Champions League, UEFA Europa League, J1, J2, J3, DFB-Pokal, Club Friendlies, Carabao Cup/EFL Cup.

## Migration stages

- Phase 0: inventory legacy files/functions/workflows and new repository.
- Phase 1: enforce data contract.
- Phase 2: build actual coverage matrix.
- Phase 3: audit PIT and leakage.
- Phase 4: implement source adapters.
- Phase 5: integrate research infrastructure.
- Phase 6: separate PIT-safe feature layers.
- Phase 7: separate/wrap models.
- Phase 8: chronological OOS.
- Phase 9: research engine.
- Phase 10: locked legacy baseline OOS.
- Phase 11: candidate OOS.
- Phase 12: adoption gate.
- Phase 13: production prediction.
- Phase 14: scheduled automation.

## Current verified state

The first Research Cycle completed successfully at the workflow level, but the engine correctly blocked model/OOS adoption because it had no auditable source-availability timestamps for the acquired historical rows. This is a safety success, not a model-performance result.

## Immediate implementation order

1. Add explicit source metadata fields to the canonical match record.
2. Add a fail-closed PIT replay utility.
3. Add source adapters with auditable availability metadata where the source supports it.
4. Import legacy V9 feature/model functions through wrappers rather than copying blindly.
5. Generate coverage reports per competition × season × source × field.
6. Run identical OOS splits for legacy and Research Engine.
7. Add independent candidate holdout and adoption gate.
8. Only then enable production prediction.
