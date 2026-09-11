# Soccer migration status

This repository is the research-controlled soccer system. The legacy production repository remains the source of existing soccer acquisition/backtest logic and is not deleted.

## Evidence reviewed
- Legacy `backtest.py`: V9 multi-source backtest using Football-Data.co.uk, Understat, SofaScore, self-calculated Elo/form/H2H/rest/season transition; chronological validation and post-match state updates are documented in the file.
- Legacy `soccer-research-ai.yml`: V12 autonomous research already has OpenAI advisory, candidate evaluation, fail-closed promotion checks, provenance and research-state artifacts.
- New repository specs: data source contract, chronological OOS, and timestamp/leakage rules.

## Migration classification

| Legacy area | Classification | Action |
|---|---|---|
| Team normalization | KEEP/MODIFY | Preserve useful aliases; make IDs/source mapping explicit |
| Elo/form/H2H/season transition | WRAP/MODIFY | Recompute strictly from prior PIT rows |
| Football-Data acquisition | MOVE/WRAP | Use new source adapter and coverage matrix |
| Understat | MOVE/MODIFY | Audit retrieval, mapping, timestamps and zero-result cause before production use |
| SofaScore player/MOM | MOVE/MODIFY | Keep historical player state only when availability timestamp is provable |
| ExtraTrees/RF/HGB/Logistic | MOVE | Candidate models under chronological validation |
| Score Poisson/Dixon-Coles | MOVE/MODIFY | Preserve, then evaluate on the same locked OOS |
| Legacy V12 research loop | WRAP | Reuse advisory ideas; adoption is controlled by new deterministic gate |
| Legacy workflows | MERGE/RETIRE gradually | Do not run duplicate research loops after parity is proven |

## Current implementation
1. Competition contract for 15 fixed competitions.
2. Football-Data historical acquisition adapter for seven mapped competitions.
3. Coverage matrix with explicit unavailable status for unmapped competitions.
4. PIT gate that refuses to substitute retrieval time for source availability time.
5. Rolling form features from prior matches only.
6. Logistic/ExtraTrees/HistGradientBoosting candidates.
7. Chronological walk-forward OOS with validation-based model selection.
8. LogLoss, Accuracy, Brier and ECE.
9. Deterministic adoption gate.
10. OpenAI research advisory that cannot directly promote a model.
11. GitHub Actions automated research cycle.

## Important current blocker
The current Football-Data adapter intentionally records `source_available_at_utc` as unknown. Therefore the research engine will report `BLOCKED` for PIT/OOS adoption until source availability can be proven or a source adapter with auditable publication timestamps is added.

This is deliberate: missing availability metadata is not converted into a guessed timestamp.
