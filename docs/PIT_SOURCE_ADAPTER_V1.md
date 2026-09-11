# PIT Source Adapter V1

## Objective

Establish independently auditable historical source-availability timestamps for soccer research. `retrieved_at_utc` is never accepted as a historical availability timestamp.

## Correct PIT semantics

A completed historical match result cannot be treated as available before the match. The adapter therefore records when the completed result first appears in an archived source snapshot. During prediction replay, a past result is usable only when:

`source_available_at_utc <= prediction_cutoff_at_utc`

and the match itself is strictly before the prediction event.

## V1 source

Football-Data.co.uk historical CSVs are the first concrete adapter. The site provides historical results/odds CSV data and describes regular data updates and fixture collection timing. The adapter uses Internet Archive captures as evidence of the historical state of the downloadable CSV, and verifies the complete result identity (date, home team, away team, home goals, away goals, result) inside the archived snapshot before assigning an availability timestamp.

## Fail-closed rules

- No archive capture containing the completed result -> `UNVERIFIABLE`.
- Archive access failure -> `UNVERIFIABLE`.
- Missing match identity -> `UNVERIFIABLE`.
- No fallback to current retrieval time.
- No fallback to file modification time.
- No missing-to-zero conversion.
- API/source existence does not imply coverage.

## Fixed 15-competition universe

The adapter matrix explicitly classifies all 15 competitions. V1 has a concrete Football-Data adapter for Premier League, Championship, Bundesliga, Serie A, La Liga, Ligue 1, and Eredivisie. UEFA Champions League, UEFA Europa League, J1, J2, J3, DFB-Pokal, Club Friendlies, and Carabao Cup/EFL Cup remain `UNVERIFIED` until a source adapter passes the same evidence standard.

## Feature replay

Feature construction uses chronological past matches. For each rolling window, every required historical result must have an independently evidenced availability timestamp no later than the prediction cutoff. If a required result is not verifiable by the cutoff, the window is invalidated rather than silently replacing the missing result with an older match.

## Current limitation

Football-Data historical CSVs do not provide a reliable historical publication timestamp per row. V1 therefore uses archived snapshots as the evidence layer. Kickoff-time precision is conservative when the historical source exposes only a date; this will be strengthened by a dedicated historical fixture-time adapter before production prediction relies on the exact cutoff.
