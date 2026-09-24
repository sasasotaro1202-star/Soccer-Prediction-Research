# Competition catalog policy

The system distinguishes **research coverage** from **production eligibility**.

Research coverage should aim to include, where reproducible data can be obtained, major domestic leagues, major domestic cups, the full UEFA association-football competition family, international friendlies, and other materially relevant competitions. The UEFA scope explicitly includes the Champions League, Europa League, Conference League, Super Cup, Youth League, Women's Champions League, Women's Europa Cup, EURO, European Qualifiers, Nations League, Women's EURO, Women's European Qualifiers, Women's Nations League, Under-21, Under-19, Under-17, Women's Under-19, Women's Under-17 and Regions' Cup. Domestic-cup coverage also includes the Emperor's Cup.

A competition must never become production-eligible merely because its fixtures or results can be downloaded. It requires auditable source availability timestamps, PIT verification, data-quality validation, pre-kickoff feature availability, chronological OOS evidence, calibration, sufficient sample size, and an explicit adoption decision.

When an adapter is unavailable or its PIT provenance is not auditable, the competition remains visible as a research target but is excluded from production. Unknown is a first-class state; it is never coerced to zero or treated as verified.

This policy is intentionally conservative: broad research coverage and narrow production eligibility are compatible and preferred for real-world robustness.


## Explicit scope expansion

The research catalog treats the competitions above as first-class targets. This is a coverage directive, not an assertion that every source currently supplies auditable historical PIT evidence. UEFA futsal competitions are intentionally outside this soccer/association-football scope and should be handled as a separate sport dataset if later requested.
