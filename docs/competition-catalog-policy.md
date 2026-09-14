# Competition catalog policy

The system distinguishes **research coverage** from **production eligibility**.

Research coverage should aim to include, where reproducible data can be obtained, major domestic leagues, major domestic cups, UEFA competitions, and other materially relevant competitions. This includes the existing target set plus FA Cup, Carabao Cup/EFL Cup, Copa del Rey, Coppa Italia, Coupe de France, KNVB Cup, Taça de Portugal, Belgian Cup, Scottish Cup, Turkish Cup, Greek Cup, J.League Cup, Emperor's Cup, Conference League and other major competitions discovered by the research process.

A competition must never become production-eligible merely because its fixtures or results can be downloaded. It requires auditable source availability timestamps, PIT verification, data-quality validation, pre-kickoff feature availability, chronological OOS evidence, calibration, sufficient sample size, and an explicit adoption decision.

When an adapter is unavailable or its PIT provenance is not auditable, the competition remains visible as a research target but is excluded from production. Unknown is a first-class state; it is never coerced to zero or treated as verified.

This policy is intentionally conservative: broad research coverage and narrow production eligibility are compatible and preferred for real-world robustness.
