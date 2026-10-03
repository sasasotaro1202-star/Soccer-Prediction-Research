# Live Source Availability Evidence

## 2026-10-03 UTC probe

The live rich-data availability workflow was executed from GitHub Actions run `37153518092` before the FotMob discovery path was added.

Observed fixture discovery:

- ESPN: J2/J3 endpoint failures were observed; other configured ESPN competitions did not produce upcoming rows in the tested window.
- SofaScore scheduled-events endpoint: retrieval failed.
- football-data.co.uk current-season CSVs: all six requested CSVs were retrieved successfully.
- Retrieved raw rows: E0=50, N1=63, SP1=69, I1=50, D1=36, F1=45.
- Total raw CSV rows retrieved: 313.
- Future fixtures inside the tested 48-hour window: 0.
- Rich match payloads: 0 because there were no eligible fixture rows.
- Result: FAILED / UNVERIFIED for live fixture availability, not a successful rich-data proof.

Interpretation:

A source can be reachable and return a valid dataset while still providing no future fixtures for the requested horizon. Reachability, non-empty raw data and future-fixture coverage are separate measurements.

The branch subsequently added FotMob public date-based fixture discovery and match-detail enrichment. The newest branch revision must pass a fresh live smoke run before FotMob is marked as EXECUTED/VERIFIED for this project.

Historical PIT is not inferred from this live probe. Retrieval timestamps remain observation timestamps only.
