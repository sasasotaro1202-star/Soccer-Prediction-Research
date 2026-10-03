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

## 2026-10-03 UTC — post-probe forensic finding

Run `37158461663` did retrieve 12 future fixture rows and therefore demonstrated live fixture discovery, but the rich-data validation was too permissive at that revision. The first three selected FotMob match-detail requests used IDs rendered as strings such as `5975044.0`; the saved responses were JSON error payloads (`error=true`, `Data not found`) rather than valid match-detail documents. The old validator counted those error responses as payloads, so the workflow incorrectly concluded availability was proven.

This is retained as a failure, not rewritten as a success: **fixture discovery = observed; valid FotMob detail payload = not verified**.

The current branch hardens this path by preserving integer provider IDs after CSV parsing, validating FotMob `general.matchId`, rejecting `error=true` responses, separating raw responses from valid payload counts, and enforcing provider league-ID identity so same-name competitions from Ghana/Algeria cannot be silently relabeled as EPL/FL1.

Relevant forensic evidence from the run artifact:
- `rows=12`, `selected_rows=3`, `raw_payload_records=3`
- all 3 FotMob raw records returned `Data not found`
- feature-channel coverage was 0% for market/lineup/injury/standings/weather/H2H/recent-form
- retrieval timestamps were before kickoff, but this is observation evidence only and does not prove historical publication availability

The branch must pass a fresh live smoke run after these fixes before FotMob match-detail acquisition is marked VERIFIED.
