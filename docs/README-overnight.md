# Overnight state

The repository uses scheduled GitHub Actions for unattended research and integrity checks.

The operating rule is conservative: broad competition coverage is encouraged, but only competitions with auditable PIT provenance, clean data, valid chronological OOS evaluation, calibration evidence, and an explicit adoption gate may reach production.

Automation is bounded and retry-safe. External failures are not hidden. Diagnostic artifacts should be preserved, and incomplete or failed research must not be promoted.

Priority order: correctness → PIT/provenance → data quality → chronological OOS → calibration → production contract → runtime optimization.
