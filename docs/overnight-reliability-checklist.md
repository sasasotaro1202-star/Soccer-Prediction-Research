# Overnight reliability checklist

Before treating an overnight run as useful, verify:

1. Workflow completed within its bounded timeout.
2. Dependency installation succeeded or exhausted bounded retries with diagnostics.
3. Source/tests compile successfully.
4. Full deterministic test suite passes.
5. PIT audit artifact exists and is internally consistent.
6. Competition coverage is separated from production eligibility.
7. No missing value was silently converted to zero.
8. No post-kickoff information entered an OOS feature.
9. Research failures are not masked by a successful workflow conclusion.
10. Production output is emitted only when the explicit adoption gate passes.
11. Artifacts are uploaded even when an intermediate audit step fails.
12. Scheduled jobs cannot overlap the same audit workload.

A successful Actions status is therefore necessary but not sufficient for production readiness.
