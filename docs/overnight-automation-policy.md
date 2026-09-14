# Overnight automation policy

## Objective

Keep unattended automation conservative and production-oriented: reliability first, chronological out-of-sample validity second, predictive quality third, and speed only where it does not weaken either.

## Rules

- Never weaken PIT, leakage, or adoption gates to obtain a green run.
- Never convert missing data to zero.
- Never use random train/test splits for temporal prediction research.
- Never use post-kickoff or post-event information in pre-kickoff features.
- Never promote a candidate solely because it improves one historical metric.
- Failed, cancelled, incomplete, or artifact-missing runs are not successful research results.
- Production prediction must fail closed when the adopted registry/model bundle is absent or inconsistent.
- Score and MOM outputs remain abstention-safe until their end-to-end pre-kickoff data contracts and tests are satisfied.
- Prefer deterministic, bounded, retry-safe Actions over long unattended jobs with weak failure diagnostics.

## Overnight execution

GitHub Actions may execute scheduled and push-triggered jobs without an interactive session. Each workflow should have bounded timeouts, explicit dependency installation, deterministic tests, artifact validation, and concurrency controls where appropriate. Heavy research jobs should remain separate from the lightweight CI gate.

## Completion standard

A green Action is necessary but not sufficient. A completed research increment requires inspectable logs/artifacts and a passing gate. If evidence is insufficient, the system must report a blocker rather than silently treating the result as production-ready.
