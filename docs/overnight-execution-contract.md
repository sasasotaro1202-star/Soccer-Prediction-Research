# Overnight execution contract

This repository is designed for unattended execution. The objective is not to guarantee that external services can never fail; it is to guarantee that an external failure cannot silently produce an unsafe production result.

## Required behavior

- Retry transient dependency/network failures within fixed bounds.
- Preserve diagnostics whenever a stage fails.
- Keep deterministic tests ahead of expensive research.
- Prevent overlapping scheduled instances of the same workload.
- Keep incomplete PIT evidence explicitly incomplete.
- Never replace unknown PIT availability with retrieval time.
- Never promote a research result without the explicit production adoption gate.
- Prefer abstention over fabricated predictions.
- Keep Score and MOM outputs gated independently from 1X2.
- Treat broad competition discovery as research coverage, not automatic production support.

## Morning readiness interpretation

`workflow success` = the automation executed to completion.

`production ready` = the explicit evidence gates passed.

These are deliberately different states.
