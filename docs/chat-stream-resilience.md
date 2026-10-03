# Chat stream resilience

The chat surface is not a durable transport for long-running research.

## Contract

1. Chat-triggered work must acknowledge quickly and dispatch long-running work asynchronously.
2. The chat request must never synchronously poll a research workflow until completion.
3. Request state is durable under `requests/state/<request_id>.json`.
4. A request is idempotent by `request_id`; `DISPATCHING` is fail-closed and must be reconciled rather than duplicated.
5. Workflow execution must continue independently if the chat UI stream disconnects, times out, or the client is backgrounded.
6. Completion and failure evidence must be recoverable from GitHub Actions state/artifacts rather than from the transient chat stream.
7. No `|| true`, forced success, fake completion, or silent failure is permitted.

## Scope

This contract improves recovery from chat transport interruption. It does not and cannot control the ChatGPT application's UI streaming transport itself.

The production research gates remain unchanged: PIT, chronological OOS/WFO, calibration, robustness, frozen holdout, artifact integrity, reproducibility, and fail-closed behavior.
