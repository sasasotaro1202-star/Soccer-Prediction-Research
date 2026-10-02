# Chat execution contract

Chat-triggered work must not synchronously wait for long research or prediction workflows.

## Request pattern

Create one JSON file under `requests/inbox/`:

```json
{
  "request_id": "unique-id",
  "operation": "daily_forecast",
  "requested_at": "2026-10-02T16:00:00+09:00"
}
```

Supported operations are deliberately allow-listed: `daily_forecast`, `matchday_intelligence`, `predictability_research`, `research_9h`.

The chat-side action only enqueues the request. The gateway validates it, records durable state, dispatches the corresponding GitHub Actions workflow, and returns control without polling for completion.

## Safety

The gateway does not accept an arbitrary workflow filename or shell command.

Request state is stored at `requests/state/<request_id>.json`. Existing state prevents duplicate dispatch. A request stuck at `DISPATCHING` is fail-closed and requires reconciliation rather than silently dispatching a duplicate job.

This mechanism does not change PIT, OOS, holdout, production, or adoption gates.