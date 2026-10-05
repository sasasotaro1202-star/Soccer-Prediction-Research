# Soccer World Model Continuous Autonomy

## Objective

Keep the dynamic World Model research moving after the chat session ends without allowing unverified research to alter Production.

## Control loop

PROSPECTIVE CAPTURE → LATER LABEL MATURITY → PIT / SCHEMA GATE → MATCH-LEVEL WFO/OOS → PREQUENTIAL CALIBRATION → ROBUSTNESS / ABLATION → INCUMBENT COMPARISON → FROZEN HOLDOUT → RELEASE GATE → SHADOW / PRODUCTION ONLY AFTER ADOPTION

## Autonomous lanes

- Soccer Prospective In-Play PIT Capture: scheduled prospective live-state observations using a free/public endpoint.
- Soccer Prospective In-Play Maturity: converts later state transitions into bounded next-event labels and executes the staged dynamic-hazard research.
- Soccer Dynamic Simulator Research: verifies frontier prerequisites and catches up the dynamic research lane.
- Soccer World Model Supervisor: state-based catch-up that dispatches current-main research lanes when idle.
- Soccer Autonomous Orchestrator: repository-wide bounded reconciliation.
- Soccer Control Plane Watchdog: stale control-plane recovery.
- Action Failure Recovery: one bounded recovery for transient/timed-out runs.

## PIT guarantees

Prospective observations are never backdated. The observation timestamp is preserved separately from retrieval metadata, alongside a source response hash. Later label maturity is a distinct phase.

A transition is accepted only when:
1. the next observation is within the bounded five-minute window;
2. exactly one goal/card state increments;
3. the state is otherwise consistent.

Ambiguous or missing information is excluded rather than imputed.

## Research authority

The World Model track remains: research_only=true, production_usable=false, performance_verified=false, promotion_candidate=false.

Green CI, a completed OOS run, or a generated artifact does not override this state.

## Long-running behavior

Missed cron ticks do not stop research. The autonomous orchestrator and watchdog independently catch up stale lanes. Concurrent work is bounded. Transient failures may be retried once; deterministic PIT/schema failures remain blocking.

The system is designed so that accumulation of prospective PIT-valid data eventually supplies the previously missing in-play research dataset without inventing historical availability.
