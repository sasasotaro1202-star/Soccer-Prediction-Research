# External OSS Research Queue v2

Status: RESEARCH-ONLY / NON-PRODUCTION  
Snapshot: 2026-10-08  
Base main SHA: `c6440d9b850bd99f91b066378ff2ef66312ca9fd`

## Purpose

This registry prioritizes external GitHub repositories for research in the soccer prediction system. It does **not** constitute Production evidence and does not authorize automatic adoption.

## Decision boundary

`DISCOVERY → SOURCE VERIFY → COST/SECURITY/LICENSE → PIT → LOCAL REPRODUCTION → CHRONOLOGICAL OOS/WFO → CALIBRATION → ROBUSTNESS → FROZEN HOLDOUT → SHADOW → ADOPT/HOLD/REJECT`

Unknown or unverifiable PIT is fail-closed.

Production assets remain protected:
- model
- feature schema/manifest
- target definition
- calibration
- frozen holdout
- production evidence

## Current research queue

| Rank | Repository | Stars | Layer | Priority | Research role |
|---:|---|---:|---|---|---|
| 1 | `google-research/timesfm` | 34,154 | Forecasting | HIGH | Time-series foundation model |
| 2 | `Panniantong/Agent-Reach` | 93,460 | Retrieval / Agent | HIGH | External information acquisition |
| 3 | `microsoft/qlib` | 49,210 | Research / Quant | HIGH | Research pipeline / model experimentation |
| 4 | `obra/superpowers` | 296,447 | Agent / Skills | MEDIUM | Research engineering workflow |
| 5 | `mattpocock/skills` | 279,882 | Agent / Skills | MEDIUM | Reusable skill orchestration |
| 6 | `pymc-devs/pymc` | 9,796 | Probabilistic | HIGH | Bayesian / hierarchical modeling |
| 7 | `openclaw/openclaw` | 391,611 | Agent / Automation | MEDIUM | Autonomous orchestration |
| 8 | `sktime/sktime` | 10,060 | Time Series | HIGH | Forecasting / time-series research |
| 9 | `NousResearch/hermes-agent` | 251,990 | Agent | MEDIUM | Autonomous research agent |
| 10 | `statsmodels/statsmodels` | 11,678 | Statistics | HIGH | Classical statistical modeling |
| 11 | `unit8co/darts` | 9,537 | Time Series | HIGH | Forecasting model zoo |
| 12 | `pyro-ppl/pyro` | 9,063 | Probabilistic | HIGH | Probabilistic programming |
| 13 | `Nixtla/statsforecast` | 4,922 | Forecasting | HIGH | Statistical forecasting baselines |
| 14 | `Nixtla/neuralforecast` | 4,279 | Forecasting | HIGH | Neural forecasting |
| 15 | `mlflow/mlflow` | 28,308 | MLOps | MEDIUM | Experiment/model lineage |
| 16 | `feast-dev/feast` | 7,323 | MLOps | MEDIUM | Feature store / lineage |
| 17 | `Arize-ai/phoenix` | 11,744 | Evaluation | MEDIUM | Evaluation / observability |
| 18 | `theDotmack/claude-mem` | 97,844 | Memory | MEDIUM | Long-term agent memory |
| 19 | `Graphify-Labs/graphify` | 124,724 | Knowledge | MEDIUM | Graph / knowledge context |
| 20 | `firecrawl/firecrawl` | 189,554 | Retrieval | MEDIUM | Web crawling / extraction |

## First research tracks

### Track A — Prediction core
`google-research/timesfm`, `pymc-devs/pymc`, `statsmodels/statsmodels`, `sktime/sktime`, `unit8co/darts`, `pyro-ppl/pyro`, `Nixtla/statsforecast`, `Nixtla/neuralforecast`

Gate: target-specific local reproduction → chronological OOS/WFO → calibration → robustness → frozen holdout.

### Track B — Research automation
`Panniantong/Agent-Reach`, `obra/superpowers`, `mattpocock/skills`, `openclaw/openclaw`, `NousResearch/hermes-agent`

Gate: PIT-safe retrieval/provenance → security/authority audit → deterministic research handoff. No Production model authority.

### Track C — Research infrastructure
`microsoft/qlib`, `mlflow/mlflow`, `feast-dev/feast`, `Arize-ai/phoenix`

Gate: local reproducibility, artifact lineage, online/offline parity, and no contamination of frozen evaluation.

### Track D — Memory / knowledge
`theDotmack/claude-mem`, `Graphify-Labs/graphify`

Gate: provenance, deduplication, source-independence audit, and contamination/leakage audit.

### Track E — External retrieval
`firecrawl/firecrawl`

Gate: license review, security review, PIT/provenance validation, cost confirmation, local reproduction. External retrieval output is not Production evidence.

## Adoption status

All entries in this file remain **RESEARCH CANDIDATES** for Production purposes. `statsmodels/statsmodels` has reached **E2_LOCAL_IMPLEMENTATION_AND_TESTS** only; it has not reached OOS-performance verification, robustness, Frozen Holdout, Shadow, or Production adoption. No repository is promoted to Production by this registry.

### Verified metadata for statsmodels

- official repository: `statsmodels/statsmodels`
- pinned release: `0.15.0`
- official license: BSD-3-Clause
- source verification: official statsmodels repository/release documentation reviewed 2026-10-08
- cost model: OSS package; no runtime SaaS/API dependency
