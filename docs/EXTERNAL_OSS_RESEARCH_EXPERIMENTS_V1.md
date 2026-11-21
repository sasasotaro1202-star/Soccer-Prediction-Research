# External OSS Research Experiment Cards v1

Status: RESEARCH-ONLY / NOT EXECUTED / NOT PRODUCTION
Registry anchor: `c6440d9b850bd99f91b066378ff2ef66312ca9fd`

## Objective

Convert the external OSS queue into reproducible challenger experiments without changing Production.

## Global experiment contract

Every experiment must lock before evaluation:

- target contract: exactly one target at a time
- data snapshot and dataset hash
- prediction cutoff
- PIT policy
- feature manifest
- training window
- chronological OOS/WFO folds
- calibration procedure
- router policy
- random seeds where applicable
- compute budget
- stopping rule

Frozen Holdout must remain unseen during candidate selection.

### Required metrics

Primary:
- LogLoss
- Brier
- target-specific accuracy / Top-k where applicable

Secondary:
- ECE / calibration error
- newest OOS block
- fold stability
- case-level error
- robustness slices
- OOD / predictability interactions
- latency / operational failure rate

### Adoption gate

A candidate can advance only if:

1. PIT is proven for every feature/input.
2. Local reproduction is deterministic enough to audit.
3. Chronological OOS/WFO shows incremental value versus the locked incumbent.
4. Calibration is non-inferior or improved.
5. Robustness does not show a material regression.
6. Newest unseen block does not show unexplained deterioration.
7. Frozen Holdout remains untouched during selection.
8. Shadow behavior is safe.
9. Complexity / maintenance / security / cost are acceptable.

Decision states after the gate are explicit:
- ADOPT: all required evidence gates pass and the change is approved for integration/shadow progression.
- HOLD: evidence is incomplete, mixed, or operationally acceptable but not yet sufficient for adoption.
- REJECT: a hard gate fails, including PIT failure, material OOS/robustness/calibration regression, security/cost violation, or irreproducible behavior.

No external benchmark or GitHub Star count satisfies these gates.

---

## Card A — TimesFM

Repository: `google-research/timesfm`
Layer: Forecasting
Priority: HIGH
Research status: CANDIDATE

Hypothesis:
A time-series foundation-model representation or forecast component may add information beyond the incumbent forecasting/statistical stack.

First test:
- use only information available at the locked prediction cutoff
- compare as a standalone challenger first
- then test calibrated probability transformation / blending only if standalone value exists

Required slices:
- competition
- season/phase
- data completeness
- high/low predictability
- OOD

Stop conditions:
- any PIT failure
- unstable chronological performance
- calibration deterioration without compensating value
- excessive compute/latency

---

## Card B — PyMC

Repository: `pymc-devs/pymc`
Layer: Probabilistic
Priority: HIGH
Research status: CANDIDATE

Hypothesis:
Hierarchical probabilistic modeling may improve latent team-strength estimation and uncertainty decomposition.

First test:
- hierarchical team/competition effects
- posterior predictive uncertainty
- compare against incumbent strength model

Do not:
- use outcome-derived future information
- tune priors on Frozen Holdout
- convert posterior uncertainty directly into confidence without calibration

---

## Card C — statsmodels

Repository: `statsmodels/statsmodels`
Layer: Statistics
Priority: HIGH
Research status: CANDIDATE

Hypothesis:
Classical statistical baselines can expose whether added complexity is actually necessary.

First test:
- Poisson / count-oriented statistical baselines where target contract permits
- compare against incumbent using identical chronological folds

Purpose:
Baseline quality / falsification / complexity control.

A win by the simpler model is valid research evidence.

---

## Card D — sktime

Repository: `sktime/sktime`
Layer: Time Series
Priority: HIGH
Research status: CANDIDATE

Hypothesis:
A standardized forecasting/model selection interface can broaden the candidate family while preserving a consistent evaluation contract.

First test:
- model-family benchmark
- fixed fold definitions
- no candidate-specific access to future data

Selection:
Nested/prequential where model or window selection depends on temporal data.

---

## Card E — Darts

Repository: `unit8co/darts`
Layer: Time Series
Priority: HIGH
Research status: CANDIDATE

Hypothesis:
A wider forecasting model family may capture nonlinear temporal structure missed by current components.

First test:
- small bounded candidate set
- fixed compute budget
- identical preprocessing
- chronological OOS

Complexity penalty:
Prefer the smallest model that produces durable incremental OOS value.

---

## Card F — Pyro

Repository: `pyro-ppl/pyro`
Layer: Probabilistic
Priority: HIGH
Research status: CANDIDATE

Hypothesis:
Probabilistic-programming flexibility may support richer latent-state or hierarchical uncertainty models.

First test:
- one narrowly defined probabilistic challenger
- compare posterior calibration and case-level uncertainty behavior

Do not:
Treat model disagreement or posterior width as prediction reversal signals.

---

## Card G — Agent-Reach

Repository: `Panniantong/Agent-Reach`
Layer: Retrieval / Agent
Priority: HIGH
Research status: CANDIDATE

Hypothesis:
Better external-information discovery may improve research coverage and source discovery rather than directly improve prediction accuracy.

First test:
- research-only retrieval
- source provenance capture
- publication/availability/retrieval timestamps
- duplicate-source and independence grouping
- no automatic promotion of retrieved data into Production features

PIT requirement:
Current-page availability does not establish historical availability. Historical PIT must be proven separately.

---

## Card H — Qlib

Repository: `microsoft/qlib`
Layer: Research / Quant
Priority: HIGH
Research status: CANDIDATE

Hypothesis:
Reusable research/feature/model/experiment mechanisms may improve systematic challenger discovery.

First test:
- transfer mechanisms, not domain assumptions
- preserve soccer-specific target and PIT contracts
- reproduce locally before any integration

The repository itself is not Production evidence.

---

## Tooling-only candidates

These are initially research-process candidates, not prediction models:

- `obra/superpowers`
- `mattpocock/skills`
- `openclaw/openclaw`
- `NousResearch/hermes-agent`
- `mlflow/mlflow`
- `feast-dev/feast`
- `Arize-ai/phoenix`
- `theDotmack/claude-mem`
- `Graphify-Labs/graphify`
- `firecrawl/firecrawl`

For these, success means better reproducibility, retrieval, experiment control, memory, monitoring, or research throughput—not an assumed accuracy gain.

---

## Experiment identity

Each future run should produce a stable identity from:

`repository + commit/tag + adapter_version + target + snapshot_hash + fold_policy + feature_manifest + calibration + config`

No duplicate experiment should be counted twice.

## Evidence states

E0 IDEA
E1 EXTERNAL_CLAIM
E2 EXTERNAL_IMPLEMENTATION
E3 LOCAL_REPRODUCTION
E4 LOCAL_OOS
E5 ROBUSTNESS
E6 FROZEN_HOLDOUT
E7 SHADOW/PRODUCTION_EVIDENCE

Current status for every card in this file:
**E1/E2 discovery only; no local performance claim.**
