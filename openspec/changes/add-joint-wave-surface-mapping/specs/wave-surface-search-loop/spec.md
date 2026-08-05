## Purpose

把联合波面的质量诊断转化为可执行、可排序、可回放的搜索动作，使系统能够迭代补测量或修正分解，并在达到决策价值时停止。

## ADDED Requirements

### Requirement: Decision value is axis and use-case aware
The system SHALL evaluate usefulness using each outcome axis's absolute width, relative width where defined, decision loss or tolerance, calibration status, and joint dependence rather than a single global width threshold.

#### Scenario: Large absolute duration is acceptable
- **WHEN** a travel-duration surface spans two hours around a 25-hour estimate and the declared decision tolerance is three hours
- **THEN** the loop may treat that axis as decision-useful even though its absolute width is large

#### Scenario: Relative width is undefined near zero
- **WHEN** an axis's reference magnitude is zero or crosses zero
- **THEN** the system does not divide by that magnitude and uses an explicitly declared absolute or loss-based criterion

### Requirement: Diagnostics produce typed loop actions
The system MUST map diagnostic evidence to typed actions including `measure`, `expand_variable`, `add_interaction`, `split_regime`, `minimize`, and `stop`, with expected value, cost, rationale, and affected entities.

#### Scenario: Measurement is the best next action
- **WHEN** sensitivity is concentrated in an uncertain observed variable and a feasible measurement is expected to reduce decision loss
- **THEN** the system ranks a `measure` action with the variable, required evidence, estimated benefit, and cost

#### Scenario: Wider sampling cannot fix missing structure
- **WHEN** diagnostics indicate an unexplained interaction or multiple regimes
- **THEN** the system proposes `add_interaction` or `split_regime` rather than only increasing particle count

### Requirement: Generator and evaluator remain separate
The system SHALL allow rules, retrieval, statistical methods, or language models to propose candidates, but every candidate MUST pass the same validation and surface-evaluation contract before it can affect the accepted result.

#### Scenario: Language model proposes a variable
- **WHEN** a language model proposes a new variable or mapping
- **THEN** the proposal remains a candidate with no factual authority until unit, provenance, evidence, and evaluation gates pass

#### Scenario: Invalid generated formula
- **WHEN** any generator proposes an unsafe, dimensionally invalid, or unsupported formula
- **THEN** the evaluator rejects it with a recorded reason and continues evaluating other valid candidates

### Requirement: Loop termination and replay
The system MUST terminate when decision value is achieved, expected marginal benefit is below threshold, the budget is exhausted, or a declared hard failure occurs, and MUST record every transition in an append-only replayable ledger.

#### Scenario: Useful surface stops the loop
- **WHEN** all required decision criteria pass and no mandatory calibration or validity gate fails
- **THEN** the loop emits `stop` with the accepted surface and the criteria evidence

#### Scenario: Budget exhaustion does not fabricate convergence
- **WHEN** the iteration, time, evidence, or compute budget is exhausted before decision criteria pass
- **THEN** the loop returns the best current surface as unresolved, records the unmet criteria, and MUST NOT label it converged

#### Scenario: Deterministic action replay
- **WHEN** an Agent replays the same ledger with matching inputs and evaluator versions
- **THEN** the system reproduces the same state transitions and action ordering within the declared tolerance
