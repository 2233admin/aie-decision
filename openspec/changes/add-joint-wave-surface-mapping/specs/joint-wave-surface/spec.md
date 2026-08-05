## Purpose

以可扩展的带权粒子波面表达未来事件的联合不确定性，保留多峰和偏斜，并严格区分可能性支持与经过校准的概率语义。

## ADDED Requirements

### Requirement: Joint surface preserves outcome structure
The system SHALL represent a joint surface across all declared outcome axes and provide axis marginals, detected modes, weighted regions, and dependence summaries without reducing the result to a single mean.

#### Scenario: Bimodal result remains bimodal
- **WHEN** valid inputs support two separated outcome regimes
- **THEN** the output reports both modes and their regions rather than returning only a mean located between them

#### Scenario: Requested marginal is produced
- **WHEN** an Agent requests the time-axis marginal from a time-by-price surface
- **THEN** the system returns the marginal with its unit, surface semantics, source surface identifier, and approximation diagnostics

### Requirement: Surface semantics gate
The system MUST label a surface as `possibility_surface` unless declared distribution semantics and a validated calibration basis justify `probability_surface`.

#### Scenario: Uncalibrated bounds yield possibility
- **WHEN** the inputs consist of subjective bounds, heuristic weights, or unmeasured calibration
- **THEN** the output is labeled `possibility_surface` and MUST NOT report its weighted regions as empirically calibrated confidence intervals

#### Scenario: Probability claim lacks calibration basis
- **WHEN** a caller requests `probability_surface` but one contributing distribution or weight lacks declared semantics or calibration evidence
- **THEN** the system rejects the probability label and identifies the blocking contributors

### Requirement: Surface quality diagnostics
The system SHALL report sharpness, calibration status, effective sample size, entropy, sensitivity, residual dependence, multimodality, and constraint failures with axis-aware units where applicable.

#### Scenario: Degenerate particle support
- **WHEN** normalized support collapses to an effective sample size below the configured validity floor
- **THEN** the surface is marked unreliable and the system recommends resampling or revising mappings instead of presenting a precise result

#### Scenario: Residual interaction is detected
- **WHEN** residual dependence remains materially above the declared threshold after accounting for existing mappings
- **THEN** the diagnostics identify affected variables or axes and emit evidence for an interaction or latent-variable action

### Requirement: Bounded-memory deterministic evaluation
The system MUST support deterministic replay from the same input, seed, schema versions, and evaluator version, and MUST NOT require materializing a complete candidate-by-variable-by-particle tensor.

#### Scenario: Replay matches
- **WHEN** a recorded evaluation is replayed with identical declared versions and seed
- **THEN** its surface summary, diagnostics, and ordered action evidence match within the declared numerical tolerance

#### Scenario: Resource budget is insufficient
- **WHEN** a requested evaluation exceeds the configured memory budget
- **THEN** the system uses a declared bounded-memory execution plan or returns a structured resource failure without silently reducing requested semantics or precision
