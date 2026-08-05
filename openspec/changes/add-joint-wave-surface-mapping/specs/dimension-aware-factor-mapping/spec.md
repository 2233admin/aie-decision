## Purpose

为多单位变量和多轴未来结果提供可验证、可追溯的映射契约，使非法量纲计算在求值前失败，并使不同来源的映射能够安全组合。

## ADDED Requirements

### Requirement: Multi-axis outcome and variable declarations
The system SHALL accept an outcome space containing one or more named axes and variables whose native unit, domain, time semantics, evidence state, provenance, and schema version are explicit.

#### Scenario: Declare heterogeneous outcome axes
- **WHEN** an Agent declares time in days, price in USD, and intensity as dimensionless axes
- **THEN** the system preserves each axis independently and returns their canonical units and domains without coercing them into one scalar

#### Scenario: Missing variable remains unknown
- **WHEN** a required variable has no observation, assumption, bound, or declared distribution
- **THEN** the system records it as unknown and MUST NOT replace it with zero, a midpoint, or another point estimate

### Requirement: Dimensional validation before evaluation
The system MUST validate every mapping against declared units and dimensions before sampling or scoring any surface.

#### Scenario: Compatible conversion succeeds
- **WHEN** a mapping combines distances expressed in kilometres and metres through a dimensionally valid operation
- **THEN** the system converts them to the mapping's canonical unit and accepts the mapping

#### Scenario: Incompatible addition fails
- **WHEN** a mapping directly adds a duration to a price
- **THEN** the system rejects the mapping with a structured error identifying the mapping, operands, units, and failed operation

### Requirement: Versioned factor mapping contract
The system SHALL represent each mapping as a versioned contract linking one or more variables to one or more outcome axes, with formula semantics, parameters, applicability conditions, provenance, and evidence references.

#### Scenario: Mapping is traceable
- **WHEN** a mapping contributes support to a surface
- **THEN** the output identifies the exact mapping version, input variable versions, parameters, evidence references, and affected outcome axes

#### Scenario: Unsupported mapping version
- **WHEN** an input uses an unknown incompatible mapping schema version
- **THEN** the system fails before evaluation and reports the supported migration boundary

### Requirement: Comparable factor contribution
The system MUST convert accepted mapping contributions to a dimensionless support score before combining multiple mappings, and MUST expose the semantic basis of that score.

#### Scenario: Heterogeneous mappings combine
- **WHEN** valid time, price, and categorical mappings contribute to the same outcome space
- **THEN** their contributions are combined only after each has produced a dimensionless score with declared direction and weight semantics

#### Scenario: Score semantics are absent
- **WHEN** a mapping produces a dimensional value or an unexplained arbitrary score
- **THEN** the system rejects it rather than silently normalizing it

