## Purpose

Provide a stateful, replayable tool loop in which an external AI can perform recursive Fermi decomposition while deterministic runtime checks govern accepted state and stopping claims.

## ADDED Requirements

### Requirement: The runtime exposes decomposition actions as tools
The system SHALL expose machine-readable actions for starting a question, inspecting state, expanding a node, proposing an alternative, estimating a leaf, declaring dependence, evaluating a branch, pruning or rolling back, and finalizing. The first version MUST be callable through a simple Python- or PowerShell-accessible process interface.

#### Scenario: Discover available actions
- **WHEN** an AI client starts the tool without repository-specific decomposition data
- **THEN** it can inspect action names, required fields, validation errors, and current legal next actions

### Requirement: The AI chooses semantic actions and the runtime settles them
The external AI SHALL choose semantic decomposition actions. The runtime SHALL validate, execute, accept, reject, prioritize, or roll them back; generated prose alone MUST NOT mutate analysis state or prove atomicity, minimality, uncertainty, or completion.

#### Scenario: AI claims completion in prose
- **WHEN** the AI states that a branch is minimal without calling the required intervention and evaluation actions
- **THEN** the runtime state remains uncertified

### Requirement: State changes are append-only and replayable
Every attempted action and runtime result SHALL be appended to an ordered trajectory with input payload, validation outcome, state revision, and deterministic calculation controls. Replaying accepted events MUST reconstruct the same visible analysis state.

#### Scenario: Replay an analysis
- **WHEN** a client replays the stored trajectory from the original question
- **THEN** it reconstructs the same tree, branch statuses, intervals, frontier tests, and final state

### Requirement: Rejected and rolled-back actions preserve history
The runtime SHALL keep rejected or rolled-back actions in the trajectory while excluding their state effects from the current visible tree.

#### Scenario: Invalid expansion is corrected
- **WHEN** an expansion fails unit validation and the AI later submits a valid replacement
- **THEN** the trajectory contains both attempts while only the valid expansion appears in current state

### Requirement: Search budgets cannot masquerade as completion
The runtime SHALL support turn, action, depth, and computation budgets. Exhausting a budget MUST produce a partial result with active frontier nodes and gaps, not a certified answer.

#### Scenario: Action budget is exhausted
- **WHEN** the AI reaches its configured action budget before frontier certification
- **THEN** the system stops with a partial status and identifies the highest-priority unfinished node

### Requirement: Provider and material formats are outside the core
The runtime SHALL NOT require a specific model provider, document format, crawler, or evidence-retrieval system. It SHALL accept model actions and scoped estimates through the same contracts regardless of how the AI obtained them.

#### Scenario: Two AI hosts use the runtime
- **WHEN** different AI hosts submit equivalent valid actions
- **THEN** the deterministic runtime produces equivalent state and calculations

### Requirement: Cold-start acceptance uses a real AI trajectory
End-to-end acceptance SHALL give a real AI only a raw quantitative question plus access to the tool interface. The accepted trace MUST show recursive expansion, concrete atoms, runtime feedback, uncertainty-guided continuation, intervention-based frontier testing, and a final or honest partial result.

#### Scenario: AI solves a novel question
- **WHEN** a real AI receives a quantitative question not represented by a built-in formula or fixture
- **THEN** its recorded tool trajectory produces an auditable decomposition without a user-supplied equation or variable set

### Requirement: Product claims follow executable state
Human and machine outputs SHALL distinguish product direction, implemented behavior, structural completeness, answerability, frontier certification, and calibration. Documentation or schemas alone MUST NOT be used as evidence that the AI can perform the workflow.

#### Scenario: Interface exists but no real trajectory passes
- **WHEN** actions and schemas are implemented but cold-start acceptance has not produced the required business result
- **THEN** the product remains unverified rather than being reported complete
