## Purpose

Turn an initially unanswerable quantitative question into a traceable tree of concrete, scoped, measurable quantities without relying on fixed domains or user-supplied formulas.

## ADDED Requirements

### Requirement: A raw question becomes the decomposition root
The system SHALL start a decomposition from a raw quantitative question and SHALL NOT require the user to supply a formula, variable list, bounds table, or domain label. Before expansion, the root MUST record the target quantity, unit, population or object scope, time scope, and any unresolved ambiguity.

#### Scenario: Start with only a question
- **WHEN** an AI starts a session with a raw question and no predeclared variables
- **THEN** the system creates a target root and reports the target fields that are known or still need resolution

### Requirement: Decomposition is recursive and relationship-preserving
The system SHALL let an AI expand any unresolved node into children connected by an executable relationship. An accepted expansion MUST preserve the parent meaning, produce the parent unit, identify every child dependency, and state why the children are more estimable than the parent.

#### Scenario: Expand an abstract target
- **WHEN** an AI expands a target into a population count, event frequency, and value per event
- **THEN** the system executes the relationship, verifies unit closure, and stores the children as the next decomposition frontier

#### Scenario: Expansion has incompatible units
- **WHEN** the child relationship cannot produce the declared parent unit
- **THEN** the system rejects the expansion and leaves the prior tree unchanged

### Requirement: Leaves represent concrete measurable atoms
The system SHALL accept a node as an atomic leaf only when it identifies a real-world object or event, unit, population, time and geographic scope where applicable, and a feasible observation, counting, survey, reference-class, or measurement procedure. An abstract label without an operational measurement procedure MUST remain unresolved.

#### Scenario: Abstract label is submitted as a leaf
- **WHEN** an AI marks a phrase such as demand strength or operating efficiency as atomic without a concrete measurement procedure
- **THEN** the system rejects the atomic status and keeps the node on the expansion frontier

#### Scenario: Concrete atom is submitted
- **WHEN** an AI defines a leaf as the number of specified people performing a specified event per day in a specified population with a counting procedure
- **THEN** the system accepts its structural atomicity and records how the quantity can be measured

### Requirement: Alternative paths are adaptive search branches
The system SHALL support alternative decompositions of any node without imposing a fixed candidate count. Each alternative MUST identify its distinguishing observable quantities and remain attached to the same parent target for later comparison.

#### Scenario: A second decomposition becomes useful
- **WHEN** the current branch contains an unbounded or dominant leaf and the AI proposes a different measurable identity for the same node
- **THEN** the system records the alternative and makes both branches available for uncertainty and answerability comparison

#### Scenario: Formula rewrite adds no information
- **WHEN** a proposed alternative is algebraically equivalent and has the same observable leaves as an existing branch
- **THEN** the system records it as redundant rather than counting it as a distinct search branch

### Requirement: Domain templates cannot settle decomposition
The system MUST NOT infer completion from keyword matching or select a hard-coded industry formula as the answer. Domain-specific identities MAY be proposed by an AI, but they MUST pass the same structural, atomicity, uncertainty, and frontier tests as every other branch.

#### Scenario: Familiar domain keywords are present
- **WHEN** a question contains words that match a known operational domain
- **THEN** the system does not auto-complete a preset formula and instead waits for validated decomposition actions

### Requirement: Unresolved abstraction remains visible
The system SHALL represent an unresolved node, missing relationship, unknown leaf, or semantic mismatch as a first-class gap and SHALL NOT fabricate a numeric value to make a branch executable.

#### Scenario: No defensible atom can be produced
- **WHEN** the AI cannot turn a required abstract node into a measurable quantity
- **THEN** the system returns a non-answerable branch with the unresolved node and its impact on the target

### Requirement: Structural results are inspectable
The system SHALL expose the selected tree, alternative branches, node scopes, relationships, units, atom measurement procedures, rejected actions, and unresolved gaps in a machine-readable form.

#### Scenario: Inspect a decomposition
- **WHEN** a client requests the current decomposition state
- **THEN** it receives enough structured information to reconstruct every accepted parent-child relationship and every active frontier node
