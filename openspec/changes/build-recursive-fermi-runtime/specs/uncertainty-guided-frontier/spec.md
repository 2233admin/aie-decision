## Purpose

Use honest target uncertainty to decide where decomposition effort is valuable and to demonstrate the point where fewer atoms materially damage the answer while further refinement has immaterial benefit.

## ADDED Requirements

### Requirement: Every leaf has probability semantics or an explicit gap
The system SHALL record each leaf as a constant, an evidence- or assumption-based probability distribution, or unknown. A numeric range MUST NOT be treated as a 90 percent probability interval unless its probability semantics are explicit.

#### Scenario: Leaf evidence lacks probability meaning
- **WHEN** an AI provides a lower and upper value without a defensible probability interpretation
- **THEN** the system records the range as non-probabilistic or unknown and does not use it to claim target 90 percent coverage

### Requirement: Target probability comes from a declared joint model
The system SHALL propagate leaf uncertainty through the selected relationship using an explicit joint model. The result MUST disclose marginal models, dependence assumptions, sample or analytic method, reproducibility controls, and calibration status.

#### Scenario: Multiple uncertain leaves are multiplied
- **WHEN** a branch contains several marginal 90 percent intervals
- **THEN** the system derives target P05 and P95 from the declared joint model rather than labeling the Cartesian endpoint range as a target 90 percent interval

#### Scenario: Dependence is unknown
- **WHEN** leaf dependencies could materially change the result but are not established
- **THEN** the system reports the dependency gap and evaluates declared sensitivity cases before accepting answerability

### Requirement: Joint model v1 scope is independence or one global equicorrelation
The v1 joint model SHALL support only two declared dependence cases applied to every non-constant, non-unknown sampled leaf:

* ``independent`` — sampled leaves receive independent uniforms.
* ``positive`` or ``negative`` — sampled leaves receive correlated uniforms via a single Gaussian copula whose off-diagonal entries are the declared ``correlation`` (sign chosen by the case).

A ``rationale`` string is REQUIRED for every non-trivial case (``positive``, ``negative``, or any deviation from independence) and the system SHALL refuse the action when the rationale is empty. The copula dimension excludes constant leaves and unknown leaves; only the sampled leaves share the equicorrelation. When ``negative`` equicorrelation would produce a non-positive-definite matrix the system SHALL reject the model before any sampling attempt with a clear, deterministic error.

The system SHALL NOT accept arbitrary correlation matrices, pairwise dependence graphs, conditional dependence graphs, or per-pair correlation overrides in v1. Requests for richer dependence MUST be rejected at validation time and reported as a documented gap (``dependence_unsupported_v1``) so the AI can either supply the rationale for one global equicorrelation, declare independence, or downgrade the joint model to scenario bounds.

#### Scenario: AI requests an unsupported joint structure
- **WHEN** the AI submits a joint model that specifies a custom correlation matrix, a conditional graph, or per-pair overrides
- **THEN** the system rejects the action, surfaces ``dependence_unsupported_v1`` as a gap, and exposes the actual joint structure it fell back to so the AI can revise the model

#### Scenario: Rationale is missing for non-independent dependence
- **WHEN** the AI declares ``positive`` or ``negative`` dependence without supplying a rationale
- **THEN** the system rejects the action with a ``rationale_required`` error and leaves the previous joint model in place

### Requirement: Uncertainty contribution controls search priority
The system SHALL estimate how much each unresolved or uncertain node contributes to the target interval width and SHALL return the node with the highest expected value from further decomposition or measurement.

#### Scenario: One node dominates width
- **WHEN** one leaf or unresolved node accounts for the largest reducible share of target uncertainty
- **THEN** the system recommends that node as the next expansion or measurement target and quantifies the expected width reduction

### Requirement: Answerability uses an explicit target tolerance
Before declaring a branch answerable, the system SHALL record a task-specific acceptable target interval width or equivalent decision-stability criterion. The AI MAY derive this criterion from the question, but it MUST be explicit and inspectable rather than a hidden global default.

#### Scenario: Interval is too wide
- **WHEN** the target 90 percent interval exceeds the recorded acceptable width
- **THEN** the branch remains insufficient and the system returns the next highest-value refinement target

#### Scenario: No tolerance can be justified
- **WHEN** neither the question nor the AI can justify a useful width or decision criterion
- **THEN** the system reports that sufficiency cannot yet be assessed

### Requirement: Necessity is demonstrated by destructive interventions
For every leaf on a proposed final frontier, the system SHALL execute deletion and at least one applicable coarsening or same-target substitution test. A retained leaf is locally necessary only when removing or coarsening it makes the branch non-computable or causes a recorded material degradation in width or decision stability.

#### Scenario: Removing a leaf has little effect
- **WHEN** deleting or coarsening a leaf leaves the target answerable within the material-degradation threshold
- **THEN** the system marks the leaf redundant and refuses to certify the frontier as minimal

#### Scenario: Removing a leaf destroys precision
- **WHEN** deleting a leaf makes the target non-computable or materially exceeds the accepted width
- **THEN** the system records the executed result as necessity evidence for that leaf

### Requirement: Saturation is demonstrated by refinement value
The system SHALL evaluate available refinements or additional atoms for frontier nodes and SHALL report whether their expected target-width improvement exceeds the recorded material-improvement threshold. It MUST describe saturation as conditional on the explored refinements, not as proof that no imaginable decomposition can improve the result.

#### Scenario: Another split has material value
- **WHEN** an available refinement is expected to narrow the target interval materially
- **THEN** the system keeps the search open and prioritizes that refinement

#### Scenario: Explored refinements add little value
- **WHEN** no explored refinement exceeds the material-improvement threshold and the answerability tolerance is met
- **THEN** the system records conditional saturation evidence for the current frontier

### Requirement: Minimal sufficient frontier requires sufficiency necessity and saturation
The system SHALL certify a minimal sufficient frontier only when answerability tolerance is met, every retained leaf has passed applicable necessity interventions, and explored refinements have passed the saturation test.

#### Scenario: Complete frontier certification
- **WHEN** sufficiency, necessity, and conditional saturation all pass with executable records
- **THEN** the system returns the frontier as certified together with every threshold and intervention result

#### Scenario: Structural completeness alone
- **WHEN** an expression is executable but one of the three frontier criteria has not passed
- **THEN** the system reports the branch as structurally complete but not frontier-certified

### Requirement: Final uncertainty output is honest
The system SHALL output target P05, P50, P95, 90 percent interval width, relative or normalized width where defined, dominant uncertainty contributors, next measurement or expansion, dependency assumptions, gaps, and empirical calibration status.

#### Scenario: No historical outcomes exist
- **WHEN** the target interval is model-derived without an outcome cohort
- **THEN** the system labels empirical calibration as unmeasured and does not claim achieved 90 percent coverage
