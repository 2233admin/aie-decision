## Context

The current branch contains a deterministic interval utility and a fixed-domain estimator that preselects formulas. The direction baseline in `docs/FERMI-PRODUCT-DIRECTION.md` rejects that architecture. This change needs a new stateful pattern spanning contracts, tree operations, probability evaluation, search control, persistence, CLI integration, and real-agent acceptance.

The useful lesson from long-horizon agent runtimes is the separation between model policy and runtime settlement: the model selects the next semantic action; typed tools validate and execute it; append-only events preserve what happened; budgets, rejection, rollback, and replay keep long searches governable. The task-specific search prompt is not the reusable product.

## Goals / Non-Goals

**Goals:**

- Let an external AI recursively construct and revise a domain-neutral decomposition.
- Make structural validity, arithmetic, uncertainty, prioritization, and frontier certification executable.
- Use uncertainty contribution as feedback during decomposition rather than only at final output.
- Preserve every attempted action and reproduce visible state from its trajectory.
- Keep the first integration simple enough for a real AI to exercise from a shell.

**Non-Goals:**

- Embedding a particular model provider or research subsystem.
- Deterministically deciding the full semantic truth of an AI-proposed real-world mechanism.
- Supporting every probabilistic formalism in the first evaluator.
- Treating evidence ingestion, web acquisition, PDFs, or user-material schemas as the decomposition core.
- Claiming global mathematical minimality over decompositions that were never explored.

## Decisions

### 1. Use an event-sourced decomposition session

The persisted source of truth is an append-only sequence of typed actions and results. Current tree state is a projection. Rejection and rollback append events; they never rewrite history. Each event records a prior revision, resulting revision, canonical payload digest, and calculation controls.

Alternative considered: mutate a single JSON tree. Rejected because it cannot faithfully audit failed expansions, alternatives, pruning, or model-visible history.

### 2. Separate the decomposition policy from the deterministic kernel

The external AI decides what a node means, how it could be split, and which real-world mechanism is plausible. The kernel validates legal state transitions, required operational fields, restricted expressions, units, graph consistency, numerical models, and certification gates. The kernel returns legal next actions and a priority queue so feedback changes the AI's next step.

Alternative considered: encode a library of industry decompositions. Rejected because that reproduces the fixed-throughput failure and does not generalize.

### 3. Represent alternatives as branch-scoped DAG projections

Each target node can own alternative expansions. A branch selects one expansion per target path; common immutable atoms can be referenced without duplicating their event history. Branch comparison is therefore adaptive and can occur at any node, not only as a fixed top-level candidate list.

Alternative considered: independent complete formulas. Rejected because it loses recursive alternatives and makes shared subproblems difficult to inspect.

### 4. Make atomicity operational rather than purely linguistic

An atom contract requires object or event type, unit, scope, time basis, measurement procedure, and estimate status. These checks cannot prove the semantic truth of a mechanism, but they prevent an abstract label from becoming a leaf merely because the AI says it is one. A later material refinement can reopen an atom and the trajectory records that revision.

Alternative considered: ask a second prompt whether a leaf looks atomic. Rejected as the only gate because it remains an unexecuted opinion.

### 5. Use evaluator plugins behind one branch contract

V1 provides a restricted algebraic evaluator with dimensional analysis and joint Monte Carlo propagation for constant and quantile-fitted continuous leaves. The branch contract carries an evaluator kind so later Bayesian, state-transition, simulation, or other models can be added without changing tree actions or frontier semantics.

Alternative considered: implement every mathematical model in v1. Rejected because the core innovation is recursive decomposition and stopping feedback; one honest evaluator is enough for the vertical slice.

### 6. Define frontier certification from three executable gates

Sufficiency checks the declared acceptable target width or decision stability. Necessity executes deletion plus applicable coarsening or substitution per retained leaf. Saturation measures the expected benefit of explored refinements. Certification is conditional on explored alternatives and records both the acceptance threshold and observed effect.

Alternative considered: minimize leaf count. Rejected because a smaller but useless model is not sufficient, and a larger model can contain variables that do not improve the target.

### 7. Schedule by reducible uncertainty contribution

After any branch becomes numerically evaluable, the runtime attributes target width to uncertain or unresolved frontier nodes using reproducible counterfactual or conditional reruns. The next-action response ranks expansion and measurement targets by expected interval reduction. Structural gaps that prevent propagation receive explicit blocking priority.

Alternative considered: depth-first recursion. Rejected because it can over-decompose low-impact branches while leaving the dominant uncertainty untouched.

### 8. Start with a Python API and JSON CLI

The kernel exposes Python functions and a CLI that accepts and emits versioned JSON actions against a session file. This is directly callable from PowerShell and from an AI with shell tools, needs no server lifecycle, and makes cold-start traces easy to capture. A protocol adapter can be added later without moving product logic out of the kernel.

Alternative considered: make the first milestone an MCP server. Deferred because transport choice is not the product question and should not block validation of the decomposition loop.

### 9. Use isolated implementer worktrees and owned files

The implementation is split into contracts/tree, uncertainty/frontier, and session/CLI tracks. Each Claude Code instance works in its own branch and owns non-overlapping production and test files. Integration occurs only through reviewed commits against the contracts fixed in this design.

## Risks / Trade-offs

- [An AI can propose a semantically wrong but dimensionally valid identity] → Preserve mechanism rationale, alternatives, and real-agent review evidence; never claim deterministic semantic truth.
- [Saturation depends on which refinements were explored] → Label it conditional, record rejected and unexplored frontier nodes, and avoid global-minimum language.
- [Quantile-fitted marginals approximate the AI's beliefs] → Preserve submitted quantiles, disclose the fitted family, and keep evaluator plugins replaceable.
- [Event logs can grow during long searches] → Use immutable compact payloads and derived snapshots while retaining append-only authority.
- [A shell-driven AI may misuse the action sequence] → Return legal next actions and structured errors; cold-start acceptance must demonstrate recovery.
- [Parallel implementations can disagree on contracts] → Land OpenSpec and shared contracts first, then keep strict file ownership and host review before integration.

## Migration Plan

1. Land this OpenSpec and shared type contracts before parallel implementation.
2. Implement tree/actions, uncertainty/frontier, and session/CLI in isolated branches.
3. Review and integrate the three commits, then remove the fixed-domain entry point and fixture.
4. Run focused and full tests, structural checks, trajectory replay, and a novel-question real-AI acceptance.
5. Replace README status only after executable acceptance passes.
6. Publish through an isolated GitHub pull request after content and reachable-history scans; rollback is a revert of that merge.
