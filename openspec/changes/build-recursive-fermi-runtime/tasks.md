## 1. Contract and Parallel-Work Baseline

- [x] 1.1 Commit the approved product-direction correction and complete OpenSpec artifacts as the immutable base for implementer worktrees
- [x] 1.2 Define integration-level public function signatures and versioned JSON examples without implementing track-owned modules
- [x] 1.3 Create three isolated Claude Code worktrees from the same baseline and enforce the file ownership listed below

## 2. Claude Track A - Recursive Tree and Atomicity

Owned files: `src/aie_decision/fermi_contracts.py`, `src/aie_decision/decomposition_tree.py`, `tests/test_fermi_contracts.py`, `tests/test_decomposition_tree.py`.

- [x] 2.1 Implement strict question, node, expansion, atom, relationship, branch, and gap contracts
- [x] 2.2 Implement restricted relationship parsing, dependency completeness, dimensional validation, and redundant-alternative detection
- [x] 2.3 Implement immutable recursive expansion, alternatives, active frontier inspection, pruning projections, and structural state export
- [x] 2.4 Test raw-question roots, recursive valid expansion, abstract-leaf rejection, unit mismatch, unresolved gaps, and alternative branches
- [x] 2.5 Commit only Track A owned files and return the commit hash plus focused test evidence

## 3. Claude Track B - Uncertainty and Minimal Frontier

Owned files: `src/aie_decision/probability.py`, `src/aie_decision/frontier.py`, `tests/test_probability.py`, `tests/test_frontier.py`.

- [x] 3.1 Implement declared marginal distributions, explicit joint assumptions, reproducible sampling, and target P05/P50/P95 computation
- [x] 3.2 Implement dependency validation and sensitivity cases without treating marginal endpoint rectangles as target probability intervals
- [x] 3.3 Implement uncertainty contribution and expected target-width reduction ranking for expansion or measurement targets
- [x] 3.4 Implement sufficiency, executed deletion/coarsening/substitution necessity, conditional refinement saturation, and frontier certification
- [x] 3.5 Test unknown probability semantics, dependence sensitivity, redundant leaves, material removal degradation, useful refinements, and honest calibration state
- [x] 3.6 Commit only Track B owned files and return the commit hash plus focused test evidence

## 4. Claude Track C - Event Runtime and Process Interface

Owned files: `src/aie_decision/trajectory.py`, `src/aie_decision/agent_runtime.py`, `src/aie_decision/agent_cli.py`, `tests/test_trajectory.py`, `tests/test_agent_runtime.py`, `tests/test_agent_cli.py`.

- [x] 4.1 Implement append-only action/result events, revision checks, payload digests, accepted-state projection, rollback markers, and replay
- [x] 4.2 Implement legal-action discovery, structured accept/reject responses, budgets, partial termination, and certification gates delegated to injected kernels
- [x] 4.3 Implement a versioned JSON CLI usable from Python and PowerShell without embedding a model provider
- [x] 4.4 Test deterministic replay, rejected-action visibility, rollback projection, illegal sequencing, budget exhaustion, and JSON process behavior
- [x] 4.5 Commit only Track C owned files and return the commit hash plus focused test evidence

## 5. Host Review and Integration

- [x] 5.1 Review every Claude commit against its OpenSpec scenarios, reject out-of-scope files, and request corrections for unexecuted claims or fixed-domain logic
- [x] 5.2 Cherry-pick approved commits into the integration branch and resolve only boundary-level conflicts
- [x] 5.3 Implement the narrow integration adapter connecting tree actions, probability evaluation, frontier scheduling, trajectory projection, and CLI dispatch
- [x] 5.4 Add end-to-end deterministic tests for a novel non-throughput question from raw question through recursive atoms and frontier evaluation
- [x] 5.5 Remove the fixed throughput command, module, fixture, and tests while retaining only reusable deterministic infrastructure

## 6. Real-AI Acceptance

- [x] 6.1 Give a fresh Claude Code process a novel raw quantitative question and only the CLI/tool discovery entry point
- [x] 6.2 Capture a real trajectory showing multiple recursive expansions, concrete measurable atoms, runtime rejection or feedback, and uncertainty-guided next actions
- [x] 6.3 Verify the trajectory reaches a certified frontier or an honest partial result without a user-supplied formula, variables, bounds table, or built-in domain template
- [x] 6.4 Replay the captured trajectory and verify identical visible tree, calculations, frontier tests, and output status

## 7. Documentation and Delivery Gates

- [x] 7.1 Update README and package metadata only with behavior proven by executable and real-AI acceptance
- [x] 7.2 Run focused tests, full tests, OpenSpec strict validation, Code Intel, Sentrux session gates, and diff review
- [x] 7.3 Scan exact release files and full reachable Git history for restricted content and credentials
- [x] 7.4 Commit the implementation on the isolated branch, create a GitHub pull request, inspect the remote diff, and wait for CI
- [x] 7.5 Merge the approved pull request once, verify the remote default branch commit and CI, and leave the user's original worktree untouched
