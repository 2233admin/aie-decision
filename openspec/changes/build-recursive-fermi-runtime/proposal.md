## Why

The repository currently presents a fixed-domain arithmetic prototype as a Fermi decomposition product, but it does not recursively turn an unanswerable question into concrete, measurable atoms. We need a real AI-operated decomposition runtime whose search is guided by target uncertainty and whose stopping point is demonstrated rather than asserted.

## What Changes

- **BREAKING**: Remove the fixed operational-throughput product path and its hard-coded candidate formulas.
- Add a domain-neutral decomposition tree whose nodes carry target semantics, units, scope, estimability, relationships, and uncertainty.
- Add executable actions for expanding an abstract node, proposing an alternative branch, estimating a leaf, revising a branch, and pruning or rolling back an unhelpful action.
- Add an uncertainty-guided scheduler that directs the AI toward nodes dominating target interval width.
- Define atomic leaves as concrete, scoped, measurable real-world quantities rather than persuasive labels.
- Find a minimal sufficient frontier by executed deletion, coarsening, substitution, and refinement-value tests: one fewer material leaf degrades answerability, while one more refinement has immaterial expected benefit.
- Propagate an explicit joint uncertainty model and report target P05/P50/P95, interval width, dependency assumptions, gaps, and sensitivity.
- Preserve an append-only, replayable trajectory of model actions, tool validation, calculations, rollbacks, and stopping evidence.
- Prove the vertical slice with a real AI starting from a raw question, without a user-supplied formula, variable set, or interval table.

## Capabilities

### New Capabilities

- `recursive-fermi-decomposition`: Domain-neutral recursive decomposition trees, measurable-atom contracts, alternative branches, and executable structural validation.
- `uncertainty-guided-frontier`: Joint interval propagation, uncertainty contribution, search priority, measurable stopping rules, and minimal-sufficient-frontier experiments.
- `agent-decomposition-runtime`: Stateful tool actions, append-only trajectories, rollback/replay, simple process integration, and real-AI cold-start acceptance.

### Modified Capabilities

None.

## Impact

- Replaces the current throughput-specific product entry point, tests, fixture, and packaging claims.
- Adds typed runtime modules and focused tests under `src/aie_decision` and `tests`.
- Retains deterministic expression and interval functionality only where it serves the new runtime.
- Uses a simple Python or PowerShell-accessible process boundary for v1; model provider and user-material formats remain outside the core contract.
- Establishes `docs/FERMI-PRODUCT-DIRECTION.md` as required product context for implementation and acceptance.
