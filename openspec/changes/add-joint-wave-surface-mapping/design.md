## Context

See `proposal.md` for motivation. The current Python runtime already has scalar interval propagation, a closed-loop candidate controller, heuristic candidate mutation, ablation, an append-only ledger, and replay. The implementation must preserve those paths while introducing multi-axis output and dimension-aware mappings. It must first produce a small deterministic CPU loop; high-throughput GPU work is an adapter to the existing GPU batch-evaluation change.

The repository's current Code Intel failure is a tooling compatibility issue: a legacy flat `sentrux-lite` baseline occupies the path now reserved for native schema v2, and an old beta.1 script checkout has been used with a beta.5 binary. This is an implementation prerequisite, not an application architecture failure.

## Goals / Non-Goals

**Goals:**

- Introduce a versioned graph from variables and evidence through mappings to a multi-axis surface.
- Make unit validation, semantics gating, provenance, failure behavior and deterministic replay first-class.
- Use one evaluator protocol so CPU and GPU paths can be compared and generators remain replaceable.
- Turn surface diagnostics into typed search-loop actions.

**Non-Goals:**

- Replace the scalar Fermi estimator or current search controller.
- Implement causal discovery, a general symbolic algebra system, dense high-dimensional grids, or continuous PDE solvers.
- Make GPU availability a prerequisite for the CPU vertical slice.

## Data Flow

```text
Question + decision policy
        |
        v
OutcomeSpace ---- VariableSpec + Evidence
        |                 |
        +------ MappingSpec candidates <--- generators
                         |
                  validate units/schema
                         |
                   compile FactorIR
                         |
              deterministic particle plan
                         |
              batch log-potential evaluation
                         |
                  ParticleSurface
                         |
          marginals + modes + diagnostics
                         |
                DecisionValuePolicy
                         |
          stop OR typed refinement action
                         |
             existing search controller
                         +-------- loop --------+
```

## Decisions

### 1. Use a bipartite factor graph as the mapping model

`VariableSpec` and `OutcomeAxis` are nodes; `MappingSpec` is a hyperedge that can connect multiple inputs and outputs. A mapping compiles to a restricted `FactorIR` with a dimensionless `log_potential` output.

This supports one-to-many, many-to-one and interaction mappings without forcing heterogeneous units into a vector addition. The restricted IR also gives validation, serialization and backend parity.

Alternative considered: retain arbitrary formula strings. Rejected because strings cannot reliably guarantee dimensional safety, backend equivalence or safe execution.

### 2. Use Pint for unit authority

Pint owns parsing, conversion and dimensional compatibility. Domain schemas store unit strings and canonical dimension signatures; application code does not implement its own conversion table.

Alternative considered: normalize all numbers to floats at ingestion. Rejected because it loses the distinction between compatible conversion and illegal operations.

### 3. Use weighted particles rather than a dense joint grid

`ParticleSurface` stores `values[particle, axis]`, `log_weight[particle]`, stable identifiers and compact diagnostics. Sampling starts with a seeded, deterministic Sobol/QMC plan. Marginals and regions are derived from particles.

This preserves multiple modes and scales with particle count rather than exponentially with axes. Evaluation is chunked across candidates and particles; it never allocates the full candidate × variable × particle cube.

Alternative considered: a dense Cartesian grid. Rejected because memory grows exponentially and fails the large-factor use case before useful evaluation begins.

### 4. Separate surface semantics from numerical representation

The same particle container can represent either `possibility_surface` or `probability_surface`. A `SemanticsGate` inspects input distributions, factor weight meanings and calibration evidence. Any missing basis downgrades to possibility; it never silently upgrades to probability.

This keeps the MVP honest while leaving room for later empirical calibration. Existing subjective intervals retain `subjective_credible_interval`, `declared_joint_input_region`, and `calibration: unmeasured` metadata.

Alternative considered: call every normalized weight a posterior probability. Rejected because normalization alone does not supply calibrated probability semantics.

### 5. Define usefulness through a DecisionValuePolicy

Each axis can provide absolute tolerance, relative tolerance where the reference is nonzero, or a loss function. The policy also checks joint constraints, calibration gates, effective sample size and multimodality. It returns a structured pass/fail decision with per-criterion evidence.

This resolves the width problem: two hours can be acceptable for a 25-hour planning estimate but unacceptable for a short deadline; price usefulness can incorporate a volatility-aware loss rather than a universal percentage.

Alternative considered: one global normalized width formula. Rejected because it is undefined near zero and ignores the action being supported.

### 6. Use diagnostics to select typed refinement actions

The diagnostic-to-action policy uses:

- high sensitivity plus uncertain observable input → `measure`;
- weakly explained residual or cross-axis dependence → `add_interaction` or `expand_variable`;
- separated modes or unstable assignments → `split_regime`;
- low contribution under ablation → `minimize`;
- passed decision policy → `stop`.

Actions include evidence, expected reduction in decision loss, estimated cost and affected identifiers. The current candidate generator consumes actions but does not score its own candidates.

Alternative considered: always ask an LLM to try another decomposition. Rejected as the sole mechanism because it conflates proposal generation with validation.

### 7. Keep one evaluator protocol with CPU as reference

Define a `WaveCandidateEvaluator` boundary accepting versioned candidate batches, a particle plan and a resource budget, returning surfaces or structured failures. The CPU implementation is deterministic and is the numerical oracle. The GPU implementation compiles the same `FactorIR` to PyTorch operations and plugs into the existing batch evaluator.

GPU batching is progressive: cheap, small particle screens remove poor candidates before expensive refinement. Chunk sizes follow explicit memory budgets; OOM produces a smaller declared batch or a structured failure, never a semantic change.

Alternative considered: build a separate CUDA-first controller. Rejected because it duplicates scheduling, ledger and fallback logic and would delay the first closed loop.

### 8. Preserve traceability through content identities and the existing ledger

Outcome spaces, variables, evidence, mappings, particle plans, evaluator versions and decision policies receive stable versioned identifiers. Every state transition records input hashes, seed, budget, diagnostics, action ranking and accepted surface identifier. Large particle arrays may live in content-addressed artifacts; the ledger retains hashes and summaries.

Replay requires matching semantic versions and seeds. A version mismatch is reported rather than being called deterministic replay.

## State Transitions

```text
DRAFT
  -> VALIDATED        mapping/schema/unit gates pass
  -> REJECTED         validation fails

VALIDATED
  -> EVALUATED        surface and diagnostics produced
  -> FAILED           numeric/resource failure

EVALUATED
  -> ACCEPTED         decision-value policy passes
  -> REFINING         a typed action is selected
  -> UNRESOLVED       budget or marginal-value stop

REFINING
  -> DRAFT            new candidate set is generated
```

No transition may turn `UNKNOWN`, `FAILED` or `UNRESOLVED` into `ACCEPTED` without a new recorded evaluation.

## Versioning and Compatibility

- Add independent schema versions for outcome space, variable, mapping, particle surface, diagnostics, decision policy and ledger events.
- Minor versions may add optional fields with deterministic defaults; semantic or required-field changes require a migration or rejection.
- Preserve the current scalar Fermi and search input contracts. The wave path is selected explicitly and uses an adapter into the existing controller.
- CPU/GPU parity fixtures pin schema, factor IR, seed, dtype and tolerance.

## Failure Semantics

- Schema, unit, unsafe formula and unsupported-version failures occur before evaluation and identify the responsible entity.
- NaN, infinity, invalid domains, zero total support and low effective sample size produce invalid/unreliable surfaces, never sharp successful surfaces.
- Missing variables remain unknown and can generate a measurement or expansion action.
- Resource exhaustion records the attempted plan and safe degradation. If the requested contract cannot be met, evaluation fails explicitly.
- A generator failure does not invalidate already evaluated candidates; it is recorded as a candidate-source failure.

## Code Intel Toolchain Migration Plan

1. Pin automation to the current canonical Code Intel checkout and installed beta.5-compatible launchers; remove use of the stale beta.1 script path from project commands.
2. Run the canonical doctor and restore the required `repowise` executable or configuration before claiming the pipeline healthy.
3. Preserve the legacy `.sentrux/baseline.json` as a recoverable audit artifact and record its hash, tool (`sentrux-lite`) and metrics.
4. Run native scan and rule check without ratchet. Only if they pass and a reviewer accepts the current debt may the native engine write a v2 baseline.
5. **Verify lite sessions write only `.sentrux/cache/lite-baseline.json` and cannot overwrite the native baseline.**
   - **STATUS: ESCALATED.** Canonical beta.5 `Invoke-SentruxAgentTool.ps1`
     calls native `sentrux gate --save` which hard-codes
     `.sentrux/baseline.json`.  No repo-level extension surface exists to
     redirect this write.  See `docs/CODE-INTEL-TOOLCHAIN-MIGRATION.md`
     "Lite-session boundary" for precise upstream file/interface gaps.
6. **Rerun the full Code Intel pipeline and require a non-`domain_failed` result with recorded artifact directory.**
   - **DONE.** Fixed with (a) `.gitignore` exclusions for transient dirs
     (`.codex/`, `.omc/`, `.sentrux/agent-sessions/`) to stabilize
     `ExplicitOverlay` snapshot identity; (b) `CODE_INTEL_INTEGRATIONS_MANIFEST`
     env var pointing to the correct release manifest to fix pipeline root
     resolution.  Verified: `code-intel . --mode normal --json` → exit 0,
     outcome `completed`, empty domain/process failures.
     Tests: `tests/test_code_intel_pipeline.py` (4/4 pass).

Rollback restores the preserved legacy artifact and previous project command configuration; it does not pretend the legacy baseline is native-compatible.

## Application Migration Plan

1. Add schemas and CPU reference primitives behind an explicit wave-surface entry point.
2. Add golden fixtures for units, multimodality, residual interactions, semantics and replay.
3. Add the evaluator adapter and diagnostic policy to the existing controller while retaining scalar behavior.
4. Expose Agent/CLI output only after schema and golden tests pass.
5. Add GPU compilation and parity benchmarks after the CPU closed loop is accepted.

Rollback disables the new entry point and leaves scalar Fermi, existing candidate search and all ledger data intact.

## Risks / Trade-offs

- [Particle approximation misses a narrow or rare mode] → use deterministic stratified plans, effective-sample diagnostics, mode fixtures and progressive refinement.
- [Many factors produce numerical underflow] → accumulate dimensionless support in log space and use stable normalization.
- [Heuristic weights appear probabilistic] → enforce the semantics gate and make calibration status visible in every summary.
- [Unit libraries accept ambiguous aliases] → maintain a small reviewed domain-unit policy on top of Pint and reject ambiguous inputs.
- [GPU and CPU diverge] → pin dtype/tolerance, compare the same IR and particles, and keep CPU as acceptance authority.
- [Version migration invalidates replay] → hash schemas/evaluator versions and label cross-version re-execution as migration, not replay.
- [New baseline hides existing quality debt] → require review and preserve the old baseline before any native ratchet initialization.

## Open Questions

- The default particle counts and screening ratios can be tuned from benchmarks without changing the public behavior contract.
- The first domain-specific decision-loss presets can be selected after generic absolute, relative and callable policies are working.
