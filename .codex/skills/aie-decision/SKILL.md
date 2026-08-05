---
name: aie-decision
description: "Run the AIE Decision closed-loop Fermi candidate search from Claude Code, Codex, or another tool-using agent. Use for quantitative decomposition, bounded candidate revision, interval propagation, mechanical ablation, replayable search, or an explicit $aie-decision invocation. Process only supplied variables and evidence; do not acquire external data."
---

# AIE Decision

Use the installed `aie-decision` CLI as the authority. Do not replace a real run
with conversational arithmetic.

## Prepare the run

1. Read [references/search-loop-contract.md](references/search-loop-contract.md)
   before constructing input.
2. Use only values supplied by the user. Keep observed, assumed, and decision
   variables distinct.
3. Treat formulas and mutation templates as candidate hypotheses, not facts.
4. Keep run files in a dedicated directory and preserve existing files.
5. Verify the installed interface with `aie-decision --help`. The public CLI
   must list `search-fermi`; otherwise report the installation mismatch.

## Run the closed loop

Write the request JSON, then run:

```text
aie-decision search-fermi <input.json> --experimental --output <result.json>
```

The exact protocol names are mandatory:

- Top level: `run_id`, `question`, `target`, `unit`, `coverage`,
  `reference_value`, `acceptable_width`, `variables`, `candidates`,
  `mutation_templates`, and `budget`.
- `variables` is an array with `name`, `lower`, `upper`, `method`, and optional
  `ablatable`. Never use a name-keyed object or `kind`.
- `candidates` is an array with `candidate_id`, `formula`, `mutation_kind`, and
  optional `prior_weight`. Every initial candidate uses
  `mutation_kind: "seed"` and has no parent.
- `mutation_templates` uses `template_id`, `formula_template`,
  `diagnostic_reasons`, `mutation_kind`, and optional `prior_multiplier`.
  Use `revise` when replacing a coarse estimate; template kinds are only
  `expand` or `revise`. Runtime-generated removals use `ablate`.
- `budget` uses `max_candidates`, `max_rounds`, `max_evaluations`, and
  `max_seconds`.
- `--experimental` is a CLI flag, not an input field.

Set `ablatable: true` only when the supplied problem explicitly permits that
variable to be removed. Uncertainty alone never makes a factor removable.

## Interpret the result

Report the terminal state, rounds and evaluations used, selected formula,
remaining variables, interval, absolute width, candidate lineage, and replay
checkpoint. Preserve these boundaries exactly:

- `result-found` means the experimental width and minimality gates passed.
- `budget-exhausted` and `insufficient-information` are valid failures; do not
  rewrite them as success.
- `pseudo_posterior` is an uncalibrated search ranking score, not a statistical
  posterior probability.
- `experimental_usable: true` does not override `usable: false`.
- `calibration: unmeasured`, `uncalibrated_*`, and
  `provisional_uncalibrated` must remain visibly labeled.

The current loop uses declared templates and CPU interval arithmetic. Do not
claim model-generated proposals, GPU evaluation, external evidence retrieval,
or empirical calibration unless a later verified runtime actually performs it.

## Other public runtime commands

The installed CLI may also expose `discover`, `start`, `apply`, `inspect`,
`finalize`, and `replay` for the versioned agent decomposition protocol. Inspect
their live `--help` before use; do not infer payload fields from this search
contract.
