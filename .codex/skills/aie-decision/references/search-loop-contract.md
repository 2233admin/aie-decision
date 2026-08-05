# Closed-loop search input contract

Use this object shape for `aie-decision search-fermi`:

```json
{
  "run_id": "stable-run-id",
  "question": "quantitative question",
  "target": "machine_target_name",
  "unit": "target/unit",
  "coverage": 0.9,
  "reference_value": 100,
  "acceptable_width": 40,
  "variables": [
    {
      "name": "factor_name",
      "lower": 0,
      "upper": 1,
      "method": "observed|assumed|decision",
      "ablatable": false
    }
  ],
  "candidates": [
    {
      "candidate_id": "seed",
      "formula": "factor_name",
      "mutation_kind": "seed",
      "prior_weight": 1.0
    }
  ],
  "mutation_templates": [
    {
      "template_id": "replace-seed",
      "formula_template": "factor_a * factor_b + removable_adjustment",
      "diagnostic_reasons": ["interval_too_wide"],
      "mutation_kind": "revise",
      "prior_multiplier": 0.8
    }
  ],
  "budget": {
    "max_candidates": 20,
    "max_rounds": 8,
    "max_evaluations": 20,
    "max_seconds": 30
  }
}
```

Rules:

- Formulas may use only declared variables and the runtime's safe arithmetic.
- Set `ablatable: true` only when removing the variable is semantically valid.
- Do not precompute or invent a successful candidate. Let diagnostics activate
  the declared mutation template and let the runtime evaluate it.
- `result-found` means the experimental width/minimality gates passed.
  `budget-exhausted` and `insufficient-information` are honest terminal states.
- Preserve `provisional_uncalibrated`, `calibration: unmeasured`, and
  `usable: false` exactly as reported.
