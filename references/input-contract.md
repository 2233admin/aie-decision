# Compiler input contract

Read this reference when constructing input for `aie-decision compile`. The compiler accepts one JSON object and never acquires external data.

## Required top-level content

`answer_contract` is required and contains:

- `question_id`, `question`, and `answer_type`
- `target.entity`, `target.measure`, and `target.unit`
- `observation_cutoff`
- Conditional and optional decision fields: `prediction_horizon` and `uncertainty_semantics` are required for `future_prediction`; `geography`, `decision_use`, `decision_thresholds`, `requested_coverage`, and `acceptable_width` are optional unless a stricter downstream contract requires them.

The following arrays are accepted. `sources`, `edges`, `minimal_sufficient_sets`, `missing_conditions`, and `derived_factors` may be empty when unavailable. A usable compile input needs at least one declared `condition`; an empty condition graph currently produces diagnostic validation errors rather than a valid partial package.

- `conditions`: each item requires `condition_id`, `name`, `value_type`, `necessity`, `status`, and `answer_impact`; `unit` is optional.
- `edges`: each item requires `from_id`, `to_id`, `relation`, and `answer_impact`.
- `minimal_sufficient_sets`: arrays of condition IDs.
- `sources`: each item requires `title` and `content`; optional attribution fields include `uri`, `publisher`, `published_at`, `retrieved_at`, `source_id`, `source_location`, `speaker`, `transformation_lineage`, and `target_relevance`.
- `missing_conditions`: each item requires `condition_id`, `estimate_type`, `lower`, `upper`, `unit`, `coverage_semantics`, `coverage`, `method`, and `dependence_case`; optional fields include `estimate_id`, `input_atom_ids`, `assumptions`, `calibration_profile_id`, and `valid_until`.
- `derived_factors`: each item requires `factor_id`, `name`, `input_condition_ids`, `hypothesis`, `observable_implications`, `falsification_conditions`, `unit`, `time_window`, and `target_paths`; `composition` is optional.

`forecast_interval`, when supplied, requires `target`, `horizon`, `unit`, `population`, `coverage_level`, `generation_method`, `reference_time`, `lower`, `upper`, and `reference_value`. It may also contain `conditional_assumptions`, `kind`, `baseline.lower`, `baseline.upper`, and `thresholds`.

## Minimal valid shape

```json
{
  "answer_contract": {
    "question_id": "q-example",
    "question": "What answer should be measured?",
    "answer_type": "current_observation",
    "target": {
      "entity": "example-entity",
      "measure": "example-measure",
      "unit": "units"
    },
    "observation_cutoff": "2026-08-01T00:00:00Z"
  },
  "conditions": [
    {
      "condition_id": "missing-value",
      "name": "current value",
      "value_type": "number",
      "necessity": "required",
      "status": "missing",
      "answer_impact": "the answer is unavailable without this observation",
      "unit": "units"
    }
  ],
  "edges": [],
  "minimal_sufficient_sets": []
}
```

This shape can legitimately compile to `status: partial`. Add only user-supplied evidence and declared estimates. Never manufacture content merely to obtain `status: complete`.

## Output contract

The compile command prints a JSON response containing `status`, `package`, `report`, `ledger`, and `validation_issues`. It returns exit code `0` only when `validation_issues` is empty; otherwise it returns `2`. A partial package can return `0`, while a written package with validation issues returns `2` and is diagnostic only.
