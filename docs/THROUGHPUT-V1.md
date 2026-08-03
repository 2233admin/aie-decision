# Operational Throughput v1

This is the intentionally narrow first product slice.

Data flow:

1. Validate a raw question and attributed materials.
2. Extract only supported observations and source-stated P05/P95 intervals.
3. Generate capacity, utilization-adjusted capacity, demand, and bottleneck candidates.
4. Compare completeness and target semantics; prefer the complete actual-throughput candidate.
5. Execute one deletion intervention for every retained leaf.
6. Sample the declared marginal distributions under an explicit primary joint assumption.
7. Report target P05/P50/P95, width, dependence stress cases, gaps, and calibration state.
8. Resolve one uncertain leaf at a time in counterfactual reruns to rank the next measurement.

The main target interval is a conditional subjective probability interval. It is not a rectangle formed from marginal bounds and is not empirically calibrated without historical outcomes.

## Supported grammar

The first version recognizes operating-unit counts, per-unit hourly rates, operating hours, utilization with an explicitly stated 90% probability interval, and daily demand with the same probability semantics. Other domains and unqualified ranges fail closed.

## Minimality contract

A leaf is retained only after the selected evaluator is actually rerun without it. The audit records whether deletion made the target non-computable and whether a complete same-target substitute existed. Membership in a formula's name list is not accepted as evidence.
