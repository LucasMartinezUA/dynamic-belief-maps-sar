# C1 gate

```json
{
  "scientific_gate": "ROBUST",
  "administrative_status": "PENDING_B3_REFERENCE",
  "scientific_valid": true,
  "b3_nominal_status": {},
  "b3_reference_status": "PENDING_B3_REFERENCE",
  "note": "The C1 experimental verdict is derived from the sensitivity data alone. The B3 reference is an administrative/provenance dependency and is not evidence of experimental fragility; it is recorded separately and does not downgrade the scientific result.",
  "sensitivity_region_cells": 24,
  "w0_control_states": {
    "1": "EQUIVALENT",
    "7": "EQUIVALENT",
    "10": "EQUIVALENT"
  },
  "nominal_states": {
    "1": "IMPROVEMENT",
    "7": "IMPROVEMENT",
    "10": "IMPROVEMENT"
  },
  "non_nominal_non_adverse_fraction_excluding_w0": 1.0,
  "non_nominal_both_positive_fraction_excluding_w0": 0.9583333333333334,
  "practical_improvement_fraction_excluding_w0": 0.9583333333333334,
  "boundary_conditions": [
    {
      "condition_id": "w_025",
      "dataset": 7,
      "dPdet_mean": -0.0230879726989007,
      "dRMST_rel_mean": -0.010603065684023999,
      "state": "INCONCLUSIVE",
      "reading": "low-weight boundary; localized, not a hard regression"
    }
  ],
  "classification": "classify_paired_endpoints",
  "parameter_selection": "none; w=.5,p_d=.8,tau=10000 remain frozen",
  "no_posthoc_tuning": true
}
```
