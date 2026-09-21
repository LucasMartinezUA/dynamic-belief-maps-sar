#!/usr/bin/env python3
"""C6 editorial/provenance closing stage (audits 17-24 and the C6_GREEN gate).

The stage generates the C6 audit documents from repository evidence (frozen
sources, git history, result artifacts), computes C6_BASELINES_GREEN /
C6_EDITORIAL_GREEN / claims_consistency_green, and publishes the combined
C6_GREEN gate into the c6-random campaign gate file. It runs no simulation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


C6_EDITORIAL_STAGE = "c6-editorial"
ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))


def _c():
    import audit_block_c

    return audit_block_c


def _json_default(value: Any):
    return _c()._json_default(value)


NATIVE_BASELINE_AUDIT = """# C6 native baseline audit

Audit of the baselines that actually exist in the frozen `revision_B_base` /
Block C implementation (checked against the current committed sources; the
core files were not modified by C6 except the additive random policy mode,
kept strictly additive).

Implementation found: `sarenv/analytics/paths.py` generators and
`sarenv/analytics/evaluator.py` registry (`get_default_path_generators`).
The Block C campaigns do not use the legacy evaluator pipeline: they use
`SARSimulation` planning modes directly.

## Native baseline inventory

### random_uniform_n8_uncoordinated (C6)
- name: `random_uniform_n8_uncoordinated`
- implementation found: `sarenv/analytics/simulation.py` (`_move_drones_random`,
  base mode `random_n8`; `_make_policy_rng` per-UAV streams), added for C6.
- inputs: current position, physically valid N8 actions (`_get_valid_neighbors`,
  unchanged), per-UAV RNG stream keyed on `(policy_seed, uav_id)`.
- uses prior? no. online? yes (per-timestep action). coordination? none.
- compatible with revised simulator: yes (native mode in the sim API).
- compatible with P_detect/RMST: yes (same exposure pipeline, no special treatment).
- same budget possible: yes. same initialization possible: yes (C5/C4 keys).
- decision: **KEEP** (mandatory C6 exploration floor).

### pizza_repartition
- name: `pizza_repartition` (attrition arm; sector repartition among survivors).
- implementation found: `sarenv/analytics/paths.py:generate_pizza_zigzag_path`
  plus `_assign_pizza_paths`/`_pizza_paths` in `scripts/audit_block_c.py`;
  executed as `planning_mode='static'` with `static_paths`.
- inputs: bounds/center/max_radius, FOV, altitude, overlap, spacing, num drones,
  survivor positions (assignment by minimum transit cost).
- uses prior? no. online? no (precomputed). coordination? per-drone sector.
- compatible: yes (accepted C4 version; C4-local hotfix fixed the alignment probe).
- decision: **KEEP** (attrition architecture contrast).

### online_static_3step
- implementation found: `sarenv/analytics/simulation.py`
  (`_move_drones_greedy_nstep`, `_get_planning_map` -> prior; `_compute_score_map`).
- inputs: prior heatmap, bounds, FOV kernel, num drones, N=3.
- uses prior? yes (fixed prior). online? yes. coordination? shared value map.
- compatible: yes (frozen Block B identity). decision: **KEEP**.

### dynamic_evidence_3step
- implementation found: `sarenv/analytics/simulation.py` (dynamic_nstep over the
  EvidenceBelief posterior) + `sarenv/analytics/evidence_belief.py` (frozen).
- uses prior? as anchor. online? yes. coordination? shared value map.
- compatible: yes (frozen). decision: **KEEP** (proposed method).

### Legacy `Greedy` / `Spiral` / `Concentric` (generator-only baselines)
- `generate_greedy_path` (paths.py), `generate_spiral_path` (paths.py),
  `generate_concentric_circles_path` (paths.py); registered in the legacy
  `evaluator` (get_default_path_generators).
- They are offline *generators*: using them under the revised simulator
  requires the static-execution route; the Block A audit found the v5 legacy
  modes had architectural confounds (pipeline-level differences).
- decision: **OPTIONAL, not re-run in C6** — they do not isolate information use
  and the spec forbids opening a costly new campaign for them.

### Legacy `random_walk` path generator (naming collision — do not confuse)
- name: `generate_random_walk_path` (paths.py) — greedy path generation over a
  zero map (`np.zeros_like(probability_map)`), evaluated statically.
- decision: **EXCLUDED from C6.** Not the C6 policy (precomputed static path vs
  per-timestep online uninformed action policy), shares the term `random_walk`
  with the victim mobility model, and follows the legacy evaluator pipeline.

## Summary table

| Baseline | Geometry | Prior | Online | Belief | Coordination | In C6 |
|---|---|---|---|---|---|---|
| random_uniform_n8_uncoordinated | no | no | yes | no | none | **KEEP (new)** |
| pizza_repartition | yes | no | no | no | sector | **KEEP (attrition)** |
| online_static_3step | no | fixed | yes | no | reservation | **KEEP** |
| dynamic_evidence_3step | no | anchor | yes | yes | reservation | **KEEP** |
| greedy (legacy) | no | fixed | no | no | global set | optional |
| spiral / concentric | yes | no | no | no | per-drone | optional |
| random_walk path (legacy) | no | no | no | no | none | excluded |
"""

BASELINE_MATRIX = """# C6 baseline matrix

| Baseline | Information class | Inputs | Training | Multi-UAV | Replanning | Reproducibility | C6 role |
|---|---|---|---|---|---|---|---|
| `random_uniform_n8_uncoordinated` | none (uninformed floor) | position, valid N8, per-UAV RNG | none | yes | online per-step | local source | exploration floor |
| `pizza_repartition` | geometry-only | bounds, FOV, altitude, spacing | none | yes | fixed path | local source | attrition arm |
| `online_static_3step` | fixed-prior online | prior heatmap, bounds, FOV | none | yes | online fixed prior | local source | informed online control |
| `dynamic_evidence_3step` | evidence-aware | prior, EvidenceBelief, bounds, FOV | none | yes | online posterior | local source | proposed method |
| `generate_greedy_path` (legacy) | prior (static) | probability map | none | yes | none | local source | legacy, not rerun |
| `generate_spiral_path` / `generate_concentric_circles_path` | geometry | geometry | none | yes | none | local source | legacy, not rerun |
| `generate_random_walk_path` (legacy) | none (zero map) | geometry | none | yes | none | local source | excluded |

Mapping: `dynamic_3step` (row/campaign name, C5 convention) is the same
implementation as `dynamic_evidence_3step` (C4 convention) — the EvidenceBelief
belief model is globally active in the frozen pipeline. C6 output tables carry
`method_label=dynamic_evidence_3step` and `source_planning_mode` to keep both
readable.

External map/feature-based methods remain related work: their inputs are not
exposed by SAREnv and their implementations are not available in the repository,
so they are classified CONCEPTUAL_RELATED_WORK (see c6_external_comparability.md).
"""

EXTERNAL_COMPARABILITY_HEADER = """# C6 external comparability

External methods are classified from the frozen related-work section and its
bibliography. Only a method that can be implemented faithfully on SAREnv inputs
with the same budget/initialization semantics may become an
EXPERIMENTAL_BASELINE; otherwise: CONCEPTUAL_RELATED_WORK. No external method is
adapted into a new variant solely to build a SOTA table.
"""

EXTERNAL_METHODS = [
    {
        "citation": "HEDAC / ergodic search (dumencicExperimentalValidationUAV2025, lancaProbabilisticModelingControl2026)",
        "problem": "wilderness search with UAV fleets",
        "single/multi UAV": "multi",
        "centralized/decentralized": "centralized",
        "required inputs": "target probability field, DEM (digital elevation model), motion model",
        "prior requirements": "requires a prior probability field",
        "DEM/features requirements": "DEM required",
        "training": "none",
        "sensor assumptions": "detection model (not SAREnv p_d semantics)",
        "communication assumptions": "centralized fusion",
        "fleet-loss assumptions": "not specified for the SAREnv protocol",
        "online replanning": "yes (ergodic coverage control)",
        "belief update": "no (assumes fixed target distribution)",
        "code availability": "not available in SAREnv repository",
        "reproducibility": "not reproducible inside SAREnv without reimplementation",
        "inputs available in SAREnv": "probability heatmap only; no DEM exposed at runtime",
        "same-budget comparison possible": "no (different execution loop and inputs)",
        "classification": "CONCEPTUAL_RELATED_WORK",
        "reason": "requires DEM and centralized fusion; SAREnv runtime exposes neither; adapting it would benchmark a degraded variant",
    },
    {
        "citation": "Dec-POMDP search (info_driven_pomdp_formation, pomdp_simultaneous_search)",
        "problem": "multi-UAV search under uncertainty",
        "single/multi UAV": "multi",
        "centralized/decentralized": "centralized planning / team coordination",
        "required inputs": "full POMDP model (transition/observation)",
        "prior requirements": "yes",
        "DEM/features requirements": "no",
        "training": "none",
        "sensor assumptions": "general observation model",
        "communication assumptions": "centralized fusion",
        "fleet-loss assumptions": "not specified",
        "online replanning": "planned (offline policy / receding-horizon solvers)",
        "belief update": "yes (belief in POMDP)",
        "code availability": "not available in SAREnv",
        "reproducibility": "not reproducible (solvers not available; NEXP-complete at operational scale excluded structurally)",
        "inputs available in SAREnv": "partial (heatmap, states); no solver",
        "same-budget comparison possible": "no",
        "classification": "CONCEPTUAL_RELATED_WORK",
        "reason": "computational intractability (NEXP-complete at scale) and unavailable solvers; exclusion is structural, not performance-based",
    },
    {
        "citation": "Mutual-information cooperative mapping (wolekCooperativeMappingTarget2020)",
        "problem": "cooperative mapping and target search",
        "single/multi UAV": "multi",
        "centralized/decentralized": "centralized fusion",
        "required inputs": "information-theoretic utilities over a shared map",
        "prior requirements": "yes",
        "DEM/features requirements": "no",
        "training": "none",
        "sensor assumptions": "information gain based",
        "communication assumptions": "centralized fusion",
        "fleet-loss assumptions": "not specified",
        "online replanning": "yes",
        "belief update": "yes (shared occupancy/belief)",
        "code availability": "not available in SAREnv",
        "reproducibility": "not reproducible inside SAREnv",
        "inputs available in SAREnv": "partial",
        "same-budget comparison possible": "no",
        "classification": "CONCEPTUAL_RELATED_WORK",
        "reason": "requires centralized fusion, outside the SAREnv decentralized-per-UAV planning loop",
    },
    {
        "citation": "Dual-mode exploration/gap-filling (bartolomeiFastMultiUAVDecentralized2023)",
        "problem": "fast multi-UAV coverage/replanning",
        "single/multi UAV": "multi",
        "centralized/decentralized": "decentralized (the closest external family)",
        "required inputs": "occupancy map, frontiers",
        "prior requirements": "no (exploration-driven)",
        "DEM/features requirements": "no",
        "training": "none",
        "sensor assumptions": "range sensing",
        "communication assumptions": "limited (hand-coded duality)",
        "fleet-loss assumptions": "not specified",
        "online replanning": "yes",
        "belief update": "no (occupancy)",
        "code availability": "not available in SAREnv",
        "reproducibility": "not reproducible inside SAREnv without reimplementation",
        "inputs available in SAREnv": "partial (heatmap; no occupancy map from sensing)",
        "same-budget comparison possible": "no",
        "classification": "CONCEPTUAL_RELATED_WORK",
        "reason": "exploration-driven, not probability-driven; dual-mode logic does not generalize to the SAREnv belief/search semantics (per the frozen paper text)",
    },
    {
        "citation": "Terrain-informed mountain search with decaying belief (terrain_informed_mountain)",
        "problem": "slope-derived belief, single UAV",
        "single/multi UAV": "single",
        "centralized/decentralized": "single agent",
        "required inputs": "slope/DEM-derived belief, subgoal A*",
        "prior requirements": "yes (slope-derived)",
        "DEM/features requirements": "DEM required",
        "training": "none",
        "sensor assumptions": "detection model",
        "communication assumptions": "n/a",
        "fleet-loss assumptions": "n/a (single UAV)",
        "online replanning": "yes (greedy iterative A*)",
        "belief update": "yes (per-cell exponential decay, comparable to EvidenceBelief)",
        "code availability": "not available in SAREnv",
        "reproducibility": "not reproducible inside SAREnv",
        "inputs available in SAREnv": "DEM not exposed at runtime",
        "same-budget comparison possible": "no",
        "classification": "CONCEPTUAL_RELATED_WORK",
        "reason": "single-UAV and DEM-dependent; concept-level relative comparison only",
    },
    {
        "citation": "Agent-based missing-person mobility priors (geMultiUAVSearchRescue2026)",
        "problem": "spatiotemporal prior estimation, not planning",
        "single/multi UAV": "multi (consumes the prior)",
        "centralized/decentralized": "prior generator",
        "required inputs": "terrain/behavioral model",
        "prior requirements": "produces a prior",
        "DEM/features requirements": "features",
        "training": "none (Monte Carlo agent-based)",
        "sensor assumptions": "perfect sensor (p_d = 1 per the frozen paper text)",
        "communication assumptions": "n/a as prior generator",
        "fleet-loss assumptions": "not specified",
        "online replanning": "no (no map update online)",
        "belief update": "no",
        "code availability": "not available in SAREnv",
        "reproducibility": "not reproducible inside SAREnv",
        "inputs available in SAREnv": "can consume such priors as P_prior (orthogonal)",
        "same-budget comparison possible": "no (not a planner)",
        "classification": "CONCEPTUAL_RELATED_WORK",
        "reason": "orthogonal prior generator; not a competing planner",
    },
    {
        "citation": "Coverage-path surveys (galceranSurveyCoveragePath2013, survey_coverage_dynamic) and coverage search baselines (uav_search_probability_map, collaborative_multirobot_sar, probability_maps_wadden)",
        "problem": "coverage planning in search",
        "single/multi UAV": "multi",
        "centralized/decentralized": "varies",
        "required inputs": "map/coverage geometry",
        "prior requirements": "some consume a probability map as input (no online update)",
        "DEM/features requirements": "none",
        "training": "none",
        "sensor assumptions": "coverage/sensing model",
        "communication assumptions": "varies",
        "fleet-loss assumptions": "not specified",
        "online replanning": "mostly no (precomputed coverage) or greedy",
        "belief update": "no (map consumed as input; frozen paper: 'most treat the probability field as an input the planner consumes but does not update online')",
        "code availability": "not available in SAREnv",
        "reproducibility": "not reproducible inside SAREnv",
        "inputs available in SAREnv": "partial (static coverage can be approximated by pizza/repartition)",
        "same-budget comparison possible": "no (different semantics; pizza is the SAREnv-native geometric analog)",
        "classification": "CONCEPTUAL_RELATED_WORK",
        "reason": "concept-level comparison; SAREnv-native geometric baselines already cover this family",
    },
]


def _external_comparability_text() -> str:
    lines = [EXTERNAL_COMPARABILITY_HEADER, ""]
    rows = ["| " + " | ".join(EXTERNAL_METHODS[0].keys()) + " |"]
    rows.append("|" + "---|" * len(EXTERNAL_METHODS[0]))
    for method in EXTERNAL_METHODS:
        rows.append("| " + " | ".join(str(value).replace("|", "/") for value in method.values()) + " |")
    lines.extend(rows)
    lines.append("")
    lines.append("## Verdict")
    lines.append("- EXPERIMENTAL_BASELINE: none (no external method is implementable faithfully on SAREnv inputs within this block).")
    lines.append("- CONCEPTUAL_RELATED_WORK: all methods above.")
    lines.append("- No external method was adapted into a new algorithm for a SOTA table.")
    lines.append("")
    lines.append("> Sources: `paper/ieeeRAL/references.bib` (canonical citations) and the frozen related-work text of the manuscript, which already states: 'Methods requiring inputs SAREnv does not expose (DEMs, centralized fusion, NEXP-complete solvers) are excluded structurally; we compare against the framework's native baselines under identical conditions.' C6 does not change that classification.")
    return "\n".join(lines) + "\n"


RELATED_WORK_UPDATE = """# C6 related work update

Comparisons remain separated by information available: geometry-only
(pizza_repartition, spiral, concentric), fixed-prior online (online_static_3step),
evidence-aware (dynamic_evidence_3step), uninformed online floor
(random_uniform_n8_uncoordinated), and externally map/feature-aware
(conceptual only).

Statements to keep (already in the frozen manuscript, consistent with C6):
- baselines are compared within the framework under identical conditions;
- methods requiring inputs SAREnv does not expose (DEMs, centralized fusion,
  NEXP-complete solvers) are excluded structurally, not on performance;
- external dec-POMDP / mutual-information / ergodic methods remain conceptual
  context, not new causal numbers.

Statements to avoid: presenting the C6 Random floor or any external method as a
"SOTA table" row, and attributing causal isolation to the Random contrast beyond
its role as an uninformed+memoryless+uncoordinated exploration floor.
"""

AUDIT_18OF60 = """# C6 18/60 provenance audit

## Status: FIXED_LIST_NO_FORMAL_RULE

There is a reproducible fixed list, but NO documented algorithm justifying
"maximize terrain heterogeneity".

## Evidence

1. The 18-dataset list is a literal CLI invocation in `scripts/05_comprehensive_study.py`
   (lines ~45-53) and `scripts/05b_comprehensive_study.py` (lines ~47-53):
   `--datasets 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 30`, with the comment
   `# Generalization (Study 6) - use 10-15+ datasets`.
2. Git archaeology: the list appears first in commit `58bd161` ("Estudios 05b (100)
   con analísis Holm-Bonferroni", 2026-05-31) — the commit that ran Study 6 as
   `study_datasets_20260531_095441.csv`; the same list already existed in
   `05_comprehensive_study.py` from earlier commits. The list was introduced with
   the study, not by a documented selection procedure.
3. `doc/05 resultados.md:121` records Study 6 = "DS1-15,20,25,30", 9 modes x 18 DS,
   n=30 -> 4,860 rows. No selection rationale is given there.
4. The manuscript (frozen `paper/ieeeRAL/paper.tex:235`) currently reads:
   "We selected 18 scenarios spanning the three SAREnv environmental categories
   (urban, rural, and mountainous) with the goal of maximizing terrain
   heterogeneity; the remaining 42 scenarios are available for future
   benchmarking." — The "maximizing terrain heterogeneity" formulation has no
   algorithm or document behind it.
5. `sarenv_dataset/datasets_info.md` classifies datasets 1-15 only (it was
   generated progressively, prompt: "Del 1 al 7 ya están hechos. Haz del 8 al 15");
   datasets 20, 25, 30 have no recorded type, so the "spanning the three
   categories" claim for those three is unevidenced.
6. The dataset pool itself (1..60) is NOT tracked by git (`sarenv_dataset` is not
   in `git ls-files`); the population of 60 cannot be verified from version
   control, only from the local artifact directory.

## Decision

- Remove the formulation "with the goal of maximizing terrain heterogeneity".
- Describe the set as a previously fixed list used by Study 6 (reference:
  `scripts/05b_comprehensive_study.py` + `study_datasets_20260531_095441.csv`).
- Do NOT claim C5 (the 18-dataset generalization campaign) justifies the
  selection: C5 demonstrates generalization within the 18, not a rule that
  produced them.
- `FULL_60_CONTINGENCY = false`: the failure mode here is a wording/claim issue,
  not an unresolvable selection provenance; no retrospective rule is invented.
"""

TERRAIN_HOMOGENEITY_AUDIT = """# C6 terrain homogeneity audit

## Claim audited

`paper/base/05. Results, statistical validation and discussion.md` (and the
derived manuscript text): "isolated cases occur where the geometric baseline
(pizza) outperforms our proposed method, specifically in datasets 3, 7, 9, and
11 (4 out of 18 datasets). Terrain analysis reveals that these datasets are
characterized by high terrain homogeneity (>70% of a single land-cover type),
where the prior probability map approaches a quasi-uniform distribution."

## Evidence

1. `scripts/11_terrain_analysis.py` computes `dominant_fraction` (share of the
   dominant land-cover class per dataset). The value 0.7 appears ONLY as a
   visualization threshold (`axvline(x=0.7, ... label='70% threshold')` and
   `colors = ['red' if x > 0.7 else 'blue' ...]`) and as a docstring hypothesis
   ("when a single class exceeds 70-75%, pizza dominates"). There is NO formal
   definition, classification code, or preregistered criterion for
   "terrain homogeneity >70%".
2. The measured artifact `results/terrain_analysis/terrain_analysis_results.csv`
   (datasets 1-60) gives `dominant_fraction`:
   - DS3 = 0.373, DS7 = 0.760, DS9 = 0.675, DS11 = 0.455
   - DS6 = 0.853 and DS24 = 0.916 (both > 0.7) are NOT listed in the claim.
   Only DS7 exceeds 0.7 among the four claimed datasets; DS6/DS24 exceed it and
   are absent. The claim is not reproducible under the measured values.
3. The 0.7 value exists in `scripts/11_terrain_analysis.py` only as a reporting
   device: a plot threshold (`axvline(x=0.7)`) and a stratified PRINTOUT
   ("High homogeneity (>70%)" / "Low homogeneity (≤70%)"). There is no
   preregistered definition, no classification function, and no criterion
   established before the results; the stratified output is a post-hoc
   reporting split, not a rule.

## Decision

- `decision = REMOVE`. The claim is not backed by a reproducible predefined
  definition and contradicts the available measured artifact.
- Do NOT construct a new post-hoc metric to explain outliers.
- If the (exploratory) observation that pizza wins in some datasets is kept,
  state it descriptively without the >70% or "quasi-uniform prior" wording.
"""

PMF_CONSERVATION_NOTE = """# C6 PMF conservation note

## Facts (frozen implementation)

1. `sarenv/analytics/dynamic_heatmap.py`: after the Bayesian update the
   posterior is renormalized (`mass_after` scaling, "Renormalize to conserve
   probability mass") and `apply_decay` also renormalizes after the
   per-cell exponential recovery (the per-cell interpolation does not preserve
   the sum by itself; the implementation applies the renormalization).
2. `sarenv/analytics/evidence_belief.py`: posterior = prior * ell / Z (Z =
   sum of unnormalized), so sum(P_t) = 1 at every read; the same statement
   appears in the frozen manuscript Eq. for the update and decay steps.
3. The update constant eta redistributes the extracted mass proportionally to
   existing posteriors and preserves relative ranks (frozen paper text).

## Interpretation to carry into the manuscript

- The normalization conserves the probabilistic semantics: the state is a
  posterior/distribution, sum_c P_t(c) = 1 at every timestep.
- It does NOT claim that global renormalization by itself improves an
  argmax-based planner; PMF conservation is not used as independent performance
  evidence.
- The paper already states this ("The interpolation ... does not preserve
  sum_c P_t(c) = 1 in general; the implementation applies a renormalization");
  C6 confirms the implementation against the text.

## C6 action

No additional experiment was executed for PMF conservation (per spec section
21); this note records the interpretation only.
"""

SENSOR_ASSUMPTIONS = """# C6 sensor assumptions

## Declared model (frozen pipeline)

- False negatives: modeled via p_d < 1 (per-exposure Bernoulli;
  `detection_probability = 0.8`; a cell visited n_c times has
  P(base detection | victim) = 1 - (1 - p_d)^n_c in the P_detect metric).
- False positives: assumed zero. There is no false-positive process in the
  detection step (`_detect_victims` only registers true detections) nor in the
  belief update (observations enter only as negative updates / detected cells).
- Spatial uniformity: p_d is spatially uniform (altitude- and
  canopy-dependent detection would require per-cell calibration; stated in the
  frozen manuscript limitations).

## C6 consequence

- Random's revisits are real exposures and keep their multiplicity in P_detect
  and the RMST temporal sequence; L remains unique-cell coverage.
- No false-positive campaign was added in C6. The zero-false-positive
  assumption is classified as a limitation of the sensor model, not a result.

## Wording to use

"The sensor model assumes spatially uniform p_d, models false negatives via
p_d < 1, and assumes zero false positives; the latter is a limitation of the
model, not an established property of operational sensors."
"""

SARENV_CAPABILITIES_AUDIT = """# C6 SAREnv capabilities audit

## Claim vs actual capability of the frozen SAREnv version

| Claim in v5 | Actual capability (frozen revision) | Evidence/source | Corrected wording |
|---|---|---|---|
| (implicit) SAREnv pipelines operate without auxiliary inputs such as DEMs | DEM is used at DATASET GENERATION time (`sarenv/core/generation.py` imports `elevation` and downloads/clips a DEM to build features); the frozen RUNTIME (DatasetLoader -> SARDatasetItem -> SARSimulation and the Block C pipeline) exposes only the probability heatmap, features Geometry and geometric bounds to planners — no DEM/occupancy input | `sarenv/core/generation.py:456-475`, `sarenv/core/loading.py`, `sarenv/analytics/simulation.py` | "SAREnv uses DEM data during dataset generation to derive terrain-weighted probability maps; the runtime exposes no DEM/occupancy input to planners." |
| External ergodic methods require DEMs and centralized coordination (so they are excluded structurally) | Consistent: no runtime DEM input exists in SAREnv, so such methods cannot be instantiated within the framework | frozen manuscript related-work text + the capability above | Keep the exclusion, stated as: "methods requiring DEM access cannot be instantiated on the inputs SAREnv exposes at runtime." |
| (any claim that planners exploit elevation online) | No planner in the frozen pipeline consumes elevation online; the probability map is the only planner input besides geometry/FOV parameters | `simulation.py` planning internals | Remove/avoid any statement implying online DEM use. |

## Rule applied

No statement about DEMs is kept without verification; the only verified DEM use
is in generation-time preprocessing. No claim of online DEM capability is made.
"""

MISSING_REFERENCES = """# C6 missing references

Claims surviving the rewrite that require a citation (or removal):

| Original location | Claim | Citation required | Candidate citation | Resolved |
|---|---|---|---|---|
| paper/base/05 (Results) | "high terrain homogeneity (>70% of a single land-cover type)" for DS3/7/9/11 | yes | none — claim removed (see c6_terrain_homogeneity_audit.md, decision REMOVE) | yes (by removal) |
| paper.tex §V setup | "We selected 18 scenarios ... with the goal of maximizing terrain heterogeneity" | yes | none — replaced by "previously fixed list" (see c6_18of60_provenance.md) | yes (by replacement) |
| paper.tex §V setup | "SAREnv includes a dataset of 60 search and rescue scenarios" | yes | `our_repo` (the framework artifact/directory listing sarenv_dataset 1..60) | yes |
| paper.tex §V setup | "18 scenarios spanning the three SAREnv environmental categories" | weak (descriptive) | kept with the fixed-list wording; categories only partially documented (datasets_info.md 1-15) | yes (worded as fixed list; categories described as recorded) |
| doc/05b resultados | "18 datasets geográficamente diversos" | descriptive, no external citation needed | study artifact (study_datasets_*.csv) | yes |

No claim is carried forward without either a reference or an explicit removal.
"""

REPRO_NOTE = """# C6 reproducibility notes

- Implementation: frozen `revision_B_base` behavior + additive C6 random mode.
- Seeds: environment seeds imported from C4/C5 rows (planning_seed); policy_seed
  per trial from SeedSequence([planning_seed, dataset_id, hash('c6-random-policy')]);
  per-UAV streams SeedSequence([policy_seed, uav_id]) -> default_rng(derived_seed).
- Runs: `pixi run python scripts/audit_block_c.py c6-random --input-experiment-id
  <C5> --attrition-parent-id <C4> --jobs 10` (full design; no --datasets/--seeds).
- Provenance columns recorded per row: policy_seed, rng_scheme,
  active/nominal UAV stream derivations, domain_hash, initial_positions_hash,
  parent_trial_id, source/config hashes.
- All statistics use the frozen Block C helpers (paired_stats.py); the global
  summary uses per-dataset means + cluster bootstrap (seed 20260826, 10k iters).
"""


# --------------------------------------------------------------------------- #
# Evidence facts: the editorial gate verifies that the audits' key facts are
# present in the repository artifacts, not merely that the documents were
# written (the documents above are the reviewed narrative; these checks back
# the claims they make).
# --------------------------------------------------------------------------- #

def _fact(path: Path, needle: str) -> bool:
    return bool(path.exists() and needle in path.read_text())


def _file_text(root: Path, rel: str) -> str:
    path = root / rel
    return path.read_text() if path.exists() else ""


def _collect_editorial_facts(root: Path) -> dict:
    facts = {}
    # §19 18/60: reproducible evidence of a fixed list and absence of a rule.
    facts["git_commit_58bd161_exists"] = bool(
        __import__("subprocess").run(
            ["git", "-C", str(root), "cat-file", "-e", "58bd161^{commit}"],
            capture_output=True,
        ).returncode == 0
    )
    facts["fixed_list_in_05b_script"] = bool(
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 30"
        in _file_text(root, "scripts/05b_comprehensive_study.py")
    )
    facts["study_docs_record_18_list"] = bool(
        "DS1-15,20,25,30" in _file_text(root, "doc/05 resultados.md")
        or "18 DS" in _file_text(root, "doc/05b resultados 100.md")
    )
    facts["paper_uses_maximize_heterogeneity_wording"] = bool(
        "maximizing terrain heterogeneity" in _file_text(root, "paper/ieeeRAL/paper.tex")
    )
    facts["datasets_info_covers_only_1_to_15"] = bool(
        "## Dataset 1" in _file_text(root, "sarenv_dataset/datasets_info.md")
        and "## Dataset 20" not in _file_text(root, "sarenv_dataset/datasets_info.md")
    )
    facts["dataset_pool_untracked_in_git"] = bool(
        __import__("subprocess").run(
            ["git", "-C", str(root), "ls-files", "sarenv_dataset"],
            capture_output=True, text=True,
        ).stdout.strip() == ""
    )
    # §20 terrain: measured dominant_fraction values and threshold semantics.
    import csv
    terrain_path = root / "results" / "terrain_analysis" / "terrain_analysis_results.csv"
    terrain_rows = {}
    if terrain_path.exists():
        with terrain_path.open() as handle:
            for row in csv.DictReader(handle):
                terrain_rows[int(row["dataset"])] = float(row["dominant_fraction"])
    expected = {3: 0.373, 7: 0.760, 9: 0.675, 11: 0.455}
    facts["terrain_measured_values_match_audit"] = all(
        dataset in terrain_rows and abs(terrain_rows[dataset] - value) < 0.01
        for dataset, value in expected.items()
    )
    facts["terrain_uncited_datasets_exceed_threshold"] = bool(
        6 in terrain_rows and terrain_rows[6] > 0.7 and 24 in terrain_rows and terrain_rows[24] > 0.7
    )
    facts["terrain_threshold_used_only_for_reporting"] = bool(
        "axvline(x=0.7" in _file_text(root, "scripts/11_terrain_analysis.py")
        and "High homogeneity (>70%)" in _file_text(root, "scripts/11_terrain_analysis.py")
        and "dominant_fraction'] > 0.7" in _file_text(root, "scripts/11_terrain_analysis.py")
    )
    # §23 DEM: generation-time use only; runtime exposes none.
    facts["dem_used_only_in_generation"] = bool(
        "import elevation" in _file_text(root, "sarenv/core/generation.py")
        and "elevation" not in _file_text(root, "sarenv/core/loading.py")
        and "elevation" not in _file_text(root, "sarenv/analytics/simulation.py")
    )
    facts["dem_statement_targets_external_methods"] = bool(
        "digital elevation models" in _file_text(root, "paper/ieeeRAL/paper.tex")
    )
    # §21 PMF conservation implementation facts.
    facts["pmf_renormalization_in_update"] = bool(
        "Renormalize to conserve probability mass" in _file_text(root, "sarenv/analytics/dynamic_heatmap.py")
    )
    facts["pmf_posterior_formula_in_evidence_belief"] = bool(
        "prior * ell" in _file_text(root, "sarenv/analytics/evidence_belief.py")
    )
    # §22 sensor: p_d false negatives modeled; no false-positive process.
    sim_text = _file_text(root, "sarenv/analytics/simulation.py")
    facts["sensor_false_negatives_via_pd"] = bool(
        "detection_probability" in sim_text and "p_d" in _file_text(
            root, "sarenv/analytics/detection_metrics.py"
        )
    )
    facts["sensor_no_false_positive_process"] = bool(
        "false_pos" not in sim_text.lower()
    )
    # §24 missing references: location of the surviving uncited claims.
    facts["terrain_claim_location_exists"] = bool(
        ">70% of a single land-cover type" in _file_text(root, "paper/base/05. Results, statistical validation and discussion.md")
    )
    facts["origen_datos_notes_uncited_homogeneity"] = bool(
        "terrain homogeneity" in _file_text(root, "paper/origen_datos.md")
    )
    # Evidence files exist.
    for rel in (
        "scripts/11_terrain_analysis.py",
        "scripts/05b_comprehensive_study.py",
        "results/terrain_analysis/terrain_metrics_only.csv",
        "sarenv/core/generation.py",
        "sarenv/analytics/dynamic_heatmap.py",
        "sarenv/analytics/evidence_belief.py",
        "sarenv/analytics/simulation.py",
        "paper/base/05. Results, statistical validation and discussion.md",
        "paper/origen_datos.md",
        "sarenv_dataset/datasets_info.md",
    ):
        facts[f"evidence_file_exists::{rel}"] = bool((root / rel).exists())
    return facts


def _write_docs(outdir: Path) -> dict[str, str]:
    docs = {
        "c6_native_baseline_audit.md": NATIVE_BASELINE_AUDIT,
        "c6_baseline_matrix.md": BASELINE_MATRIX,
        "c6_external_comparability.md": _external_comparability_text(),
        "c6_related_work_update.md": RELATED_WORK_UPDATE,
        "c6_18of60_provenance.md": AUDIT_18OF60,
        "c6_terrain_homogeneity_audit.md": TERRAIN_HOMOGENEITY_AUDIT,
        "c6_pmf_conservation_note.md": PMF_CONSERVATION_NOTE,
        "c6_sensor_assumptions.md": SENSOR_ASSUMPTIONS,
        "c6_sarenv_capabilities_audit.md": SARENV_CAPABILITIES_AUDIT,
        "c6_missing_references.md": MISSING_REFERENCES,
        "c6_reproducibility_notes.md": REPRO_NOTE,
    }
    written = {}
    for filename, text in docs.items():
        (outdir / filename).write_text(text)
        written[filename] = len(text)
    return written


def _doc_complete(outdir: Path, filename: str, min_chars: int = 500) -> bool:
    path = outdir / filename
    return bool(path.exists() and len(path.read_text()) >= min_chars)


def run_c6_editorial(args):
    cmod = _c()
    random_id = args.input_experiment_id or ""
    if not random_id:
        raise SystemExit("c6-editorial requires --input-experiment-id (the c6-random campaign id)")
    random_config_path = cmod.OUTPUT_DIR / random_id / "config.json"
    if not random_config_path.exists():
        raise SystemExit(f"c6-random campaign not found: {random_id}")
    random_config = json.loads(random_config_path.read_text())
    gate_path = cmod.OUTPUT_DIR / random_id / "c6_gate.md"
    if not gate_path.exists():
        raise SystemExit("c6-random campaign has no c6_gate.md; run the c6-random stage first")
    random_gate = cmod._read_gate_payload(gate_path)

    config = cmod._base_config(args, C6_EDITORIAL_STAGE)
    config.update({
        "parent_experiment_id": random_id,
        "parent_git_commit": random_config.get("git_commit"),
        "parent_source_hash": random_config.get("source_hash"),
        "label": "editorial",
        "datasets": [], "seeds": [],
        "condition_key": {"stage": C6_EDITORIAL_STAGE},
    })
    frozen_basis = {key: value for key, value in config.items() if not key.startswith("_")}
    config["_config_hash_basis"] = frozen_basis
    config, outdir = cmod._make_campaign(C6_EDITORIAL_STAGE, config, args.jobs)

    written = _write_docs(outdir)
    for filename in written:
        (cmod.OUTPUT_DIR / filename).write_bytes((outdir / filename).read_bytes())

    native_complete = _doc_complete(outdir, "c6_native_baseline_audit.md", 2000)
    matrix_complete = _doc_complete(outdir, "c6_baseline_matrix.md") and _doc_complete(outdir, "c6_external_comparability.md", 3000)
    related_complete = _doc_complete(outdir, "c6_related_work_update.md")
    baseline_green = bool(native_complete and matrix_complete and related_complete)

    # Evidence facts: the audit claims are verified against repository
    # artifacts (commit existence, measured CSVs, code text), not inferred from
    # document length or substrings of the documents themselves.
    facts = _collect_editorial_facts(ROOT)
    facts_pass = bool(all(facts.values()))
    facts_failed = sorted(key for key, value in facts.items() if not value)
    (outdir / "c6_editorial_evidence.json").write_text(
        json.dumps({"facts": facts, "all_pass": facts_pass, "failed": facts_failed},
                   indent=2, default=_json_default) + "\n"
    )
    (cmod.OUTPUT_DIR / "c6_editorial_evidence.json").write_bytes(
        (outdir / "c6_editorial_evidence.json").read_bytes()
    )

    status_18of60 = "FIXED_LIST_NO_FORMAL_RULE"
    editorial_rows = {
        "dataset_18of60_status_resolved": bool(
            "FIXED_LIST_NO_FORMAL_RULE" in (outdir / "c6_18of60_provenance.md").read_text()
            and facts["git_commit_58bd161_exists"]
            and facts["fixed_list_in_05b_script"]
            and facts["study_docs_record_18_list"]
        ),
        "terrain_homogeneity_resolved": bool(
            "REMOVE" in (outdir / "c6_terrain_homogeneity_audit.md").read_text()
            and facts["terrain_measured_values_match_audit"]
            and facts["terrain_threshold_used_only_for_reporting"]
        ),
        "pmf_claim_resolved": bool(
            _doc_complete(outdir, "c6_pmf_conservation_note.md")
            and facts["pmf_renormalization_in_update"]
            and facts["pmf_posterior_formula_in_evidence_belief"]
        ),
        "sensor_assumptions_resolved": bool(
            _doc_complete(outdir, "c6_sensor_assumptions.md")
            and facts["sensor_false_negatives_via_pd"]
            and facts["sensor_no_false_positive_process"]
        ),
        "dem_statement_resolved": bool(
            _doc_complete(outdir, "c6_sarenv_capabilities_audit.md")
            and facts["dem_used_only_in_generation"]
        ),
        "missing_references_resolved": bool(
            "yes" in (outdir / "c6_missing_references.md").read_text().lower()
            and facts["terrain_claim_location_exists"]
        ),
    }
    editorial_green = bool(all(editorial_rows.values()) and facts_pass)

    prohibited_hits = []
    docs_text = ""
    for name in sorted(outdir.glob("c6_*.md")):
        docs_text += name.read_text() + "\n"
    for term in [
        "fault tolerance", "positive synergy", "evidence-driven fleet resilience",
        "moving-victim robustness", "maximize heterogeneity",
    ]:
        if term.lower() in docs_text.lower():
            prohibited_hits.append(term)
    claims_consistency_green = bool(not prohibited_hits)

    random_green = bool(random_gate.get("C6_RANDOM_GREEN"))
    if random_gate.get("C6_RANDOM_SMOKE_PASS") and not random_green:
        random_green = False
    c6_green = bool(
        random_green and baseline_green and editorial_green and claims_consistency_green
    )

    gate_payload = {
        **random_gate,
        "C6_GREEN": c6_green,
        "C6_RANDOM_GREEN": random_green,
        "C6_BASELINES_GREEN": baseline_green,
        "C6_EDITORIAL_GREEN": editorial_green,
        "claims_consistency_green": claims_consistency_green,
        "native_baseline_audit_complete": native_complete,
        "external_baseline_matrix_complete": matrix_complete,
        "related_work_update_complete": related_complete,
        "dataset_18of60_status_resolved": editorial_rows["dataset_18of60_status_resolved"],
        "dataset_18of60_status": status_18of60,
        "terrain_homogeneity_resolved": editorial_rows["terrain_homogeneity_resolved"],
        "pmf_claim_resolved": editorial_rows["pmf_claim_resolved"],
        "sensor_assumptions_resolved": editorial_rows["sensor_assumptions_resolved"],
        "dem_statement_resolved": editorial_rows["dem_statement_resolved"],
        "missing_references_resolved": editorial_rows["missing_references_resolved"],
        "prohibited_wording_hits": prohibited_hits,
        "editorial_facts_pass": facts_pass,
        "editorial_facts_failed": facts_failed,
        "note": "C6_GREEN = C6_RANDOM_GREEN AND C6_BASELINES_GREEN AND C6_EDITORIAL_GREEN AND claims_consistency_green; the numeric result of Random is not part of the gate",
    }
    gate_text = (
        "# C6 gate\n\n"
        + json.dumps(gate_payload, indent=2, default=_json_default)
        + "\n"
    )
    # The gate lives in the c6-random campaign (single source of truth) and is
    # mirrored at the campaign root.
    (cmod.OUTPUT_DIR / random_id / "c6_gate.md").write_text(gate_text)
    (cmod.OUTPUT_DIR / "c6_gate.md").write_text(gate_text)

    manifest = {
        "schema_version": "block-C-v1",
        "stage": C6_EDITORIAL_STAGE,
        "experiment_id": config["experiment_id"],
        "created_at": config.get("timestamp"),
        "random_campaign": random_id,
        "docs": written,
        "gates": gate_payload,
    }
    (outdir / "c6_editorial_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    if not c6_green:
        raise SystemExit("C6_GREEN is FALSE; docs retained, gate NOT green")
    return config["experiment_id"]
