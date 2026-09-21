# Dynamic Belief Maps and Vectorized Lookahead Planning for Wilderness Search and Rescue

Anonymous reproducibility package. It contains the SAREnv implementation used to produce the
study, the locked environment, the frozen measurement tables, and the audit tooling that names
them. The scientific narrative, the reported numbers and every interpretation of them live in
the manuscript, not in this repository.

## Repository structure

```text
sarenv/            library (see "Library")
scripts/           runners, audits and the verification CLI (see "Scripts")
tests/             unit, regression and reproducibility tests
results/           frozen measurement tables (see "Results")
paper/figures/     figure scripts for the two data figures of the manuscript
                   (fig_generalization_c5.py, fig_c7_state_matched.py); they read only results/
legacy_v5/         archived v5 material: data/, figure_scripts/, experiments/ (see "Legacy v5 material")
scripts/legacy_v5/ archived v5 study runners and the archived v5 pipeline
pixi.toml          environment specification
pixi.lock          locked environment
```

## Library

`sarenv/`

| Module | Purpose |
|---|---|
| `audit.py` | Reproducibility records for the Block A audits: `AuditFixes` (independent switches for the differential-fix audit), `AuditRunRecord`, `stable_hash`, `make_audit_record`. |
| `analytics/dynamic_heatmap.py` | `DynamicHeatmap`: mass-conserving dynamic Bayesian map (legacy belief model, retained as a comparison arm). |
| `analytics/evidence_belief.py` | `EvidenceBelief`: evidence-anchored posterior `prior * ell / Z` with exact exponential recovery of `ell` toward the prior. |
| `analytics/detection_metrics.py` | Exact stationary single-victim detection metrics: `exposure_counts`, `expected_detection`, `expected_rmst`. |
| `analytics/midflight.py` | Pure-geometry helpers for fault-triggered pizza repartitioning: route trimming/splitting, min-entry task assignment, transit-path construction. Reads no probability source and uses no RNG. |
| `analytics/simulation.py` | `SARSimulation`, `SimulationResult`, `SimulationSnapshot`: the mission simulator and planner entry points. |
| `analytics/evaluator.py`, `analytics/metrics.py`, `analytics/paths.py`, `analytics/visualization.py` | Coverage-path generation, `PathEvaluator`, comparative evaluation and visualization. |
| `core/` | Dataset loading (`DatasetLoader`, `SARDatasetItem`), environment generation (`EnvironmentBuilder`, `DataGenerator`), geometries, lost-person location generation, victim movement models. |
| `io/`, `utils/` | OpenStreetMap queries; geospatial, plotting, logging and behavior helpers. |

## Scripts

| Script | Purpose |
|---|---|
| `audit_block_c.py` | Block C campaign orchestrator and the single CLI for every Block C stage. Freezes the campaign configuration and provenance, emits arm-level rows, validates pairing, writes summaries. |
| `run_block_c.sh` | Stage launcher: `scripts/run_block_c.sh --stage <stage> [runner options]`. |
| `c6_stage.py` | C6 closing block runner, invoked as `--stage c6-random`. |
| `c6_editorial.py` | C6 editorial stage, invoked as `--stage c6-editorial`. Runs no simulation. |
| `c7_stage.py` | C7 stages: `c7-midflight`, `c7-state-matched`, `c7-timing`, `c7-severity`, `c7-editorial`. |
| `audit_block_a.py` | Block A audit runner (`online`, `tree`, `differential`, `table` stages). No stage runs implicitly; it loads `05_comprehensive_study.py` for the study configuration and single-job pipeline. |
| `05_comprehensive_study.py` | v5-era study runner kept as the import target of `audit_block_a.py` (`StudyConfig`, `run_single_job`). Not a legacy/v5 archive: the Block A stages load it. |
| `audit_block_b.py` | Block B campaign orchestrator (`belief-tests`, `b1-differential`, `b2-stationary`, `b3-operational`, `b4-moving`, `b5-contrasts`). |
| `a2_microbench.py` | A2 microbenchmark: separates value-map cost from per-UAV selection cost over frozen mission states. |
| `c7a_analysis_common.py`, `c7a_tref_analysis.py`, `c7a_pizza_diagnostic.py`, `c7a_dynamic_vs_fixed.py` | C7A post-hoc analysis: producers of the shipped `C7A_*.csv` tables (retention at `T_ref`, per-trial retention under the distance budget, repartitioning overhead, and the `dynamic − pizza_fixed` per-dataset contrast). They read `c7_midflight_runs.csv.gz` and reuse `paired_stats.py` and `c7_stage.py`. |
| `paired_stats.py` | Shared paired non-parametric statistics helpers (matched pairs by `trial`, cluster-aware bootstrap, effect sizes). |
| `download_datasets.py` | Downloads `heatmap.npy` and `features.geojson` for the 18 datasets, pinned to an immutable upstream commit. |
| `verify_public_artifacts.py` | Verifies the published artifacts against `results/audit_block_C/public_manifest.json` (see "Verification"). |
| `run_full_pipeline.sh` | Release flow (see "Full reproduction"). |
| `legacy_v5/run_full_pipeline_v5.sh` | Archived v5 pipeline — legacy/v5, retained for provenance, not part of this cut. |
| `legacy_v5/analyze_study_results.py`, `legacy_v5/comprehensive_study.py`, `legacy_v5/drone_failure_resilience.py`, `legacy_v5/extra_studies.py`, `legacy_v5/nstep_sweep.py`, `legacy_v5/tree_crossover.py`, `legacy_v5/tree_search_baseline.py` | Legacy v5 study runners, archived: they are the scripts that produced the `data/` tables and are not exercised or verified by this cut. Two of them (`analyze_study_results.py`, `extra_studies.py`) import `scripts/paired_stats.py` by module name, so `scripts/` must stay on `PYTHONPATH` when they are run. |

## Results

```text
results/audit_block_A/            six A2 measurement tables produced by scripts/a2_microbench.py
results/audit_block_B/            six campaigns (belief-tests, b1-differential, b2-stationary,
                                  b3-operational, b4-moving, b5-contrasts)
results/audit_block_C/            nine campaigns (c1-sensitivity, c2-mismatch, c3-mobility,
                                  c4-attrition, c5-generalization, c6-random, c7-midflight,
                                  c7-state-matched, c7-timing)
```

Authority rule: **the campaign folder rules**. A campaign directory is self-contained — its
`config.json` fixes the identity of the run (`stage`, `git_commit`, `source_hash`,
`config_hash`, `expected_rows`) and the raw runs file named there is the primary measurement
table; the summaries in the same folder are derived from it. Nothing outside a campaign folder
overrides it.

Campaign folders carry machine-readable gates, audit dumps and test logs, and no interpretive
verdict documents: the reading of the results belongs to the manuscript, not to this tree.

Additional root-level files in `results/audit_block_C/`:

| File | Role |
|---|---|
| `c7_protocol_frozen.json` | The frozen C7 protocol. `scripts/c7_stage.py` aborts if it is missing. |
| `c0_manifest.json` | The frozen C0 implementation identity. `scripts/audit_block_c.py` reads `c0_impl_commit` from here and includes it in every campaign configuration, so it enters `config_hash`. |
| `c0_preexisting_untracked.json` | The untracked-file baseline policy consulted by the clean-tree gate before a campaign runs. |
| `public_manifest.json` | The published artifact manifest: critical-source hashes, per-campaign identity map and a `sha256` index of every file under the three result trees. Written and checked by `scripts/verify_public_artifacts.py`. |
| `C7A_TREF_GLOBAL.csv` | Retention at the fixed horizon `T_ref = 8000` s per arm (the manuscript's C7A end-to-end quantities). |
| `C7A_TREF_TRIALS.csv` | Per-trial retention under the distance budget and at `T_ref`. |
| `C7A_PIZZA_DIAGNOSTIC_TRIALS.csv` | Per-trial repartitioning overhead (deadhead, removed prefix, task splits, empty sectors). |
| `C7A_DYNAMIC_VS_PIZZA_FIXED_DATASETS.csv` | Per-dataset evidence-guided minus precomputed-route contrast (mean `dPdet = +0.0032`), the C7A end-to-end diagnostic. |

## Paper experiments and campaign folders

| Paper | Campaign folder(s) under `results/` | Planner identifiers in the tables |
|---|---|---|
| E1 (confirmatory) | `audit_block_C/c5-generalization_20260827_181139_043334_662411fc` | `dynamic_3step` (method), `online_static_3step` |
| E2 (mechanism, ablation, sensitivity) | `audit_block_B/b3-operational_20260825_130210_101492_c372f280`, `audit_block_B/b5-contrasts_20260825_153608_604640_b5b93fdc`, `audit_block_C/c1-sensitivity_20260826_073133_703049_c676526b`, `audit_block_C/c2-mismatch_20260826_181638_227156_4f4ff330` | B3/B5 carry the arms in column names (`Pdet_static3`/`Pdet_dyn3`, `L_os3`/`L_dyn3`); C1/C2 use the `planning_mode` column |
| E3 (boundary: mobility) | `audit_block_C/c3-mobility_20260827_043124_280564_ae41202c` | `dynamic_3step`, `online_static_3step` |
| E4 (fleet reduction) | `audit_block_C/c4-attrition_20260827_075624_501873_327229d1`, `audit_block_C/c6-random_20260831_182014_803486_5cf558a2` | `dynamic_evidence_3step`, `online_static_3step`, `pizza_repartition`, `random_uniform_n8_uncoordinated` |
| E5 (planner fidelity and cost) | `audit_block_A/a2_*.csv` | A2 tables; no arm column |
| E6 (mid-mission UAV loss) | `audit_block_C/c7-midflight_20260909_155840_581849_989802fd` (diagnostic), `audit_block_C/c7-state-matched_20260910_070544_572098_9ae47248` (reported), `audit_block_C/c7-timing_20260911_041506_496602_705c3cf8` (supporting) | `dynamic_evidence_3step`, `online_static_3step`, `pizza_replan_midflight`, `pizza_fixed_midflight`, `random_uniform_n8_uncoordinated` |

`dynamic_3step` (E1–E3) and `dynamic_evidence_3step` (E4, E6) are the same planner; every campaign `config.json` records `belief_model: "evidence"` and `fix_profile: "revision_B_base"`. B3/B4/B5 record no `planning_mode` value.

## Requirements

- macOS or Linux
- Bash-compatible shell
- Python environment managed by [`pixi`](https://pixi.prefix.dev/latest/)
- Network access for the one-time dataset download

## Installation

Install the locked environment from `pixi.toml` and `pixi.lock`:

```bash
pixi install
```

Then run every command from the repository root.

```bash
pixi run python <script>
```

## Dataset download

The 18 required datasets are not stored in this repository. Download them into `sarenv_dataset/`:

```bash
pixi run python scripts/download_datasets.py
```

The script retrieves `heatmap.npy` and `features.geojson` for each dataset and pins the download
to an immutable upstream commit, so the result is byte-stable across machines.

## Figures

The two data figures of the manuscript are generated from this repository's result tables:

```bash
pixi run python paper/figures/fig_generalization_c5.py   # Fig. 2 (C5, 18 scenarios)
pixi run python paper/figures/fig_c7_state_matched.py    # Fig. 3 (C7B branches and contrasts)
```

Both read only `results/` and the frozen `scripts/paired_stats.py`, print the numbers they draw,
and write their PDFs to `paper/figures/output/` (ignored by git). The manuscript's Fig. 1 is a
TikZ graphic and needs no data.

## Verification

Verify the published result trees against the manifest:

```bash
pixi run python scripts/verify_public_artifacts.py
```

Exit code 0 means every check passed; any failure is reported on stderr with exit code 1.
`--update` regenerates `results/audit_block_C/public_manifest.json` and then re-runs all checks.

The checks cover: the critical-source hash of the implementation, every hashed artifact, the
completeness of the three result trees, the identity fields of every shipped campaign, the
declared row counts, and the integrity flags recorded in the C7 measurement tables.

## Running Block C stages

```bash
scripts/run_block_c.sh --stage <stage> [runner options]
```

Stages executable with this tree: `c1-sensitivity`, `c2-mismatch`, `c2-lost-person`,
`c3-mobility`, `c4-attrition`, `c5-generalization`, `c6-random`, `c6-editorial`, `c7-midflight`,
`c7-state-matched`, `c7-timing`, `c7-severity`, `c7-editorial`, plus the `*-summary` stages.
Stages that consume a parent campaign take `--input-experiment-id <campaign directory name>`;
the parent campaigns are the ones shipped under `results/audit_block_C/`.

A campaign whose exact configuration already exists is skipped instead of re-run; `--rerun`
forces a new campaign ID.

Two limitations of this tree:

- `c-final` is not executable here: it requires `block_C_manifest_final.pre_c6.json`, which is
  not distributed.
- The editorial stages (`c6-editorial`, `c7-editorial`) are available as code, but their outputs
  are not part of this artifact.

Run the stages directly through `run_block_c.sh`; no full-chain wrapper is distributed.

### Re-running a C7 campaign

Re-launching a complete C7 campaign requires its `ckpt_*.zip` checkpoint shards, which are not
included in this package. The supported way to check a C7 campaign is therefore identity
verification rather than relaunch — for example:

```bash
SARENV_C_IMPL_VERSION=712b395 pixi run python scripts/audit_block_c.py c7-midflight \
  --campaign-id c7-midflight_20260909_155840_581849_989802fd
```

This recomputes the critical-source hash and the campaign configuration hash from the frozen
inputs and reports the campaign as already complete only if both match the values recorded in
its `config.json`. `SARENV_C_IMPL_VERSION` fixes the recorded implementation label; without it
the run is refused with `campaign id does not match frozen config`.

The frozen runner signals an already-complete campaign by raising `CampaignAlreadyComplete`, so
the script entry point prints that message inside a traceback and exits 1 — that traceback is the
success signal, and it is only reachable after both hashes match. Invoking the same runner
in-process reports it plainly and exits 0:

```bash
SARENV_C_IMPL_VERSION=712b395 pixi run python -c "
import sys
sys.path.insert(0, 'scripts')
import audit_block_c as c
sys.argv = ['audit_block_c', 'c7-midflight', '--campaign-id', 'c7-midflight_20260909_155840_581849_989802fd']
raise SystemExit(c.main())"
# campaign c7-midflight_20260909_155840_581849_989802fd already complete; skipping
```

Both forms read the campaign folder under `results/audit_block_C/`; `c7_midflight_runs.csv` in
the C7A folder is also the input the `c7-state-matched` and `c7-timing` preflights require.

## Full reproduction

The release flow installs the locked environment, fetches the pinned datasets, verifies the
artifact manifest and runs the test suite:

```bash
pixi run bash scripts/run_full_pipeline.sh
```

The legacy v5 pipeline is archived as `scripts/legacy_v5/run_full_pipeline_v5.sh`; it is not part
of this cut and is not exercised by it.

## Legacy v5 material

The following are archived under the `legacy/v5` label: they are the record of the earlier study
and are **not** reproduced from this repository.

- `legacy_v5/data/study_main_20260531_095441.csv`, `legacy_v5/data/study_datasets_20260531_095441.csv`,
  `legacy_v5/data/cliff_delta_pairwise_likelihood.csv`, `legacy_v5/data/cliff_delta_per_dataset_likelihood.csv`,
  `legacy_v5/data/lmm_lrt_likelihood.csv`, `legacy_v5/data/tost_equivalence_likelihood.csv`,
  `legacy_v5/data/per_dataset_dynamic_3step_vs_pizza_likelihood.csv`, `legacy_v5/data/resilience_summary.csv`,
  `legacy_v5/data/nstep_sweep.csv`, `legacy_v5/data/tree_search.csv`, `legacy_v5/data/tree_crossover.csv`,
  `legacy_v5/data/teaser_cells_*.npy`, `legacy_v5/data/teaser_perdrone_*.npz`, `legacy_v5/data/teaser_likelihoods.txt`
- `legacy_v5/data/figures/`
- `legacy_v5/figure_scripts/`
- `legacy_v5/experiments/`
- the legacy study runners and the v5 pipeline under `scripts/legacy_v5/` (listed under "Scripts")

Everything else — `sarenv/`, the Block A/B/C scripts, `tests/` and `results/` — is the material
covered by this cut. Per-file provenance of the legacy tables is in `legacy_v5/data/README.md`.

## License

This repository builds on SAREnv and preserves the original MIT license notice in
[LICENSE](LICENSE). New code in this repository is distributed under the same MIT license.
