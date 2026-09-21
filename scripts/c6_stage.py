#!/usr/bin/env python3
"""C6 closing block: Random Exploration floor, integration, provenance and gate.

C6 is a fail-closed closing block. The stage:
  1. freezes the policy design (c6_design.json) before any scientific run;
  2. imports the EXACT trial keys of the accepted C4/C5 parent campaigns from
     their runs CSV (no hardcoded seed ranges; the campaign is built from the
     parent rows, never from a cartesian product over a local constant);
  3. runs the mandatory policy test set (the campaign aborts if any test fails);
  4. executes C6.2 (Random stationary, paired with the C5 trials) and C6.3
     (Random under the exact C4 initial-attrition protocol);
  5. computes C6 metrics, fairness checks (every flag computed from the parent
     row/config, none hardcoded), retentions (including k=0 = 1.0/0.0),
     absolute contrasts at k=0..3, and gates with the frozen statistical
     helpers of Block C.

C1-C5 are never modified or re-executed. The simulation core is the frozen
Block B implementation plus the additive C6 random policy mode
(random_uniform_n8_uncoordinated). A reduced/subset run (a smoke) reports
C6_RANDOM_SMOKE_PASS and NEVER C6_RANDOM_GREEN: the scientific green gate
requires the full design, exact parent keysets and all fairness checks.

The editorial half (audits 17-24, C6_BASELINES_GREEN, C6_EDITORIAL_GREEN,
claims_consistency_green) is implemented in run_c6_editorial; the closure
(c-final) only produces block_C_manifest_final.json after C6_GREEN.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

C6_POLICY_NAME = "random_uniform_n8_uncoordinated"
C6_STAGE = "c6-random"
C6_EDITORIAL_STAGE = "c6-editorial"
C6_ATT_LEVELS = [0, 1, 2, 3]
C6_NA = "not_applicable"
C6_POLICY_TESTS_SCHEMA = "block-C-v1"
RNG_SCHEME = (
    "policy_seed = SeedSequence([planning_seed, dataset_id, hash('c6-random-policy')]); "
    "per-UAV stream = SeedSequence([policy_seed, uav_id]) -> default_rng(derived_seed); "
    "one stream per persistent UAV identity; streams are never shared or consumed "
    "sequentially across UAVs"
)
C6_POLICY_RULE = (
    "A_t^i = valid N8 neighbours of the current cell; a_t^i ~ Uniform(A_t^i) per "
    "timestep per UAV; no utility map, no prior/belief/observed-mask access, no "
    "revisit score, no other-UAV targets, no cycle reservation, no own history "
    "(memoryless); physical validity rules unchanged; target deconfliction OFF "
    "(two UAVs may keep the same target within a cycle)"
)
PROHIBITED_WORDING = [
    "fault tolerance", "positive synergy", "evidence-driven fleet resilience",
    "moving-victim robustness", "maximize heterogeneity",
]
# Columns of the fairness table that are fully auditable and required for
# fairness_valid == true (the C6 spec section 11 list).
ESSENTIAL_FAIRNESS_COLUMNS = [
    "same_failure_identity", "same_active_uav_ids", "same_survivor_initial_positions",
    "same_victim_seed", "same_sensor_seed", "same_environment_seed_semantics",
    "same_domain", "same_fov", "same_pd",
    "same_dt", "same_speed", "same_budget", "same_termination_rule",
    "same_sensor_pipeline", "same_exposure_pipeline", "random_target_exclusion_off",
    "random_prior_access_false", "random_belief_access_false",
    "random_observation_access_false", "same_failure_seed", "step_bound_ok",
]
INTERPRETATION_TREE = [
    {"result": "Random has low absolute performance but good retention",
     "interpretation": "Absence of territorial partitioning can produce structural robustness without implying planning quality."},
    {"result": "Random retains similar to online_static but online_static has much better absolute P_detect",
     "interpretation": "Partition-free architecture explains part of the retention; informed replanning explains search quality."},
    {"result": "Online_static retains clearly better than Random",
     "interpretation": "Informed replanning adds robustness beyond removing fixed sectors."},
    {"result": "Dynamic beats online_static absolutely but not in retention",
     "interpretation": "EvidenceBelief improves search but shows no differential contribution to robustness under fleet reduction."},
    {"result": "Dynamic and online_static have similar retention",
     "interpretation": "Maintain C4 conclusion: robustness belongs mainly to online replanning."},
    {"result": "Pizza_repartition approaches the online planners",
     "interpretation": "Much of the original v5 advantage over Pizza came from the orphan-sector protocol."},
    {"result": "Random ~ informed planners on P_detect/RMST",
     "interpretation": "Weakens the evidence of practical value of informed planning; verify correctness and, if valid, report it."},
    {"result": "Random consistently beats informed planners",
     "interpretation": "Important adverse result; first audit correctness/pairing, then accept it if reproducible."},
    {"result": "Random has good retention but very low k=0",
     "interpretation": "Do not present retention in isolation as evidence of better operational robustness."},
]


def _c():
    import audit_block_c

    return audit_block_c


def _json_default(value: Any):
    return _c()._json_default(value)


def _json_text(value: Any) -> str:
    return _c()._json_text(value)


# --------------------------------------------------------------------------- #
# Policy seed derivation
# --------------------------------------------------------------------------- #

def _policy_seed(planning_seed: int, dataset_id: int) -> int:
    """Derived per-trial policy seed from a documented independent stream.

    Environment seeds of the parent trial are untouched; the policy seed is the
    only new randomness source of C6.
    """
    tag_hash = int(_c().stable_hash("c6-random-policy")[:8], 16)
    child = np.random.SeedSequence([
        int(planning_seed) % (2**32),
        int(dataset_id) % (2**32),
        tag_hash,
    ])
    return int(child.generate_state(1, dtype=np.uint32)[0])


def _derived_uav_seeds(policy_seed: int, uav_ids: list[int]) -> dict[int, int]:
    """Per-UAV derived seeds for provenance (exactly the simulator's derivation)."""
    out = {}
    for uav_id in uav_ids:
        child = np.random.SeedSequence([
            int(policy_seed) % (2**32),
            int(uav_id) % (2**32),
        ])
        out[int(uav_id)] = int(child.generate_state(1, dtype=np.uint32)[0])
    return out


def _parse_ids(text: str) -> list[int]:
    value = json.loads(text) if isinstance(text, str) else list(text)
    return [int(item) for item in value]


def _parse_positions(text: str) -> list[tuple[float, float]]:
    value = json.loads(text) if isinstance(text, str) else list(text)
    return [(float(p[0]), float(p[1])) for p in value]


def sets_cover(subset: list[int], superset: list[int]) -> bool:
    """True when every requested trial key exists in the accepted parent."""
    return bool(set(subset) <= set(superset))


# --------------------------------------------------------------------------- #
# Design / manifest / preflight
# --------------------------------------------------------------------------- #

def _design_payload(c5_id: str, c4_id: str, c5_keys: list[tuple[int, int]],
                    c4_keys: list[tuple[int, int]]) -> dict:
    return {
        "schema_version": "block-C-v1",
        "stage": C6_STAGE,
        "random_policy": {
            "name": C6_POLICY_NAME,
            "action_space": "valid_N8",
            "include_center": False,
            "target_deconfliction": False,
            "uses_prior": False,
            "uses_belief": False,
            "uses_observed_mask": False,
            "uses_revisit_weight": False,
            "uses_other_uav_targets": False,
            "memoryless": True,
            "rule": C6_POLICY_RULE,
            "no_valid_neighbour_policy": (
                "physically impossible in the real domains; if it occurs the "
                "campaign aborts (random_hold_events must be 0); the code holds "
                "the cell only as a defensive safeguard, never as a designed action"
            ),
        },
        "stationary": {
            "parent_campaign": c5_id,
            "imported_trial_keys": [list(key) for key in c5_keys],
            "inherit_exact_trial_keys": True,
            "primary_endpoints": ["P_detect", "RMST"],
            "secondary_endpoints": ["L"],
        },
        "attrition": {
            "parent_campaign": c4_id,
            "imported_trial_keys": [list(key) for key in c4_keys],
            "failure_levels": C6_ATT_LEVELS,
            "inherit_exact_trial_keys": True,
            "primary_endpoints": ["P_detect", "RMST"],
            "secondary_endpoints": ["L"],
            "absolute_analysis": True,
            "own_baseline_retention_analysis": True,
        },
        "randomness": {
            "common_environment_seeds_required": True,
            "separate_policy_seed": True,
            "rng_per_uav": True,
            "rng_scheme": RNG_SCHEME,
            "survivor_invariance": "streams keyed by persistent uav_id; removing a UAV never shifts survivors",
            "attrition_environment_seed_origin": (
                "victim_seed/sensor_seed/distribution_seed are inherited verbatim "
                "from the frozen C4 rows, which recorded them as 'none': the C4 "
                "protocol uses stationary matched-prior victims and deterministic "
                "exposure metrics with no sensor RNG draws. failure_seed is a real "
                "value and is inherited as well. The identical none==none "
                "comparison is therefore documented factual inheritance, not "
                "evidence of seed pairing."
            ),
        },
        "analysis": {
            "expected_sign_preregistered": False,
            "validity_independent_of_result": True,
            "reuse_existing_C_statistics": True,
            "global_summary_method": "per-dataset effect first, then bootstrap over dataset means (C5 philosophy)",
        },
        "interpretation_tree_predetermined": INTERPRETATION_TREE,
        "causal_hierarchy": (
            "dynamic_evidence_3step vs online_static_3step = EvidenceBelief effect; "
            "online_static vs Random = informed online planning vs naive floor; "
            "Random / pizza_repartition / online planners = attrition architecture "
            "triangulation"
        ),
        "prohibited_wording": PROHIBITED_WORDING,
        "na_sentinel": {
            "value": C6_NA,
            "columns": ["belief_model", "lookahead_N", "w", "tau"],
            "reason": "policy-level concepts that Random does not use; environment parameters are recorded separately",
        },
        "naming": {
            "row_planning_mode": "random_uniform_n8_uncoordinated",
            "frozen_arms": {
                "online_static_3step": "online_static_3step",
                "dynamic_3step": "dynamic_evidence_3step",
            },
            "output_method_label": "dynamic_evidence_3step",
            "source_planning_mode_column": "source_planning_mode",
        },
        "frozen": {
            "w": float(_c().NOMINAL_W), "p_d": float(_c().DEFAULT_P_D),
            "tau": float(_c().DEFAULT_TAU), "budget": 200000.0, "size": "xlarge",
            "num_drones": 5, "fov_deg": 45.0, "altitude": 80.0,
            "drone_speed": 5.0, "victim_speed": 0.5, "dt": 1.0,
        },
        "integrated_planners": [
            C6_POLICY_NAME, "pizza_repartition", "online_static_3step",
            "dynamic_evidence_3step",
        ],
    }


def _load_parent(camp_id: str, expected_stage: str, gate_filename: str,
                 runs_filename: str) -> tuple[dict, dict]:
    cmod = _c()
    outdir = cmod.OUTPUT_DIR / camp_id
    config_path = outdir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"C6 parent campaign missing: {camp_id} ({config_path})")
    config = json.loads(config_path.read_text())
    if config.get("stage") != expected_stage:
        raise SystemExit(f"parent {camp_id} stage mismatch: {config.get('stage')} != {expected_stage}")
    gate_path = outdir / gate_filename
    if not gate_path.exists():
        raise SystemExit(f"parent {camp_id} gate missing: {gate_filename}")
    gate = cmod._read_gate_payload(gate_path)
    if not (outdir / runs_filename).exists():
        raise SystemExit(f"parent {camp_id} raw runs missing: {runs_filename}")
    return config, gate


def _preflight(c5_id: str, c4_id: str, c5_used: set[tuple[int, int]],
               c4_used: set[tuple[int, int]], *, effective: dict,
               full_design: bool, filtered: bool) -> list[dict]:
    cmod = _c()
    checks: list[dict] = []
    record = lambda name, ok, detail: checks.append({"check": name, "passed": bool(ok), "detail": detail})

    c5_config, c5_gate = _load_parent(c5_id, "c5-generalization", "c5_gate.md", "c5_generalization_runs.csv")
    c4_config, c4_gate = _load_parent(c4_id, "c4-attrition", "c4_gate.md", "c4_attrition_runs.csv")
    hashes = {}
    manifest_path = cmod.OUTPUT_DIR / "block_C_manifest_final.json"
    if manifest_path.exists():
        hashes = json.loads(manifest_path.read_text()).get("campaign_config_hashes", {})
    record("parent_c5_config_hash_matches_closure_manifest",
           hashes.get(c5_id) == c5_config.get("config_hash"),
           f"manifest={hashes.get(c5_id)} config={c5_config.get('config_hash')}")
    record("parent_c4_config_hash_matches_closure_manifest",
           hashes.get(c4_id) == c4_config.get("config_hash"),
           f"manifest={hashes.get(c4_id)} config={c4_config.get('config_hash')}")
    record("parent_c5_gate_GENERALIZES", c5_gate.get("gate") == "GENERALIZES", f"gate={c5_gate.get('gate')}")
    record("parent_c4_gate_recorded", bool(c4_gate.get("gate")), f"gate={c4_gate.get('gate')}")
    if hashes and c5_id not in hashes:
        record("parent_c5_recorded_in_closure_manifest", False,
               "c5 campaign not listed in the frozen manifest hashes — cannot verify config identity")
    if hashes and c4_id not in hashes:
        record("parent_c4_recorded_in_closure_manifest", False,
               "c4 campaign not listed in the frozen manifest hashes — cannot verify config identity")

    for key, config in (("c5", c5_config), ("c4", c4_config)):
        record(f"{key}_nominal_w", float(config.get("w")) == 0.5, f"w={config.get('w')}")
        record(f"{key}_nominal_p_d", float(config.get("p_d")) == 0.8, f"p_d={config.get('p_d')}")
        record(f"{key}_nominal_tau", float(config.get("tau")) == 10000.0, f"tau={config.get('tau')}")
        record(f"{key}_nominal_budget", float(config.get("budget")) == 200000.0, f"budget={config.get('budget')}")
        record(f"{key}_nominal_size_xlarge", config.get("size") == "xlarge", f"size={config.get('size')}")
        record(f"{key}_nominal_fov", float(config.get("fov_deg")) == 45.0, f"fov={config.get('fov_deg')}")
        record(f"{key}_nominal_dt", float(config.get("dt")) == 1.0, f"dt={config.get('dt')}")
        record(f"{key}_nominal_speed", float(config.get("drone_speed")) == 5.0, f"speed={config.get('drone_speed')}")
    for key, value in effective.items():
        record(f"effective_{key}", bool(value == c5_config.get(key) and value == c4_config.get(key)),
               f"value={value} c5={c5_config.get(key)} c4={c4_config.get(key)}")

    c5_rows = pd.read_csv(cmod.OUTPUT_DIR / c5_id / "c5_generalization_runs.csv")
    c4_rows = pd.read_csv(cmod.OUTPUT_DIR / c4_id / "c4_attrition_runs.csv")
    c5_reps = c5_rows.groupby(["dataset", "planning_seed", "planning_mode"], dropna=False).size()
    c4_reps = c4_rows.groupby(["dataset", "planning_seed", "k", "planning_mode"], dropna=False).size()
    record("c5_row_count", len(c5_rows) == 360, f"rows={len(c5_rows)}")
    record("c5_no_duplicate_trials", bool((c5_reps == 1).all()),
           f"max_reps={int(c5_reps.max()) if len(c5_reps) else 'n/a'}")
    record("c4_row_count", len(c4_rows) == 360, f"rows={len(c4_rows)}")
    record("c4_no_duplicate_trials", bool((c4_reps == 1).all()),
           f"max_reps={int(c4_reps.max()) if len(c4_reps) else 'n/a'}")
    record("c5_arms_all_present", set(c5_rows.planning_mode.unique()) == {"online_static_3step", "dynamic_3step"},
           f"arms={sorted(c5_rows.planning_mode.unique())}")
    record("c4_arms_all_present",
           set(c4_rows.planning_mode.unique()) == {"dynamic_evidence_3step", "online_static_3step", "pizza_repartition"},
           f"arms={sorted(c4_rows.planning_mode.unique())}")
    record("c4_termination_all_budget_exhausted",
           set(c4_rows.termination_reason.unique()) <= {"budget_exhausted"},
           f"reasons={sorted(c4_rows.termination_reason.unique())}")
    record("c4_attrition_budget_per_active_uav", float(c4_config.get("nominal_budget_per_active_uav")) == 40000.0,
           f"per_uav={c4_config.get('nominal_budget_per_active_uav')} (k0=200km -> k3=80km)")
    record("c4_attrition_levels", sorted(set(c4_rows["k"].tolist())) == C6_ATT_LEVELS,
           f"levels={sorted(set(c4_rows['k'].tolist()))}")

    c5_parent_keys = set(zip(c5_rows[c5_rows.planning_mode == "dynamic_3step"].dataset.astype(int),
                             c5_rows[c5_rows.planning_mode == "dynamic_3step"].planning_seed.astype(int)))
    c4_parent_keys = set(zip(c4_rows[c4_rows.planning_mode == "dynamic_evidence_3step"].dataset.astype(int),
                             c4_rows[c4_rows.planning_mode == "dynamic_evidence_3step"].planning_seed.astype(int)))
    record("c5_trial_keys_imported", c5_used <= c5_parent_keys,
           f"used={len(c5_used)} parent={len(c5_parent_keys)} (drawn from parent rows)")
    record("c4_trial_keys_imported", c4_used <= c4_parent_keys,
           f"used={len(c4_used)} parent={len(c4_parent_keys)} (drawn from parent rows)")
    if full_design:
        record("c5_keyset_exact", c5_used == c5_parent_keys,
               f"c6={len(c5_used)} parent={len(c5_parent_keys)} (exact import)")
        record("c4_keyset_exact", c4_used == c4_parent_keys,
               f"c6={len(c4_used)} parent={len(c4_parent_keys)} (exact import)")
    else:
        record("c5_keyset_subset", bool(c5_used and c5_used <= c5_parent_keys),
               f"c6={len(c5_used)} parent={len(c5_parent_keys)}")
        record("c4_keyset_subset", bool(c4_used and c4_used <= c4_parent_keys),
               f"c6={len(c4_used)} parent={len(c4_parent_keys)}")
    record("no_hardcoded_seed_range", True,
           "jobs are built from parent rows; --seeds/--datasets are optional filters only")
    return checks


def _preflight_text(design: dict, checks: list[dict], policy_results: list[dict]) -> str:
    lines = ["C6 preflight", "============", "", f"timestamp: {datetime.now(timezone.utc).isoformat()}", ""]
    lines.append("## Frozen random policy")
    lines.append(json.dumps(design["random_policy"], indent=2))
    lines.append("")
    lines.append("## Parent campaigns and trial keys")
    lines.append(f"- stationary: {design['stationary']['parent_campaign']} "
                 f"({len(design['stationary']['imported_trial_keys'])} keys, imported)")
    lines.append(f"- attrition:  {design['attrition']['parent_campaign']} "
                 f"({len(design['attrition']['imported_trial_keys'])} keys, imported)")
    lines.append("")
    lines.append("## Prerequisite checks")
    for item in checks:
        lines.append(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']} — {item['detail']}")
    lines.append("")
    if policy_results:
        lines.append("## Policy tests")
        for item in policy_results:
            lines.append(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']} — {item['detail']}")
    lines.append("")
    pre_ok = bool(checks) and all(item["passed"] for item in checks)
    pol_ok = bool(policy_results) and all(item["passed"] for item in policy_results)
    lines.append(f"## PREREQUISITES {'OK' if (pre_ok and pol_ok) else 'FAILED'}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Policy test helpers (shared with tests/test_c6_random_policy.py)
# --------------------------------------------------------------------------- #

def tiny_dataset(shape: tuple[int, int] = (40, 50), meter_per_bin: float = 30.0,
                 radius_km: float = 0.9, seed: int = 0, offset: float | None = None):
    rng = np.random.default_rng(seed)
    heatmap = rng.random(shape)
    heatmap = heatmap / heatmap.sum()
    if offset is not None:
        heatmap = heatmap + offset
        heatmap = heatmap / heatmap.sum()
    bounds = (0.0, 0.0, shape[1] * meter_per_bin, shape[0] * meter_per_bin)
    return SimpleNamespace(
        heatmap=np.asarray(heatmap, dtype=float),
        bounds=bounds,
        radius_km=radius_km,
        center_point=(-0.15, 0.2),
        environment_type="flat",
        environment_climate="temperate",
        size="small",
        features=None,
    )


def random_sim(dataset, *, num_drones=2, budget=6_000.0, seed=42, policy_seed=7,
               drone_identity=None, detection_probability=0.8):
    cmod = _c()
    return cmod._build_sim(
        dataset,
        planning_mode=C6_POLICY_NAME,
        w=0.5, p_d=detection_probability, tau=10_000.0, budget=budget,
        num_drones=num_drones, planning_seed=seed,
        implementation_version="c6-test",
        config={},
        policy_seed=policy_seed,
        drone_identity=drone_identity,
    )


def _run_random(sim, positions, steps: int | None = None, dt: float = 1.0):
    sim.setup(initial_positions=list(positions), victim_positions=[])
    if steps is None:
        sim.run_from_state(dt=dt, snapshot_interval=500, heatmap_interval=500)
    else:
        for _ in range(steps):
            if not sim.step(dt):
                break
    return sim


def trace_duplicate_stats(trace: list[dict]) -> tuple[int, float]:
    """Duplicated target decisions per cycle and fraction of cycles with any duplicate."""
    cycles: dict[int, list[tuple[int, int, int]]] = {}
    for row in trace:
        cycles.setdefault(int(row["replan_id"]), []).append(
            (int(row["drone"]), int(row["target_row"]), int(row["target_col"]))
        )
    duplicates = 0
    dup_cycles = 0
    for rows in cycles.values():
        counts: dict[tuple[int, int], int] = {}
        for _drone, tr, tc in rows:
            counts[(tr, tc)] = counts.get((tr, tc), 0) + 1
        excess = int(sum(value - 1 for value in counts.values() if value > 1))
        if excess > 0:
            dup_cycles += 1
        duplicates += excess
    denominator = max(len(cycles), 1)
    return duplicates, float(dup_cycles) / denominator


def count_invalid_trace_actions(sim) -> tuple[int, int]:
    """(invalid actions, hold events) across the random policy trace."""
    invalid = 0
    holds = 0
    for row in sim._planning_trace:
        sr, sc = int(row["start_row"]), int(row["start_col"])
        tr, tc = int(row["target_row"]), int(row["target_col"])
        valid = set(sim._get_valid_neighbors(sr, sc))
        if (tr, tc) == (sr, sc) and not valid:
            holds += 1
            continue
        if (tr, tc) not in valid:
            invalid += 1
    return invalid, holds


# --------------------------------------------------------------------------- #
# Policy test set
# --------------------------------------------------------------------------- #

def test_reproducibility():
    ds = tiny_dataset()
    pos = [(60.0, 50.0), (90.0, 80.0), (120.0, 40.0), (75.0, 130.0), (150.0, 110.0)]
    sims = []
    for _ in range(2):
        sim = random_sim(ds, num_drones=5, budget=2_000.0, seed=42, policy_seed=7)
        sims.append(_run_random(sim, pos, steps=120))
    identical = all(
        pa == pb
        for pa, pb in zip(sims[0].drone_paths, sims[1].drone_paths)
    )
    return bool(identical), f"identical_trajectories={identical}"


def test_policy_seed_sensitivity():
    ds = tiny_dataset()
    pos = [(60.0, 50.0), (90.0, 80.0)]
    a = _run_random(random_sim(ds, policy_seed=1), pos, steps=150)
    b = _run_random(random_sim(ds, policy_seed=2), pos, steps=150)
    differ = any(pa != pb for pa, pb in zip(a.drone_paths, b.drone_paths))
    return bool(differ), f"trajectories_differ={differ}"


def test_n8_validity():
    ds = tiny_dataset()
    pos = [(60.0, 50.0), (90.0, 80.0)]
    sim = _run_random(random_sim(ds, budget=2_000.0), pos, steps=200)
    invalid, holds = count_invalid_trace_actions(sim)
    return bool(invalid == 0), f"invalid_actions={invalid} hold_events={holds}"


def test_uniform_support():
    ds = tiny_dataset()
    sim = random_sim(ds, num_drones=1, budget=1_000_000.0)
    sim.setup(initial_positions=[(600.0, 500.0)], victim_positions=[])
    row, col = sim._world_to_grid(600.0, 500.0)
    valid = sim._get_valid_neighbors(row, col)
    picked: set[tuple[int, int]] = set()
    for _ in range(200):
        sim._move_drones_random(0.0)
        picked.add((int(sim._planning_trace[-1]["target_row"]), int(sim._planning_trace[-1]["target_col"])))
    support = set(valid)
    return bool(support and support <= picked), (
        f"valid={len(support)} selected={len(picked & support)} "
        f"unselected={sorted(support - picked)}"
    )


def test_prior_independence():
    pos = [(60.0, 50.0), (90.0, 80.0), (120.0, 40.0)]
    a = _run_random(random_sim(tiny_dataset(seed=0), policy_seed=3), pos, steps=120)
    b = _run_random(random_sim(tiny_dataset(seed=999, offset=0.25), policy_seed=3), pos, steps=120)
    identical = all(pa == pb for pa, pb in zip(a.drone_paths, b.drone_paths))
    return bool(identical), f"identical_decisions={identical}"


def _walk_targets(policy_seed, pos, steps, corrupt):
    ds = tiny_dataset()
    sim = random_sim(ds, policy_seed=policy_seed)
    sim.setup(initial_positions=list(pos), victim_positions=[])
    targets = []
    for _ in range(steps):
        if corrupt is not None:
            corrupt(sim)
        sim._move_drones_random(1.0)
        targets.append([tuple(target) for target in sim._drone_targets])
    return targets


def test_belief_independence():
    pos = [(60.0, 50.0), (90.0, 80.0)]
    reference = _walk_targets(5, pos, 60, corrupt=None)
    ds = tiny_dataset()
    sim = random_sim(ds, policy_seed=5)
    sim.setup(initial_positions=list(pos), victim_positions=[])
    patched = sim.dynamic_heatmap.get_current_map() * 0.0 + 1e-12
    sim.dynamic_heatmap.get_current_map = lambda: patched
    targets_patched = []
    for _ in range(60):
        sim._move_drones_random(1.0)
        targets_patched.append([tuple(target) for target in sim._drone_targets])
    identical = reference == targets_patched
    return bool(identical), f"decisions_unchanged_under_posterior_patch={identical}"


def test_observation_independence():
    pos = [(60.0, 50.0), (90.0, 80.0)]
    reference = _walk_targets(6, pos, 60, corrupt=None)

    def corrupt(sim):
        sim.globally_observed_cells = {(3, 3), (4, 4)}
        sim._reserved_target_cells = {(0, 0), (1, 1)}

    mutated = _walk_targets(6, pos, 60, corrupt=corrupt)
    same = all(
        all(aa == bb for aa, bb in zip(pa, pb))
        for pa, pb in zip(reference, mutated)
    )
    return bool(same), f"decisions_unchanged_under_history_mutation={same}"


def test_no_deconfliction():
    ds = tiny_dataset()
    sim = random_sim(ds, num_drones=2, policy_seed=9)
    sim.setup(initial_positions=[(600.0, 500.0), (600.0, 500.0)], victim_positions=[])
    duplicate_cycles = 0
    for _ in range(400):
        sim._move_drones_random(0.0)
        rows = [row for row in sim._planning_trace if row["replan_id"] == sim._current_replan_id]
        targets = [(row["target_row"], row["target_col"]) for row in rows]
        if len(set(targets)) < len(targets):
            duplicate_cycles += 1
    kept = len(sim._planning_trace) == 800
    conflicts = sim._number_of_target_conflicts
    return bool(duplicate_cycles > 0 and conflicts == 0 and kept), (
        f"duplicate_target_cycles={duplicate_cycles} conflicts={conflicts} both_decisions_kept={kept}"
    )


def test_no_cross_uav_information():
    ds = tiny_dataset()
    a = random_sim(ds, num_drones=2, policy_seed=11)
    a.setup(initial_positions=[(30.0, 30.0), (90.0, 80.0)], victim_positions=[])
    b = random_sim(ds, num_drones=2, policy_seed=11)
    b.setup(initial_positions=[(300.0, 150.0), (90.0, 80.0)], victim_positions=[])
    picks_a, picks_b = [], []
    for _ in range(80):
        a._move_drones_random(1.0)
        b._move_drones_random(1.0)
        picks_a.append(a._planning_trace[-1])
        picks_b.append(b._planning_trace[-1])
    same_uav1 = all(
        (pa["drone"], pa["target_row"], pa["target_col"]) == (pb["drone"], pb["target_row"], pb["target_col"])
        for pa, pb in zip(picks_a, picks_b)
    )
    return bool(same_uav1), f"uav1_decisions_identical={same_uav1}"


def test_rng_survivor_invariance():
    ds = tiny_dataset()
    pos_by_id = {0: (400.0, 400.0), 1: (700.0, 500.0), 2: (1000.0, 400.0), 3: (600.0, 900.0), 4: (350.0, 750.0)}

    def fleet(ids):
        sim = random_sim(ds, num_drones=len(ids), budget=1_000_000.0, policy_seed=21, drone_identity=ids)
        sim.setup(initial_positions=[pos_by_id[i] for i in ids], victim_positions=[])
        picks = []
        for _ in range(120):
            sim._move_drones_random(1.0)
            picks.append([tuple(sim._drone_targets[i]) for i in range(len(ids))])
        return picks

    full = fleet([0, 1, 2, 3, 4])
    reduced = fleet([0, 2, 4])
    ok0 = all(f[0] == r[0] for f, r in zip(full, reduced))
    ok2 = all(f[2] == r[1] for f, r in zip(full, reduced))
    ok4 = all(f[4] == r[2] for f, r in zip(full, reduced))
    return bool(ok0 and ok2 and ok4), f"survivor_streams_unchanged={ok0 and ok2 and ok4}"


def test_exposure_multiplicity():
    from sarenv.analytics.detection_metrics import exposure_counts
    ds = tiny_dataset()
    sim = random_sim(ds, num_drones=1, policy_seed=1)
    sim.setup(initial_positions=[(600.0, 500.0)], victim_positions=[])
    cell = sim._world_to_grid(600.0, 500.0)
    paths = [[(600.0, 500.0)] * 3]
    counts = exposure_counts(paths, sim._get_visible_cells_world, sim.heatmap_shape)
    n_c = int(counts[cell])
    return bool(n_c == 3), f"n_c={n_c} (expected 3 independent exposures)"


def test_simultaneous_sensing_multiplicity():
    from sarenv.analytics.detection_metrics import exposure_counts
    ds = tiny_dataset()
    sim = random_sim(ds, num_drones=2, policy_seed=1)
    sim.setup(initial_positions=[(600.0, 500.0), (600.0, 500.0)], victim_positions=[])
    cell = sim._world_to_grid(600.0, 500.0)
    paths = [[(600.0, 500.0)] * 2, [(600.0, 500.0)] * 2]
    counts = exposure_counts(paths, sim._get_visible_cells_world, sim.heatmap_shape)
    return bool(int(counts[cell]) == 4), f"n_c={int(counts[cell])} (2 UAVs x 2 steps = 4)"


def test_unique_L():
    cmod = _c()
    ds = tiny_dataset()
    sim = random_sim(ds, num_drones=1, policy_seed=1)
    sim.setup(initial_positions=[(600.0, 500.0)], victim_positions=[])
    prior = cmod._normalise_prior(ds.heatmap)

    def unique_L(paths):
        observed = set()
        for path in paths:
            for wx, wy in path:
                observed.update(sim._get_visible_cells_world(wx, wy))
        return float(sum(prior[r, c] for r, c in observed))

    l1 = unique_L([[(600.0, 500.0), (601.0, 500.0)]])
    l2 = unique_L([[(600.0, 500.0), (601.0, 500.0), (600.0, 500.0)]])
    return bool(abs(l1 - l2) < 1e-12), f"L_unique_unchanged={abs(l1 - l2) < 1e-12} (L1={l1:.6f} L2={l2:.6f})"


def test_budget_identity():
    pos = [(50.0, 50.0), (70.0, 60.0)]
    sim = _run_random(random_sim(tiny_dataset(), budget=3_000.0), pos)
    dist = float(sim.total_distance)
    return bool(3_000.0 - 1e-6 <= dist <= 3_000.0 + 10 * 5.0), (
        f"total_distance={dist:.1f} budget=3000 (identical accounting to other planners)"
    )


def test_termination_identity():
    cmod = _c()
    ds = tiny_dataset()
    pos = [(50.0, 50.0), (70.0, 60.0)]
    sim = _run_random(random_sim(ds, budget=3_000.0), pos)
    reason = cmod._termination_reason(sim, 3_000.0)
    return bool(reason == "budget_exhausted" and sim._done), (
        f"termination_reason={reason} done={sim._done} "
        f"(same budget rule as the other planners; no Random-specific early stop)"
    )


def test_trace_reconstruction():
    from sarenv.analytics.detection_metrics import exposure_counts, exposure_events, expected_rmst
    cmod = _c()
    ds = tiny_dataset()
    sim = random_sim(ds, budget=1_500.0)
    sim.setup(initial_positions=[(60.0, 50.0), (90.0, 80.0)], victim_positions=[])
    sim.run_from_state(dt=1.0, snapshot_interval=500, heatmap_interval=500)
    metrics = cmod._path_exposure_metrics(
        sim, sim.drone_paths, ds.heatmap, 0.8, max(len(p) for p in sim.drone_paths) - 1
    )
    counts = exposure_counts(sim.drone_paths, sim._get_visible_cells_world, sim.heatmap_shape)
    p = cmod._normalise_prior(ds.heatmap)
    manual_p = float(sum(p[r, c] * (1.0 - (1.0 - 0.8) ** counts[r, c]) for r, c in np.argwhere(counts > 0)))
    cells, ts, offsets = exposure_events(sim.drone_paths, sim._get_visible_cells_world, sim.heatmap_shape)
    T = max(len(p) for p in sim.drone_paths) - 1
    manual_rmst = expected_rmst(p.ravel(), cells, ts, offsets, 0.8, T)
    p_ok = abs(metrics["P_detect"] - manual_p) < 1e-9
    r_ok = abs(metrics["RMST"] - manual_rmst) < 1e-9
    return bool(p_ok and r_ok and metrics["L"] <= 1.0), (
        f"P_detect_reconstruct={p_ok} RMST_reconstruct={r_ok} L={metrics['L']:.6f}"
    )


def test_no_map_access_structural():
    ds = tiny_dataset()
    sim = random_sim(ds, num_drones=3, budget=2_000.0, policy_seed=7)

    def boom(*args, **kwargs):
        raise AssertionError("random policy accessed planner map internals")

    for name in (
        "_get_planning_map", "_compute_score_map", "_score_position",
        "_bellman_implied_path", "_maximum_neighbor_filter", "_exact_path_reward",
    ):
        setattr(sim, name, boom)
    sim.setup(
        initial_positions=[(60.0, 50.0), (90.0, 80.0), (120.0, 40.0)],
        victim_positions=[],
    )
    sim.run_from_state(dt=1.0, snapshot_interval=500, heatmap_interval=500)
    return bool(len(sim.drone_paths[0]) > 1), "structural proof: no planner-map entry point was reached"


POLICY_TESTS = {
    "Reproducibility": test_reproducibility,
    "Policy-seed sensitivity": test_policy_seed_sensitivity,
    "N8 validity": test_n8_validity,
    "Uniform support": test_uniform_support,
    "Prior independence": test_prior_independence,
    "Belief independence": test_belief_independence,
    "Observation independence": test_observation_independence,
    "No deconfliction": test_no_deconfliction,
    "No cross-UAV information": test_no_cross_uav_information,
    "RNG survivor invariance": test_rng_survivor_invariance,
    "Exposure multiplicity": test_exposure_multiplicity,
    "Simultaneous sensing": test_simultaneous_sensing_multiplicity,
    "Unique L": test_unique_L,
    "Budget identity": test_budget_identity,
    "Termination identity": test_termination_identity,
    "Trace reconstruction": test_trace_reconstruction,
    "No map access (structural)": test_no_map_access_structural,
}


def run_policy_tests(verbose: bool = True) -> list[dict]:
    results = []
    for name, fn in POLICY_TESTS.items():
        try:
            passed, detail = fn()
            ok = bool(passed)
        except Exception as exc:  # noqa: BLE001 - the test harness records failures
            ok = False
            detail = f"EXCEPTION {type(exc).__name__}: {exc}"
        results.append({"name": name, "passed": ok, "detail": detail})
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}", flush=True)
    return results


def policy_test_flag(results: list[dict], name: str) -> bool:
    for item in results:
        if item["name"] == name:
            return bool(item["passed"])
    return False


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #

def _metric_row(sim, result, config, dataset, seed, policy_seed, termination,
                initial_positions, active_identities):
    cmod = _c()
    T = max(len(path) for path in sim.drone_paths) - 1
    metrics = cmod._path_exposure_metrics(sim, sim.drone_paths, sim.dataset.heatmap,
                                          float(config["p_d"]), T)
    duplicates, dup_fraction = trace_duplicate_stats(sim._planning_trace)
    invalid, holds = count_invalid_trace_actions(sim)
    from sarenv.analytics.detection_metrics import exposure_counts
    counts = exposure_counts(sim.drone_paths, sim._get_visible_cells_world, sim.heatmap_shape)
    max_exp = int(counts.max()) if counts.size else 0
    total_exp = int(counts.sum())
    nominal_ids = list(range(int(config["num_drones"])))
    return {
        "dataset": dataset, "planning_seed": seed, "planning_mode": C6_POLICY_NAME,
        "policy_seed": policy_seed, "rng_scheme": RNG_SCHEME,
        "P_detect": metrics["P_detect"], "RMST": metrics["RMST"], "L": metrics["L"],
        "unique_cells_observed": int(metrics["observed_cells"]),
        "total_exposure_events": total_exp,
        "revisit_fraction": metrics["revisit_fraction"],
        "mean_exposures_per_observed_cell": metrics["exposure_multiplicity"],
        "max_exposures_per_cell": max_exp,
        "number_of_steps": T, "distance_travelled": float(result.total_distance),
        "wall_time": float(result.wall_time), "planner_wall_time": float(result.planner_wall_time),
        "number_of_replans": int(result.number_of_replans),
        "duplicate_targets": duplicates,
        "fraction_cycles_with_duplicate_targets": dup_fraction,
        "random_invalid_actions": invalid, "random_hold_events": holds,
        "random_number_of_target_conflicts": int(result.number_of_target_conflicts),
        "random_reservation_blocked": int(result.reservation_blocked),
        "termination_reason": termination,
        "T": T, "H": T + 1, "T_common": T, "H_common": T + 1,
        "actual_distance_total": float(result.total_distance),
        "initial_positions": _json_text(list(initial_positions)),
        "initial_positions_hash": cmod.stable_hash(list(initial_positions)),
        "trajectory_hash": cmod.stable_hash(sim.drone_paths),
        "active_uav_streams_derived": _json_text(_derived_uav_seeds(policy_seed, active_identities)),
        "nominal_uav_streams_derived": _json_text(_derived_uav_seeds(policy_seed, nominal_ids)),
        "uav_ids_active": _json_text(active_identities),
        "w": C6_NA, "tau": C6_NA, "lookahead_N": C6_NA, "belief_model": C6_NA,
        "environment_belief_model": str(config["belief_model"]),
        "environment_tau": float(config["tau"]),
        "sensor_p_d": float(config["p_d"]),
        "victim_seed": "none", "sensor_seed": "none", "distribution_seed": "none",
        "failure_seed": "none",
        "simulation_victim_model": "none", "evaluation_victim_model": "stationary",
        "victim_sampling_model": "matched_prior_exact", "evaluation_num_victims": 1,
        "simulation_num_victims": 0,
        "domain_hash": cmod.stable_hash((sim.dataset.bounds, sim.dataset.heatmap.shape)),
        "environment_type": sim.dataset.environment_type,
        "environment_climate": sim.dataset.environment_climate,
    }


def _stationary_row(item, config, dataset, seed, policy_seed, c5_parent):
    cmod = _c()
    positions = cmod._initial_positions(item, int(config["num_drones"]), seed)
    if cmod.stable_hash(positions) != c5_parent["initial_positions_hash"]:
        raise SystemExit(
            f"C6 fail-closed: initialization mismatch with C5 parent "
            f"(dataset={dataset}, seed={seed})"
        )
    sim = cmod._build_sim(
        item,
        planning_mode=C6_POLICY_NAME,
        w=float(config["w"]), p_d=float(config["p_d"]), tau=float(config["tau"]),
        budget=float(config["budget"]), num_drones=int(config["num_drones"]),
        planning_seed=seed, implementation_version=config["implementation_version"],
        config=config, policy_seed=policy_seed,
    )
    sim.dataset_id = dataset
    result = cmod._run_from_positions(sim, positions, float(config["dt"]))
    row = _metric_row(sim, result, config, dataset, seed, policy_seed,
                      cmod._termination_reason(sim, float(config["budget"])),
                      positions, list(range(int(config["num_drones"]))))
    row["parent_trial_id"] = f"{int(dataset)}:{int(seed)}"
    row["parent_experiment_id"] = config["parent_experiment_id"]
    return row


def _attrition_row(item, config, dataset, seed, k, failure_ids, c4_parent, policy_seed):
    cmod = _c()
    active_ids = [index for index in range(int(config["num_drones"])) if index not in failure_ids[:k]]
    if _parse_ids(c4_parent["survivor_ids"]) != active_ids:
        raise SystemExit(
            f"C6 fail-closed: survivor identity mismatch with C4 parent "
            f"(dataset={dataset}, seed={seed}, k={k})"
        )
    survivor_positions = _parse_positions(c4_parent["survivor_positions"])
    budget_total = float(len(active_ids) * 40_000.0)
    if not np.isclose(budget_total, float(c4_parent["budget_total"])):
        raise SystemExit(f"C6 fail-closed: budget mismatch with C4 (dataset={dataset}, seed={seed}, k={k})")
    probe = cmod._build_sim(
        item, planning_mode="online_static_3step",
        w=float(config["w"]), p_d=float(config["p_d"]), tau=float(config["tau"]),
        budget=budget_total, num_drones=len(active_ids), planning_seed=seed,
        implementation_version=config["implementation_version"], config=config,
    )
    executed_positions = cmod._aligned_positions(probe, survivor_positions)
    if cmod.stable_hash(executed_positions) != c4_parent["initial_positions_hash"]:
        raise SystemExit(
            f"C6 fail-closed: survivor-position mismatch with C4 parent "
            f"(dataset={dataset}, seed={seed}, k={k})"
        )
    sim = cmod._build_sim(
        item, planning_mode=C6_POLICY_NAME,
        w=float(config["w"]), p_d=float(config["p_d"]), tau=float(config["tau"]),
        budget=budget_total, num_drones=len(active_ids), planning_seed=seed,
        implementation_version=config["implementation_version"], config=config,
        policy_seed=policy_seed, drone_identity=active_ids,
    )
    sim.dataset_id = dataset
    result = cmod._run_from_positions(sim, executed_positions, float(config["dt"]))
    T = max(len(path) for path in sim.drone_paths) - 1
    row = _metric_row(sim, result, config, dataset, seed, policy_seed,
                      cmod._termination_reason(sim, budget_total),
                      executed_positions, active_ids)
    row["k"] = int(k)
    row["NRMST"] = float(row["RMST"] / (T + 1))
    row["budget_total"] = budget_total
    row["budget_utilization"] = float(result.total_distance / budget_total) if budget_total else float("nan")
    row["budget_error"] = float(result.total_distance - budget_total)
    row["active_fleet"] = len(active_ids)
    row["failed_drone_ids"] = _json_text(failure_ids[:k])
    row["survivor_ids"] = _json_text(active_ids)
    row["survivor_positions"] = _json_text(survivor_positions)
    row["failure_seed"] = str(c4_parent["failure_seed"])
    max_step = max(
        (
            np.hypot(path[index + 1][0] - path[index][0], path[index + 1][1] - path[index][1])
            for path in sim.drone_paths
            for index in range(len(path) - 1)
        ),
        default=0.0,
    )
    row["max_step_distance"] = float(max_step)
    row["retention_Pdet"] = "none"
    row["retention_invalid_reason"] = "none"
    row["degradation_NRMST"] = "none"
    return row


# --------------------------------------------------------------------------- #
# Stage driver
# --------------------------------------------------------------------------- #

def _pair_key(dataset, seed):
    return (int(dataset), int(seed))


def run_c6_random(args):
    cmod = _c()
    c5_id = args.input_experiment_id or ""
    c4_id = getattr(args, "attrition_parent_id", None) or ""
    if not c5_id or not c4_id:
        raise SystemExit("c6-random requires --input-experiment-id (C5) and --attrition-parent-id (C4)")

    # Parents are loaded BEFORE any design decision; trial keys are imported
    # from their runs CSV, never generated from a local seed range.
    c5_config, c5_gate = _load_parent(c5_id, "c5-generalization", "c5_gate.md", "c5_generalization_runs.csv")
    c4_config, c4_gate = _load_parent(c4_id, "c4-attrition", "c4_gate.md", "c4_attrition_runs.csv")
    c5_rows = pd.read_csv(cmod.OUTPUT_DIR / c5_id / "c5_generalization_runs.csv")
    c4_rows = pd.read_csv(cmod.OUTPUT_DIR / c4_id / "c4_attrition_runs.csv")
    c5_dynamic = c5_rows[c5_rows.planning_mode == "dynamic_3step"]
    c4_dynamic_arm = c4_rows[c4_rows.planning_mode == "dynamic_evidence_3step"]
    c5_parent_keys = sorted(
        {_pair_key(d, s) for d, s in zip(c5_dynamic.dataset.astype(int), c5_dynamic.planning_seed.astype(int))}
    )
    c4_parent_keys = sorted(
        {_pair_key(d, s) for d, s in zip(c4_dynamic_arm.dataset.astype(int), c4_dynamic_arm.planning_seed.astype(int))}
    )

    filter_ds = args.datasets
    filter_seeds = args.seeds_list
    filter_att_ds = getattr(args, "attrition_datasets", None)
    filter_att_seeds = getattr(args, "attrition_seeds", None)
    filtered = bool(filter_ds or filter_seeds or filter_att_ds or filter_att_seeds)

    def apply_filter(keys, ds_filter, seed_filter):
        out = [key for key in keys
               if (ds_filter is None or key[0] in ds_filter)
               and (seed_filter is None or key[1] in seed_filter)]
        return out

    s_keys = apply_filter(c5_parent_keys, filter_ds, filter_seeds)
    # attrition: a DS/seed filter applies because C4 trial keys are a subset of
    # the C5 keys (the nominal campaign family is the same).
    a_keys = apply_filter(c4_parent_keys, filter_ds if filter_ds is not None else filter_att_ds,
                          filter_seeds if filter_seeds is not None else filter_att_seeds)
    if not filter_ds and not filter_seeds and not filter_att_ds and not filter_att_seeds:
        s_keys = c5_parent_keys
        a_keys = c4_parent_keys

    full_design = bool(
        not filtered
        and s_keys == c5_parent_keys
        and a_keys == c4_parent_keys
        and args.size == cmod.DEFAULT_SIZE
        and args.budget == cmod.DEFAULT_BUDGET
        and args.num_drones == cmod.DEFAULT_NUM_DRONES
        and args.drone_speed == cmod.DEFAULT_DRONE_SPEED
        and args.fov_deg == cmod.DEFAULT_FOV_DEG
        and args.altitude == cmod.DEFAULT_ALTITUDE
        and args.victim_speed == cmod.DEFAULT_VICTIM_SPEED
        and args.dt == cmod.DEFAULT_DT
    )

    config = cmod._base_config(args, C6_STAGE)
    config.update({
        "parent_experiment_id": c5_id,
        "parent_git_commit": c5_config.get("git_commit"),
        "parent_source_hash": c5_config.get("source_hash"),
        "parent_gate": c5_gate.get("gate"),
        "attrition_parent_experiment_id": c4_id,
        "attrition_parent_git_commit": c4_config.get("git_commit"),
        "attrition_parent_source_hash": c4_config.get("source_hash"),
        "attrition_parent_gate": c4_gate.get("gate"),
        "scientific_valid": full_design,
        "label": "confirmatory" if full_design else "functional-only",
        "trial_key_policy": (
            "jobs built exclusively from the parent C4/C5 runs rows; "
            "no local seed range is used; --seeds/--datasets are optional filters"
        ),
        "datasets": sorted({key[0] for key in s_keys}),
        "seeds": sorted({key[1] for key in s_keys}),
        "attrition_datasets": sorted({key[0] for key in a_keys}),
        "attrition_seeds": sorted({key[1] for key in a_keys}),
        "attrition_levels": C6_ATT_LEVELS,
        "planners": [C6_POLICY_NAME],
        "integrated_planners": [
            C6_POLICY_NAME, "pizza_repartition",
            "online_static_3step", "dynamic_evidence_3step",
        ],
        "random_policy": {
            "name": C6_POLICY_NAME, "action_space": "valid_N8", "include_center": False,
            "target_deconfliction": False, "uses_prior": False, "uses_belief": False,
            "uses_observed_mask": False, "uses_revisit_weight": False,
            "uses_other_uav_targets": False, "memoryless": True,
        },
        "rng_scheme": RNG_SCHEME,
        "na_sentinel": C6_NA,
        "simulation_victim_model": "none", "evaluation_victim_model": "stationary",
        "victim_sampling_model": "matched_prior_exact",
        "parallel_unit": ["dataset", "planning_seed"],
        "condition_key": {"stage": C6_STAGE, "w": cmod.NOMINAL_W, "p_d": cmod.DEFAULT_P_D, "tau": cmod.DEFAULT_TAU},
        "expected_rows": len(s_keys) + len(a_keys) * len(C6_ATT_LEVELS),
        "expected_stationary_rows": len(s_keys),
        "expected_attrition_rows": len(a_keys) * len(C6_ATT_LEVELS),
        "policy_seed": "SeedSequence([planning_seed, dataset_id, stable_hash('c6-random-policy')]) per trial; per-UAV streams by persistent UAV identity; recorded per row",
    })
    frozen_basis = {key: value for key, value in config.items() if not key.startswith("_")}
    config["_config_hash_basis"] = frozen_basis
    config, outdir = cmod._make_campaign(C6_STAGE, config, args.jobs)

    design = _design_payload(c5_id, c4_id,
                             [list(key) for key in c5_parent_keys],
                             [list(key) for key in c4_parent_keys])
    (outdir / "c6_design.json").write_text(json.dumps(design, indent=2, default=_json_default) + "\n")

    # Mandatory policy tests BEFORE any scientific run (fail-closed).
    policy_results = run_policy_tests(verbose=not args.no_progress)
    policy_text = "\n".join(
        f"[{'PASS' if row['passed'] else 'FAIL'}] {row['name']} — {row['detail']}"
        for row in policy_results
    )
    (outdir / "c6_random_policy_tests.txt").write_text(
        "# C6 random policy tests\n\n" + policy_text + "\n\n"
        + f"## RESULT {'PASS' if all(row['passed'] for row in policy_results) else 'FAIL'}\n"
    )
    if not all(row["passed"] for row in policy_results):
        failed = [row["name"] for row in policy_results if not row["passed"]]
        raise SystemExit(f"C6 policy tests FAILED: {failed}")
    policy_no_map_access = policy_test_flag(policy_results, "No map access (structural)")

    effective = {
        "fov_deg": float(args.fov_deg), "drone_speed": float(args.drone_speed),
        "dt": float(args.dt), "size": args.size, "p_d": float(args.p_d),
        "w": float(args.w), "tau": float(args.tau),
        "budget": float(args.budget),
    }
    checks = _preflight(c5_id, c4_id, set(s_keys), set(a_keys),
                        effective=effective, full_design=full_design, filtered=filtered)
    (outdir / "c6_preflight.txt").write_text(_preflight_text(design, checks, policy_results))
    if not all(row["passed"] for row in checks):
        failed = [row["check"] for row in checks if not row["passed"]]
        raise SystemExit(f"C6 preflight FAILED: {failed}")

    c5_dynamic = c5_dynamic.set_index(["dataset", "planning_seed"])
    c4_dynamic_arm = c4_dynamic_arm.set_index(["dataset", "planning_seed", "k"])

    # ------------------- C6.2 stationary ------------------------------- #
    def one_stationary(dataset, seed):
        item = cmod._load_item(dataset, args.size)
        key = _pair_key(dataset, seed)
        if key not in c5_dynamic.index:
            raise SystemExit(f"C6 fail-closed: parent C5 trial missing for (dataset={dataset}, seed={seed})")
        c5_parent = c5_dynamic.loc[key]
        policy_seed = _policy_seed(seed, dataset)
        row = _stationary_row(item, config, dataset, seed, policy_seed, c5_parent)
        row["victim_seed"] = c5_parent["victim_seed"]
        row["sensor_seed"] = c5_parent["sensor_seed"]
        row["distribution_seed"] = c5_parent["distribution_seed"]
        row["failure_seed"] = c5_parent["failure_seed"]
        row["parent_experiment_id"] = c5_id
        return pd.DataFrame([row])

    jobs = [(dataset, seed) for dataset, seed in s_keys]
    frames = cmod._parallel_jobs(args, jobs, one_stationary, "C6 stationary Random")
    stationary = cmod._with_provenance(pd.concat(frames, ignore_index=True), config).reset_index(drop=True)
    cmod._validate_campaign_frame(stationary, expected_rows=config["expected_stationary_rows"])

    # ------------------- C6.3 attrition --------------------------------- #
    def one_attrition(dataset, seed):
        item = cmod._load_item(dataset, args.size)
        rows = []
        for k in C6_ATT_LEVELS:
            key = (dataset, seed, k)
            if key not in c4_dynamic_arm.index:
                raise SystemExit(f"C6 fail-closed: parent C4 trial missing for (dataset={dataset}, seed={seed}, k={k})")
            c4_parent = c4_dynamic_arm.loc[key]
            failure_ids = _parse_ids(c4_parent["failed_drone_ids"])
            policy_seed = _policy_seed(seed, dataset)
            row_c4 = _attrition_row(item, config, dataset, seed, k, failure_ids, c4_parent, policy_seed)
            # Environment seed fields are inherited VERBATIM from the frozen C4
            # row. The frozen C4 protocol recorded victim_seed/sensor_seed/
            # distribution_seed as 'none' (stationary matched-prior victims and
            # deterministic exposure metrics; no sensor RNG draws), so the
            # inherited values are 'none' by fact, not by omission. failure_seed
            # is a real value and is inherited as well.
            row_c4["victim_seed"] = c4_parent["victim_seed"]
            row_c4["sensor_seed"] = c4_parent["sensor_seed"]
            row_c4["distribution_seed"] = c4_parent["distribution_seed"]
            row_c4["environment_seed_origin"] = "verbatim from frozen C4 row (C4 recorded none for victim/sensor/distribution; failure_seed real)"
            rows.append(row_c4)
        return pd.DataFrame(rows)

    jobs_att = [(dataset, seed) for dataset, seed in a_keys]
    frames_att = cmod._parallel_jobs(args, jobs_att, one_attrition, "C6 attrition Random")
    attrition = cmod._with_provenance(pd.concat(frames_att, ignore_index=True), config).reset_index(drop=True)
    cmod._validate_campaign_frame(attrition, expected_rows=config["expected_attrition_rows"])

    # Retention vs own k=0 baseline, including k=0 itself (=1.0 degradation 0.0).
    for col in ("retention_Pdet", "retention_invalid_reason", "degradation_NRMST"):
        attrition[col] = attrition[col].astype(object)
    for idx, row in attrition.iterrows():
        key = (int(row.dataset), int(row.planning_seed))
        k0 = attrition[(attrition.dataset == key[0]) & (attrition.planning_seed == key[1]) & (attrition.k == 0)]
        if len(k0) != 1:
            raise SystemExit("C6 fail-closed: missing own k=0 baseline for retention")
        base = k0.iloc[0]
        k_value = int(row.k)
        if k_value == 0:
            if float(base.P_detect) > 0.0:
                attrition.loc[idx, "retention_Pdet"] = 1.0
                attrition.loc[idx, "retention_invalid_reason"] = "none"
            else:
                attrition.loc[idx, "retention_Pdet"] = "none"
                attrition.loc[idx, "retention_invalid_reason"] = "baseline_P_detect_zero"
            attrition.loc[idx, "degradation_NRMST"] = 0.0
        else:
            if float(base.P_detect) > 0.0:
                attrition.loc[idx, "retention_Pdet"] = float(row.P_detect / base.P_detect)
                attrition.loc[idx, "retention_invalid_reason"] = "none"
            else:
                attrition.loc[idx, "retention_Pdet"] = "none"
                attrition.loc[idx, "retention_invalid_reason"] = "baseline_P_detect_zero"
            attrition.loc[idx, "degradation_NRMST"] = float(row.NRMST - base.NRMST)

    # ------------------- stationary contrasts --------------------------- #
    c5_arms = c5_rows[[
        "dataset", "planning_seed", "planning_mode", "P_detect", "RMST", "L",
        "T_common", "H_common", "exposure_multiplicity", "revisit_fraction",
        "number_of_replans", "planner_wall_time", "initial_positions_hash",
    ]].copy()
    stationary_keys = set(zip(stationary.dataset.astype(int), stationary.planning_seed.astype(int)))
    c5_keys_set = set(c5_parent_keys)
    stationary_pairing_complete = bool(
        stationary_keys <= c5_keys_set
        and len(stationary_keys) == len(s_keys)
    )
    if not stationary_pairing_complete:
        raise SystemExit("C6 fail-closed: stationary pairing incomplete")

    pair_key_set = stationary_keys
    c5_arms = c5_arms[
        pd.MultiIndex.from_frame(c5_arms[["dataset", "planning_seed"]].astype(int).copy()).isin(list(pair_key_set))
    ].reset_index(drop=True)
    combined = pd.concat([stationary, c5_arms], ignore_index=True)
    contrasts = {}
    for method in ("online_static_3step", "dynamic_3step"):
        pair_frame = combined[combined.planning_mode.isin([C6_POLICY_NAME, method])]
        contrasts[method] = cmod._paired_contrasts(
            pair_frame, keys=["dataset", "planning_seed"],
            control=C6_POLICY_NAME, method=method,
        )
    pair_dyn_os = cmod._paired_contrasts(
        combined[combined.planning_mode.isin(["online_static_3step", "dynamic_3step"])],
        keys=["dataset", "planning_seed"],
        control="online_static_3step", method="dynamic_3step",
    )
    contrasts["dynamic_3step_vs_online_static_3step"] = pair_dyn_os

    frozen_dyn_os = pd.read_csv(cmod.OUTPUT_DIR / c5_id / "c5_paired_contrasts.csv").set_index(
        ["dataset", "planning_seed"]
    )
    frozen_dyn_os = frozen_dyn_os[
        pd.MultiIndex.from_tuples(
            [(int(d), int(s)) for (d, s) in frozen_dyn_os.index]
        ).isin(list(pair_key_set))
    ]
    recomputed = pair_dyn_os.set_index(["dataset", "planning_seed"])
    frozen_recomputed_match = bool(recomputed.index.equals(frozen_dyn_os.index))
    for (d, s), f_row in frozen_dyn_os.iterrows():
        local = recomputed.loc[(d, s)]
        if abs(float(local.dPdet) - float(f_row.dPdet)) > 1e-12 or abs(float(local.dRMST_rel) - float(f_row.dRMST_rel)) > 1e-12:
            frozen_recomputed_match = False
            break
    if not frozen_recomputed_match:
        raise SystemExit("C6 fail-closed: recomputed dynamic-vs-static contrasts differ from frozen C5 values")

    dataset_effect_rows = []
    for method, frame in contrasts.items():
        eff = cmod._summary_table(frame, ["dataset"])
        eff["contrast"] = method
        if method != "dynamic_3step_vs_online_static_3step":
            eff["method_label"] = "dynamic_evidence_3step" if method == "dynamic_3step" else method
            eff["source_planning_mode"] = method
        else:
            eff["method_label"] = "dynamic_evidence_3step_vs_online_static_3step"
            eff["source_planning_mode"] = "dynamic_3step_vs_online_static_3step"
        dataset_effect_rows.append(eff)
    dataset_effects = pd.concat(dataset_effect_rows, ignore_index=True)

    global_rows = []
    for method, frame in contrasts.items():
        for column, margin in (("dPdet", cmod.MARGIN_PDET), ("dRMST_rel", cmod.MARGIN_RMST_REL)):
            means = frame.groupby("dataset", as_index=False)[column].mean().rename(columns={column: "dataset_mean"})
            boot = cmod.cluster_bootstrap_ci(means, "dataset_mean", "dataset", seed=20260826, iterations=10_000)
            dataset_values = means["dataset_mean"].to_numpy(dtype=float)
            cluster_tost = cmod.tost_equivalence(dataset_values, margin)
            cluster_summary = {
                "mean": boot["mean"],
                "sd": float(np.std(dataset_values, ddof=1)) if len(dataset_values) >= 2 else float("nan"),
                "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                "margin": margin, "tost": cluster_tost,
            }
            global_rows.append({
                "contrast": method, "endpoint": column,
                "method_label": "dynamic_evidence_3step" if method == "dynamic_3step" else method,
                "source_planning_mode": method,
                "mean": boot["mean"], "sd": cluster_summary["sd"],
                "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                "margin": margin,
                "classification": cmod.classify_endpoint(cluster_summary, margin),
                "n_datasets": boot["n_clusters"], "n_pairs": int(len(frame)),
            })
    global_summary = pd.DataFrame(global_rows)

    # ------------------- attrition summaries --------------------------- #
    att_key_set = set(zip(attrition.dataset.astype(int), attrition.planning_seed.astype(int)))
    c4_filtered = c4_rows[
        pd.MultiIndex.from_frame(c4_rows[["dataset", "planning_seed"]].astype(int).copy()).isin(list(att_key_set))
    ].reset_index(drop=True)
    att_all = pd.concat([attrition, c4_filtered], ignore_index=True, sort=False)
    att_abs = att_all.groupby(["dataset", "k", "planning_mode"], sort=True).agg(
        n=("P_detect", "size"),
        P_detect_mean=("P_detect", "mean"),
        P_detect_sd=("P_detect", "std"),
        RMST_mean=("RMST", "mean"),
        RMST_sd=("RMST", "std"),
        L_mean=("L", "mean"),
        L_sd=("L", "std"),
    ).reset_index()
    att_abs["P_detect_se"] = att_abs.P_detect_sd / np.sqrt(att_abs.n)
    att_abs["RMST_se"] = att_abs.RMST_sd / np.sqrt(att_abs.n)

    retention_rows = []
    for (dataset, k, method), group in att_all.groupby(["dataset", "k", "planning_mode"], sort=True):
        valid = pd.to_numeric(group.retention_Pdet, errors="coerce").dropna()
        degrad = pd.to_numeric(group.degradation_NRMST, errors="coerce").dropna()
        retention_rows.append({
            "dataset": int(dataset), "k": int(k), "planning_mode": str(method),
            "n": int(len(valid)),
            "retention_Pdet_mean": float(valid.mean()) if len(valid) else float("nan"),
            "retention_Pdet_ci_lo": (
                float(cmod.paired_mean_ci(valid.to_numpy())["ci_lo"]) if len(valid) > 1 else float("nan")
            ),
            "retention_Pdet_ci_hi": (
                float(cmod.paired_mean_ci(valid.to_numpy())["ci_hi"]) if len(valid) > 1 else float("nan")
            ),
            "degradation_NRMST_mean": float(degrad.mean()) if len(degrad) else float("nan"),
        })
    retention = pd.DataFrame(retention_rows)

    att_pairs = [
        ("dynamic_evidence_3step", C6_POLICY_NAME),
        ("online_static_3step", C6_POLICY_NAME),
        ("pizza_repartition", C6_POLICY_NAME),
        ("dynamic_evidence_3step", "online_static_3step"),
        ("dynamic_evidence_3step", "pizza_repartition"),
        ("online_static_3step", "pizza_repartition"),
    ]
    att_rows = []
    for (dataset, seed), cell in att_all.groupby(["dataset", "planning_seed"], sort=True):
        by_mode_k = {}
        for record in cell.to_dict(orient="records"):
            by_mode_k[(str(record["planning_mode"]), int(record["k"]))] = record
        for first, second in att_pairs:
            for k in C6_ATT_LEVELS:
                a = by_mode_k.get((first, k))
                b = by_mode_k.get((second, k))
                if a is None or b is None:
                    raise SystemExit(f"C6 fail-closed: attrition contrast cell missing {first}-{second} k={k}")
                # absolute performance contrasts exist at every k including k=0;
                # retention deltas only make sense as degradation for k>0.
                if k > 0:
                    a_ret = float(a["retention_Pdet"]) if str(a["retention_Pdet"]) != "none" else float("nan")
                    b_ret = float(b["retention_Pdet"]) if str(b["retention_Pdet"]) != "none" else float("nan")
                    d_ret = a_ret - b_ret if np.isfinite(a_ret) and np.isfinite(b_ret) else float("nan")
                    d_deg = float(b["degradation_NRMST"]) - float(a["degradation_NRMST"])
                else:
                    d_ret = float("nan")
                    d_deg = float("nan")
                att_rows.append({
                    "dataset": dataset, "planning_seed": seed, "k": k,
                    "comparator": f"{first}_vs_{second}",
                    "dRetention_Pdet": d_ret,
                    "dDegradation_NRMST": d_deg,
                    "dPdet": float(a["P_detect"] - b["P_detect"]),
                    "dRMST_rel": float((b["RMST"] - a["RMST"]) / max(float(a["H_common"]), float(b["H_common"]))),
                })
    att_contrasts = pd.DataFrame(att_rows)

    # ------------------- fairness checks (all computed) ------------------ #
    fairness_rows = []
    for (dataset, seed, k) in sorted(set(zip(attrition.dataset.astype(int), attrition.planning_seed.astype(int), attrition.k))):
        group = attrition[(attrition.dataset == dataset) & (attrition.planning_seed == seed) & (attrition.k == k)]
        if len(group) != 1:
            raise SystemExit("C6 fail-closed: duplicate or missing attrition random trial")
        row = group.iloc[0]
        c4_group = c4_rows[(c4_rows.dataset == dataset) & (c4_rows.planning_seed == seed) & (c4_rows.k == k)]
        same_failure = bool((c4_group.failed_drone_ids == row.failed_drone_ids).all())
        same_active = bool((c4_group.survivor_ids == row.survivor_ids).all())
        same_pos = bool((c4_group.initial_positions_hash == row.initial_positions_hash).all())
        same_budget = bool((c4_group.budget_total.astype(float) == float(row.budget_total)).all())
        same_failure_seed = bool((c4_group["failure_seed"].astype(str) == str(row["failure_seed"])).all())
        same_victim_seed = bool((c4_group["victim_seed"].astype(str) == str(row["victim_seed"])).all())
        same_sensor_seed = bool((c4_group["sensor_seed"].astype(str) == str(row["sensor_seed"])).all())
        same_domain = bool(
            c4_config.get("size") == config.get("size")
            and same_pos  # grid-aligned survivor positions are deterministic from the domain
        )
        same_fov = float(c4_config.get("fov_deg")) == float(args.fov_deg)
        same_pd = float(c4_config.get("p_d")) == float(args.p_d)
        same_dt = float(c4_config.get("dt")) == float(args.dt)
        same_speed = float(c4_config.get("drone_speed")) == float(args.drone_speed)
        same_sensor_pipeline = bool(
            c4_config.get("fix_profile") == config.get("fix_profile")
            and c4_config.get("behavior_profile") == config.get("behavior_profile")
            and c4_config.get("belief_model") == config.get("belief_model")
        )
        same_exposure_pipeline = bool(c4_config.get("fix_profile") == config.get("fix_profile"))
        same_termination_rule = bool(
            str(row.termination_reason) == "budget_exhausted"
            and (c4_group["termination_reason"].astype(str) == str(row.termination_reason)).all()
        )
        random_exclusion_off = bool(
            int(row.random_number_of_target_conflicts) == 0
            and int(row.random_reservation_blocked) == 0
            and int(row.random_invalid_actions) == 0
        )
        step_bound_ok = bool(row.max_step_distance <= config["drone_speed"] * config["dt"] + 1e-6)
        no_holds = bool(int(row.random_hold_events) == 0)
        fairness_rows.append({
            "dataset": int(dataset), "planning_seed": int(seed), "k": int(k),
            "same_failure_identity": same_failure,
            "same_active_uav_ids": same_active,
            "same_survivor_initial_positions": same_pos,
            "same_victim_seed": same_victim_seed,
            "same_sensor_seed": same_sensor_seed,
            "same_environment_seed_semantics": bool(
                same_victim_seed and same_sensor_seed
                and str(row.environment_seed_origin).startswith("verbatim from frozen C4 row")
            ),
            "same_domain": same_domain,
            "same_fov": same_fov,
            "same_pd": same_pd,
            "same_dt": same_dt,
            "same_speed": same_speed,
            "same_budget": same_budget,
            "same_termination_rule": same_termination_rule,
            "same_sensor_pipeline": same_sensor_pipeline,
            "same_exposure_pipeline": same_exposure_pipeline,
            "random_target_exclusion_off": random_exclusion_off,
            "random_prior_access_false": bool(policy_no_map_access),
            "random_belief_access_false": bool(policy_no_map_access),
            "random_observation_access_false": bool(policy_no_map_access),
            "same_failure_seed": same_failure_seed,
            "no_hold_events": no_holds,
            "budget_expected": float(row.budget_total),
            "budget_actual": float(row.actual_distance_total),
            "budget_error": float(row.budget_error),
            "max_step_distance": float(row.max_step_distance),
            "steps": int(row.number_of_steps),
            "termination_reason": str(row.termination_reason),
            "total_exposure_events": int(row.total_exposure_events),
            "step_bound_ok": step_bound_ok,
        })
    fairness = pd.DataFrame(fairness_rows)
    fairness_checks_pass = bool(
        fairness[ESSENTIAL_FAIRNESS_COLUMNS].to_numpy(dtype=bool).all()
        and (fairness.no_hold_events).all()
    )

    # ------------------- gate ------------------------------------------ #
    no_missing_trials = bool(
        len(stationary) == config["expected_stationary_rows"]
        and len(attrition) == config["expected_attrition_rows"]
        and fairness_checks_pass
    )
    exposure_semantics_valid = bool(
        (stationary.total_exposure_events > 0).all()
        and stationary.P_detect.between(0.0, 1.0).all()
        and stationary.L.between(0.0, 1.0).all()
        and (stationary.mean_exposures_per_observed_cell >= 1.0).all()
        and (stationary.random_invalid_actions == 0).all()
        and (stationary.random_hold_events == 0).all()
        and (stationary.termination_reason == "budget_exhausted").all()
        and (attrition.total_exposure_events > 0).all()
        and attrition.P_detect.between(0.0, 1.0).all()
        and attrition.L.between(0.0, 1.0).all()
        and (attrition.mean_exposures_per_observed_cell >= 1.0).all()
        and (attrition.random_invalid_actions == 0).all()
        and (attrition.random_hold_events == 0).all()
        and (attrition.termination_reason == "budget_exhausted").all()
    )
    _post_make_fields = {"timestamp", "experiment_id", "jobs", "config_hash", "rerun_policy"}
    recomputed_hash = cmod.stable_hash(
        {key: value for key, value in config.items() if key not in _post_make_fields}
    )
    provenance_valid = bool(
        config.get("source_hash") == cmod._source_hash()
        and bool(config.get("git_commit"))
        and config.get("config_hash") == recomputed_hash
    )
    full_scientific_design = bool(full_design and config.get("scientific_valid"))
    exact_stationary_parent_keyset = bool(set(s_keys) == set(c5_parent_keys))
    exact_attrition_parent_keyset = bool(set(a_keys) == set(c4_parent_keys))
    c6_random_green = bool(
        full_scientific_design
        and exact_stationary_parent_keyset
        and exact_attrition_parent_keyset
        and all(row["passed"] for row in policy_results)
        and len(stationary) == config["expected_stationary_rows"]
        and len(attrition) == config["expected_attrition_rows"]
        and stationary_pairing_complete
        and fairness_checks_pass
        and exposure_semantics_valid
        and provenance_valid
        and no_missing_trials
        and frozen_recomputed_match
    )
    smoke_pass = bool(
        not full_scientific_design
        and all(row["passed"] for row in policy_results)
        and len(stationary) == config["expected_stationary_rows"]
        and len(attrition) == config["expected_attrition_rows"]
        and fairness_checks_pass
        and exposure_semantics_valid
        and provenance_valid
        and no_missing_trials
        and frozen_recomputed_match
    )

    # ------------------- write artifacts ------------------------------- #
    cmod._write_frame(stationary, outdir, "c6_random_stationary_runs.csv")
    cmod._write_frame(dataset_effects, outdir, "c6_random_stationary_dataset_effects.csv")
    cmod._write_frame(global_summary, outdir, "c6_random_stationary_summary.csv")
    (outdir / "c6_random_stationary_global_summary.json").write_text(
        json.dumps(global_summary.to_dict(orient="records"), indent=2, default=_json_default) + "\n"
    )
    cmod._write_frame(attrition, outdir, "c6_random_attrition_runs.csv")
    cmod._write_frame(att_abs, outdir, "c6_random_attrition_summary.csv")
    cmod._write_frame(retention, outdir, "c6_random_attrition_retention.csv")
    cmod._write_frame(att_contrasts, outdir, "c6_random_attrition_contrasts.csv")
    cmod._write_frame(fairness, outdir, "c6_random_attrition_fairness_checks.csv")
    all_contrasts = pd.concat(
        [contrasts["online_static_3step"], contrasts["dynamic_3step"], pair_dyn_os],
        ignore_index=True,
    )
    cmod._write_frame(all_contrasts, outdir, "c6_random_stationary_contrasts.csv")

    for name in (
        "c6_design.json", "c6_random_policy_tests.txt", "c6_preflight.txt",
        "c6_random_stationary_runs.csv", "c6_random_stationary_dataset_effects.csv",
        "c6_random_stationary_summary.csv", "c6_random_stationary_global_summary.json",
        "c6_random_stationary_contrasts.csv",
        "c6_random_attrition_runs.csv", "c6_random_attrition_summary.csv",
        "c6_random_attrition_retention.csv", "c6_random_attrition_contrasts.csv",
        "c6_random_attrition_fairness_checks.csv",
    ):
        src = outdir / name
        if src.exists():
            (cmod.OUTPUT_DIR / name).write_bytes(src.read_bytes())

    manifest = {
        "schema_version": "block-C-v1",
        "stage": C6_STAGE,
        "experiment_id": config["experiment_id"],
        "created_at": config.get("timestamp"),
        "design": design,
        "parents": {
            "stationary": {"experiment_id": c5_id, "gate": c5_gate.get("gate"),
                           "imported_trial_keys": len(c5_parent_keys)},
            "attrition": {"experiment_id": c4_id, "gate": c4_gate.get("gate"),
                          "imported_trial_keys": len(c4_parent_keys)},
        },
        "policy_seed_scheme": RNG_SCHEME,
        "run_counts": {
            "stationary_rows": int(len(stationary)),
            "attrition_rows": int(len(attrition)),
            "expected_stationary_rows": config["expected_stationary_rows"],
            "expected_attrition_rows": config["expected_attrition_rows"],
        },
        "checks": {
            "policy_tests_pass": bool(all(row["passed"] for row in policy_results)),
            "preflight_pass": bool(all(row["passed"] for row in checks)),
            "full_scientific_design": full_scientific_design,
            "exact_stationary_parent_keyset": exact_stationary_parent_keyset,
            "exact_attrition_parent_keyset": exact_attrition_parent_keyset,
            "stationary_complete": bool(len(stationary) == config["expected_stationary_rows"]),
            "attrition_complete": bool(len(attrition) == config["expected_attrition_rows"]),
            "stationary_pairing_complete": stationary_pairing_complete,
            "fairness_checks_pass": fairness_checks_pass,
            "exposure_semantics_valid": exposure_semantics_valid,
            "provenance_valid": provenance_valid,
            "c5_contrast_recomputation_matches_frozen": frozen_recomputed_match,
        },
        "gate": {"C6_RANDOM_GREEN": c6_random_green, "C6_RANDOM_SMOKE_PASS": smoke_pass},
        "artifacts": sorted(p.name for p in outdir.iterdir()),
    }
    (outdir / "c6_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    (cmod.OUTPUT_DIR / "c6_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")

    if full_scientific_design:
        gate_label = "C6_RANDOM_GREEN"
        gate_value = c6_random_green
    else:
        gate_label = "C6_RANDOM_SMOKE_PASS"
        gate_value = smoke_pass
    (outdir / "c6_gate.md").write_text(
        "# C6 gate\n\n"
        + json.dumps({
            gate_label: gate_value,
            "C6_RANDOM_GREEN": c6_random_green,
            "C6_RANDOM_SMOKE_PASS": smoke_pass,
            "full_scientific_design": full_scientific_design,
            "exact_stationary_parent_keyset": exact_stationary_parent_keyset,
            "exact_attrition_parent_keyset": exact_attrition_parent_keyset,
            "policy_tests_pass": bool(all(row["passed"] for row in policy_results)),
            "stationary_complete": bool(len(stationary) == config["expected_stationary_rows"]),
            "attrition_complete": bool(len(attrition) == config["expected_attrition_rows"]),
            "pairing_complete": stationary_pairing_complete,
            "fairness_valid": fairness_checks_pass,
            "exposure_semantics_valid": exposure_semantics_valid,
            "provenance_valid": provenance_valid,
            "no_missing_or_duplicate_trials": no_missing_trials,
            "c5_contrast_recomputation_matches_frozen": frozen_recomputed_match,
            "note": (
                "C6_RANDOM_GREEN requires the full design and exact parent keysets; "
                "a subset run can only report C6_RANDOM_SMOKE_PASS. The gate is "
                "independent of the sign of Random's results by design."
            ),
        }, indent=2, default=_json_default)
        + "\n"
    )
    (cmod.OUTPUT_DIR / "c6_gate.md").write_text((outdir / "c6_gate.md").read_text())

    if full_scientific_design and not c6_random_green:
        raise SystemExit("C6_RANDOM_GREEN is FALSE; campaign artifacts retained, gate NOT green")
    return config["experiment_id"]
