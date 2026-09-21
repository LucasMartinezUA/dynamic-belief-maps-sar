import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_block_c as c  # noqa: E402

def _parallel_probe(value):
    return os.getpid(), value * value


def test_seed_bundle_is_deterministic_independent_and_planner_agnostic():
    condition = {"stage": "c3-mobility", "w": 0.5, "p_d": 0.8, "tau": 10_000.0}
    left = c._seed_bundle(1, 42, condition)
    right = c._seed_bundle(1, 42, condition)
    changed = c._seed_bundle(1, 42, {**condition, "tau": 32_000.0})
    assert left == right
    assert len({left["victim_seed"], left["sensor_seed"], left["failure_seed"], left["distribution_seed"]}) == 4
    assert left["victim_seed"] != left["sensor_seed"]
    assert left["condition_key"] == condition
    assert left["victim_seed"] != changed["victim_seed"]


def test_valid_support_and_uniform_mix_include_zero_prior_cells():
    class Sim:
        _valid_domain_mask = np.array([[True, True], [False, True]])

    prior = np.array([[0.0, 1.0], [0.0, 0.0]])
    mask = c.valid_prior_support(Sim(), prior)
    target = c.mix_uniform_target(prior, mask, 0.5)
    assert np.isclose(target.sum(), 1.0)
    assert target[0, 0] > 0.0
    assert target[1, 1] > 0.0
    assert target[1, 0] == 0.0


def test_assign_pizza_paths_minimizes_cost_with_lexicographic_tie_break():
    positions = [(0.9, 0.1), (0.1, 0.9)]
    paths = [LineString([(0.0, 0.0), (1.0, 0.0)]), LineString([(0.0, 0.0), (0.0, 1.0)])]
    ordered, assignment = c._assign_pizza_paths(positions, paths)
    assert assignment == {0: 0, 1: 1}
    assert tuple(ordered[0].coords)[1] == (1.0, 0.0)


def test_assign_pizza_paths_rejects_count_mismatch():
    try:
        c._assign_pizza_paths([(0.0, 0.0)], [LineString([(0.0, 0.0), (1.0, 0.0)]), LineString([(0.0, 0.0), (0.0, 1.0)])])
    except ValueError as exc:
        assert "path count" in str(exc)
    else:
        raise AssertionError("expected count mismatch to fail")


def test_mobility_overlay_uses_common_horizon_and_realized_sensor_stream():
    class Dataset:
        heatmap = np.ones((2, 2), dtype=float)
        features = None

    class Sim:
        dataset = Dataset()
        heatmap_shape = (2, 2)
        bounds = (0.0, 0.0, 2.0, 2.0)
        detection_radius = 0.75

        def _grid_to_world(self, row, col):
            return (float(col) + 0.5, float(row) + 0.5)

    paths = [[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)], [(1.5, 1.5)]]
    result = c._mobility_overlay(Sim(), paths, "random_walk", victim_seed=42, sensor_seed=43, p_d=0.8, T_common=2)
    assert 0.0 <= result["P_detect"] <= 1.0
    assert 0.0 <= result["RMST"] <= 3.0
    assert result["sensor_uniforms_hash"]
    assert result["victim_trajectory_hash"]


def test_validate_campaign_frame_rejects_nulls_and_accepts_complete_pair():
    config = {
        "implementation_version": "revision_C_base@test",
        "git_commit": "a" * 40,
        "source_hash": "b" * 64,
        "experiment_id": "exp",
        "config_hash": "c" * 64,
        "condition_key": {"stage": "test"},
        "condition_key_hash": "d" * 64,
    }
    frame = pd.DataFrame(
        [
            {"dataset": 1, "planning_seed": 42, "planning_mode": "online_static_3step", "P_detect": 0.2, "RMST": 2.0, "L": 0.5, "T_common": 2, "H_common": 3},
            {"dataset": 1, "planning_seed": 42, "planning_mode": "dynamic_3step", "P_detect": 0.3, "RMST": 1.0, "L": 0.6, "T_common": 2, "H_common": 3},
        ]
    )
    frame = c._with_provenance(frame, config)
    result = c._validate_campaign_frame(frame, pair_keys=["dataset", "planning_seed"], expected_arms=["online_static_3step", "dynamic_3step"])
    assert result["valid"] is True
    broken = frame.copy()
    broken.loc[0, "L"] = np.nan
    try:
        c._validate_campaign_frame(broken)
    except ValueError as exc:
        assert "null" in str(exc)
    else:
        raise AssertionError("expected null provenance validation to fail")

def test_build_sim_propagates_revisit_weight_to_constructor():
    with patch.object(c, "SARSimulation") as constructor:
        c._build_sim(
            object(),
            planning_mode="dynamic_3step",
            w=0.75,
            p_d=0.8,
            tau=10_000.0,
            budget=1_000.0,
            num_drones=2,
            planning_seed=42,
            implementation_version="test",
        )
    assert constructor.call_args.kwargs["revisit_weight"] == 0.75

def test_c4_degradation_uses_common_horizon_fields():
    rows = []
    for k, values in {
        0: {"dynamic_evidence_3step": (0.50, 1.00), "online_static_3step": (0.50, 1.00)},
        1: {"dynamic_evidence_3step": (0.40, 0.90), "online_static_3step": (0.30, 0.80)},
    }.items():
        for mode, (pdet, degradation) in values.items():
            rows.append({
                "dataset": 1, "planning_seed": 42, "k": k, "planning_mode": mode,
                "P_detect": pdet, "RMST": 10.0, "NRMST": 0.9 if k else 1.0,
                "retention_Pdet": pdet / 0.50, "degradation_NRMST": degradation,
                "T_common": 10, "H_common": 11,
            })
    contrasts = c._c4_degradation_contrasts(pd.DataFrame(rows), "online_static_3step")
    row = contrasts.iloc[0]
    assert np.isclose(row["dRetention_Pdet_Dyn_OS"], 0.2)
    assert np.isclose(row["dDegradation_NRMST_Dyn_OS"], -0.1)

def test_lost_person_distribution_reports_requested_sample_fraction():
    class Generator:
        def generate_locations(self, n, percent_random_samples):
            return [c.Point(0.5, 0.5)]

    with patch.object(c, "LostPersonLocationGenerator", return_value=Generator()):
        _, diagnostics = c.target_distribution_from_lost_person(
            object(), samples=100, distribution_seed=7, shape=(2, 2), bounds=(0.0, 0.0, 1.0, 1.0)
        )
    assert diagnostics["generated_fraction"] == 0.01
    assert diagnostics["valid_fraction"] == 0.01

def test_c2_alpha_frame_and_c4_comparator_finiteness_ignore_structural_nans():
    frame = c._c2_alpha_frame(
        [
            {"planning_mode": "online_static_3step", "dataset": 1, "planning_seed": 42},
            {"planning_mode": "dynamic_3step", "dataset": 1, "planning_seed": 42},
        ],
        0.5,
    )
    assert frame["condition_id"].tolist() == ["alpha_0.5", "alpha_0.5"]
    contrasts = pd.DataFrame([
        {"comparator": "online_static_3step", "dRetention_Pdet_Dyn_OS": 0.1, "dDegradation_NRMST_Dyn_OS": 0.1, "dRetention_Pdet_Dyn_Pizza": np.nan, "dDegradation_NRMST_Dyn_Pizza": np.nan},
        {"comparator": "pizza_repartition", "dRetention_Pdet_Dyn_OS": np.nan, "dDegradation_NRMST_Dyn_OS": np.nan, "dRetention_Pdet_Dyn_Pizza": 0.1, "dDegradation_NRMST_Dyn_Pizza": 0.1},
    ])
    assert c._c4_degradation_finiteness(contrasts) == {
        "online_static_3step": True,
        "pizza_repartition": True,
    }

def test_c5_gate_represents_heterogeneous_and_limited_outcomes():
    heterogeneous = c._classify_c5_gate(
        scientific_valid=True,
        global_valid=True,
        non_adverse_count=10,
        practically_positive_count=8,
        lower_ok=False,
        global_means=[0.01, -0.001],
        global_classifications=["INCONCLUSIVE", "EQUIVALENT"],
    )
    limited = c._classify_c5_gate(
        scientific_valid=True,
        global_valid=True,
        non_adverse_count=10,
        practically_positive_count=8,
        lower_ok=False,
        global_means=[-0.01, -0.02],
        global_classifications=["INCONCLUSIVE", "WORSENED"],
    )
    assert heterogeneous["gate"] == "HETEROGENEOUS"
    assert limited["gate"] == "LIMITED"
    assert heterogeneous["global_mean_positive"] is True
    assert limited["global_not_worse"] is False

def test_revisit_weight_changes_informative_planner_score():
    heatmap = np.zeros((5, 5), dtype=float)
    heatmap[2, 2] = 1.0
    dataset = SimpleNamespace(
        heatmap=heatmap,
        bounds=(0.0, 0.0, 500.0, 500.0),
        radius_km=0.25,
        size="test",
        center_point=(0.0, 0.0),
        features=None,
        environment_type="flat",
        environment_climate="temperate",
    )
    config = {
        "implementation_version": "sanity",
        "fov_deg": 45.0,
        "altitude": 80.0,
        "drone_speed": 5.0,
        "victim_speed": 0.5,
        "dt": 1.0,
    }
    scores = []
    for weight in (0.0, 1.0):
        sim = c._build_sim(
            dataset,
            planning_mode="online_static_3step",
            w=weight,
            p_d=0.8,
            tau=10_000.0,
            budget=20.0,
            num_drones=1,
            planning_seed=42,
            implementation_version="sanity",
            config=config,
        )
        sim.setup(initial_positions=[(250.0, 250.0)], victim_positions=[])
        sim.globally_observed_cells = {(2, 2)}
        scores.append(sim._score_position(250.0, 250.0, sim._get_planning_map()))
    assert scores[0] == 0.0
    assert scores[1] > scores[0]

def test_c4_arms_construct_probe_and_sim_for_all_modes():
    # Regression (2026-08-27): C4 crashed in two ways before arms ran:
    # 1) the alignment probe used planning_mode="static" without static_paths
    #    (rejected by SARSimulation) — only hit for pizza_repartition;
    # 2) the non-pizza branches referenced the removed sim_planning_mode
    #    variable (NameError). All three modes must construct cleanly.
    heatmap = np.zeros((5, 5), dtype=float)
    heatmap[2, 2] = 1.0
    dataset = SimpleNamespace(
        heatmap=heatmap,
        bounds=(0.0, 0.0, 500.0, 500.0),
        radius_km=0.25,
        size="test",
        center_point=(0.0, 0.0),
        features=None,
        environment_type="flat",
        environment_climate="temperate",
    )
    config = {
        "implementation_version": "sanity",
        "fov_deg": 45.0,
        "altitude": 80.0,
        "drone_speed": 5.0,
        "victim_speed": 0.5,
        "dt": 1.0,
    }
    args = SimpleNamespace(dt=1.0)
    seed = 42
    initial = [
        (250.0, 100.0), (350.0, 100.0), (150.0, 200.0),
        (250.0, 200.0), (350.0, 200.0),
    ]
    failure_ids = [0, 1, 2, 3, 4]

    def fake_run(sim, positions, dt):
        # setup must run so drone_paths/_path_progress exist downstream
        sim.setup(initial_positions=list(positions), victim_positions=[])
        return SimpleNamespace(
            total_distance=0.0, planner_wall_time=1.0, number_of_replans=0,
        )

    for mode in ("pizza_repartition", "dynamic_evidence_3step", "online_static_3step"):
        with patch.object(c, "_run_from_positions", side_effect=fake_run):
            with patch.object(c, "_path_exposure_metrics") as metrics_mock:
                metrics_mock.return_value = {
                    "P_detect": 0.5, "RMST": 10.0, "L": 0.5,
                    "exposure_multiplicity": 1.0, "revisit_fraction": 0.0,
                }
                result = c._run_attrition_arm(
                    dataset, config, 0, seed, 1, mode,
                    initial, failure_ids, 99, args,
                )
        assert result["planning_mode"] == mode
        if mode == "pizza_repartition":
            assert result["pizza_assignment_hash"] != "none"
        assert result["survivor_ids"] != "[]"



    # Regression (2026-08-27): C4 crashed before the pizza arm ran because the
    # alignment probe was built with planning_mode="static" without
    # static_paths, which SARSimulation rejects. The probe only grid-aligns
    # positions; its planning mode must not require paths.
    heatmap = np.zeros((5, 5), dtype=float)
    heatmap[2, 2] = 1.0
    dataset = SimpleNamespace(
        heatmap=heatmap,
        bounds=(0.0, 0.0, 500.0, 500.0),
        radius_km=0.25,
        size="test",
        center_point=(0.0, 0.0),
        features=None,
        environment_type="flat",
        environment_climate="temperate",
    )
    config = {
        "implementation_version": "sanity",
        "fov_deg": 45.0,
        "altitude": 80.0,
        "drone_speed": 5.0,
        "victim_speed": 0.5,
        "dt": 1.0,
    }
    args = SimpleNamespace(dt=1.0)
    seed = 42
    initial = [
        (250.0, 100.0), (350.0, 100.0), (150.0, 200.0),
        (250.0, 200.0), (350.0, 200.0),
    ]
    failure_ids = [0, 1, 2, 3, 4]

    def fake_run(sim, positions, dt):
        # setup must run so drone_paths/_path_progress exist downstream
        sim.setup(initial_positions=list(positions), victim_positions=[])
        return SimpleNamespace(
            total_distance=0.0, planner_wall_time=1.0, number_of_replans=0,
        )

    with patch.object(c, "_run_from_positions", side_effect=fake_run):
        with patch.object(c, "_path_exposure_metrics") as metrics_mock:
            metrics_mock.return_value = {
                "P_detect": 0.5, "RMST": 10.0, "L": 0.5,
                "exposure_multiplicity": 1.0, "revisit_fraction": 0.0,
            }
            result = c._run_attrition_arm(
                dataset, config, 0, seed, 1, "pizza_repartition",
                initial, failure_ids, 99, args,
            )
    assert result["planning_mode"] == "pizza_repartition"
    assert result["pizza_assignment_hash"] != "none"
    assert result["survivor_ids"] != "[]"


def test_full_design_flags_reduced_c2_c3_c4_inputs():
    args = SimpleNamespace(
        size="xlarge", budget=200_000.0, w=0.5, p_d=0.8, tau=10_000.0,
        num_drones=5, fov_deg=45.0, altitude=80.0, drone_speed=5.0,
        victim_speed=0.5, dt=1.0,
    )
    assert c._is_full_scientific_design(args, [1, 7, 10], list(range(42, 52)), [1, 7, 10])
    assert not c._is_full_scientific_design(args, [1], [42], [1, 7, 10])

def test_parallel_jobs_uses_requested_workers_and_preserves_results():
    args = SimpleNamespace(jobs=2, no_progress=True)
    results = c._parallel_jobs(
        args,
        [(value,) for value in range(8)],
        _parallel_probe,
        "parallel probe",
    )
    assert sorted(square for _, square in results) == [
        value * value for value in range(8)
    ]
    assert len({pid for pid, _ in results}) >= 2
