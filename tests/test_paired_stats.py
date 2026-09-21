import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from paired_stats import (  # noqa: E402
    classify_endpoint,
    classify_paired_endpoints,
    cluster_bootstrap_ci,
    paired_mean_ci,
    summarize_paired,
    tost_equivalence,
    validate_paired_keys,
)


def test_paired_mean_ci_is_student_t_and_deterministic():
    values = np.array([1.0, 2.0, 3.0])
    got = paired_mean_ci(values)
    expected_half = stats.t.ppf(0.975, 2) / np.sqrt(3)
    assert got["n"] == 3
    assert got["mean"] == 2.0
    assert got["sd"] == 1.0
    assert np.isclose(got["ci_lo"], 2.0 - expected_half)
    assert np.isclose(got["ci_hi"], 2.0 + expected_half)


def test_tost_constant_inside_margin_is_equivalent():
    result = tost_equivalence(np.full(5, 0.005), margin=0.01)
    assert result["equivalent"] is True
    assert result["p_max"] == 0.0


def test_tost_rejects_exact_boundary_and_outside_margin():
    assert tost_equivalence(np.full(5, 0.01), margin=0.01)["equivalent"] is False
    assert tost_equivalence(np.full(5, 0.02), margin=0.01)["equivalent"] is False


def test_tost_rejects_insufficient_or_nonfinite_vectors():
    assert tost_equivalence([0.0], margin=0.01)["equivalent"] is False
    assert tost_equivalence([0.0, np.nan], margin=0.01)["equivalent"] is False
    assert tost_equivalence([0.0, 0.0], margin=0.0)["equivalent"] is False


def test_classification_uses_ci_and_margin():
    improved = summarize_paired([0.04, 0.05, 0.06], margin=0.01)
    worsened = summarize_paired([-0.04, -0.05, -0.06], margin=0.01)
    equivalent = summarize_paired([0.0, 0.0, 0.0], margin=0.01)
    inconclusive = summarize_paired([0.0, 0.2], margin=0.01)
    assert classify_endpoint(improved, 0.01) == "IMPROVED"
    assert classify_endpoint(worsened, 0.01) == "WORSENED"
    assert classify_endpoint(equivalent, 0.01) == "EQUIVALENT"
    assert classify_endpoint(inconclusive, 0.01) == "INCONCLUSIVE"


def test_joint_classification_adverse_endpoint_has_precedence():
    good = summarize_paired([0.04, 0.05, 0.06], margin=0.01)
    bad = summarize_paired([-0.04, -0.05, -0.06], margin=0.01)
    assert classify_paired_endpoints(good, bad) == "WORSENING"
    assert classify_paired_endpoints(good, summarize_paired([0.0, 0.0, 0.0], margin=0.01)) == "IMPROVEMENT"


def test_pair_key_validation_rejects_collisions_and_missing_arms():
    duplicate = pd.DataFrame(
        {"dataset": [1, 1, 1], "seed": [42, 42, 42], "planning_mode": ["a", "a", "b"]}
    )
    status = validate_paired_keys(duplicate, ["dataset", "seed"], "planning_mode", ["a", "b"])
    assert status["valid"] is False
    assert status["duplicate_cells"]

    missing = duplicate.iloc[:2].copy()
    status = validate_paired_keys(missing, ["dataset", "seed"], "planning_mode", ["a", "b"])
    assert status["valid"] is False
    assert status["incomplete_cells"]


def test_cluster_bootstrap_is_reproducible_and_dataset_weighted():
    frame = pd.DataFrame({"dataset": [1, 2, 3], "effect": [0.0, 1.0, 2.0]})
    left = cluster_bootstrap_ci(frame, "effect", "dataset", seed=20260826, iterations=500)
    right = cluster_bootstrap_ci(frame, "effect", "dataset", seed=20260826, iterations=500)
    assert left == right
    assert left["n_clusters"] == 3
    assert left["mean"] == 1.0
