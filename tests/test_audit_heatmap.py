"""Regression tests for isolated belief fixes used by Block A."""
import numpy as np

from sarenv.analytics.dynamic_heatmap import DynamicHeatmap


def test_t0_observation_is_eligible_for_decay_only_in_corrected_profile():
    prior = np.full((2, 2), 0.25)
    legacy = DynamicHeatmap(
        prior,
        detection_probability=0.8,
        decay_tau=10.0,
        time_zero_observations=False,
    )
    corrected = DynamicHeatmap(
        prior,
        detection_probability=0.8,
        decay_tau=10.0,
        time_zero_observations=True,
    )
    observed = {(0, 0)}
    legacy.update(observed, current_time=0.0)
    corrected.update(observed, current_time=0.0)
    legacy.apply_decay(current_time=10.0)
    corrected.apply_decay(current_time=10.0)

    assert corrected.posterior[0, 0] > legacy.posterior[0, 0]
    assert np.isclose(corrected.posterior.sum(), prior.sum())
    assert np.isclose(legacy.posterior.sum(), prior.sum())


def test_detected_cells_are_not_reduced_by_corrected_sensing_order():
    prior = np.full((2, 2), 0.25)
    heatmap = DynamicHeatmap(prior, detection_probability=0.8)
    heatmap.update(
        {(0, 0), (0, 1)},
        current_time=1.0,
        detected_cells={(0, 0)},
    )
    assert heatmap.posterior[0, 0] > heatmap.posterior[0, 1]
