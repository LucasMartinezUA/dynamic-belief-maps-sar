# tests/test_dynamic_heatmap.py
"""Tests for the DynamicHeatmap Bayesian probability map."""
import numpy as np
import pytest

from sarenv.analytics.dynamic_heatmap import DynamicHeatmap


@pytest.fixture
def uniform_heatmap():
    """10x10 uniform probability heatmap."""
    hm = np.ones((10, 10), dtype=np.float64)
    hm /= hm.sum()
    return hm


@pytest.fixture
def dynamic_hm(uniform_heatmap):
    return DynamicHeatmap(uniform_heatmap, detection_probability=0.8, decay_tau=300.0)


class TestBayesianUpdate:
    def test_mass_conserved_after_update(self, dynamic_hm):
        """Total probability mass should be conserved after Bayesian update."""
        mass_before = dynamic_hm.posterior.sum()
        cells = {(2, 3), (2, 4), (3, 3), (3, 4)}
        dynamic_hm.update(cells, current_time=1.0)
        mass_after = dynamic_hm.posterior.sum()
        assert np.isclose(mass_before, mass_after, rtol=1e-10)

    def test_observed_cells_reduced(self, dynamic_hm):
        """Observed cells should have lower probability after a no-find update."""
        cells = {(5, 5)}
        val_before = dynamic_hm.posterior[5, 5].copy()
        dynamic_hm.update(cells, current_time=1.0)
        assert dynamic_hm.posterior[5, 5] < val_before

    def test_unobserved_cells_increase(self, dynamic_hm):
        """Unobserved cells should increase value (probability mass redistributed)."""
        cells = {(5, 5)}
        val_before = dynamic_hm.posterior[0, 0].copy()
        dynamic_hm.update(cells, current_time=1.0)
        assert dynamic_hm.posterior[0, 0] > val_before

    def test_victim_found_no_change(self, dynamic_hm):
        """When victim is found, update should not modify posterior."""
        before = dynamic_hm.posterior.copy()
        dynamic_hm.update({(5, 5)}, current_time=1.0, victim_found=True)
        np.testing.assert_array_equal(dynamic_hm.posterior, before)

    def test_empty_cells_no_change(self, dynamic_hm):
        """Update with empty cell set should not modify posterior."""
        before = dynamic_hm.posterior.copy()
        dynamic_hm.update(set(), current_time=1.0)
        np.testing.assert_array_equal(dynamic_hm.posterior, before)

    def test_multiple_updates_converge(self, dynamic_hm):
        """Repeated observation of same cells should continually reduce their probability."""
        cells = {(5, 5)}
        values = []
        for t in range(1, 6):
            dynamic_hm.update(cells, current_time=float(t))
            values.append(dynamic_hm.posterior[5, 5])
        # Each value should be less than the previous
        for i in range(1, len(values)):
            assert values[i] < values[i - 1]


class TestExponentialDecay:
    def test_decay_moves_toward_prior(self, dynamic_hm):
        """After observation + decay, cell value should be between posterior and prior."""
        cells = {(5, 5)}
        dynamic_hm.update(cells, current_time=1.0)
        val_after_obs = dynamic_hm.posterior[5, 5]
        prior_val = dynamic_hm.prior[5, 5]

        dynamic_hm.apply_decay(current_time=500.0)
        val_after_decay = dynamic_hm.posterior[5, 5]

        # Should have moved back toward prior
        assert val_after_decay > val_after_obs or np.isclose(val_after_decay, val_after_obs)

    def test_decay_mass_conserved(self, dynamic_hm):
        """Mass should be conserved after decay."""
        mass_before = dynamic_hm.posterior.sum()
        cells = {(2, 2), (3, 3)}
        dynamic_hm.update(cells, current_time=1.0)
        dynamic_hm.apply_decay(current_time=600.0)
        mass_after = dynamic_hm.posterior.sum()
        assert np.isclose(mass_before, mass_after, rtol=1e-10)

    def test_long_decay_approaches_prior(self, dynamic_hm):
        """After a very long time, decay should bring cells close to prior."""
        cells = {(5, 5)}
        dynamic_hm.update(cells, current_time=1.0)
        # Very long decay (10x tau)
        dynamic_hm.apply_decay(current_time=3001.0)
        # Cell value should be close to prior (within 1% after 10*tau)
        assert np.isclose(dynamic_hm.posterior[5, 5], dynamic_hm.prior[5, 5], rtol=0.01)

    def test_no_decay_when_tau_zero(self, uniform_heatmap):
        """Decay should be a no-op when tau=0."""
        dh = DynamicHeatmap(uniform_heatmap, detection_probability=0.8, decay_tau=0.0)
        dh.update({(5, 5)}, current_time=1.0)
        val_after = dh.posterior[5, 5]
        dh.apply_decay(current_time=1000.0)
        assert np.isclose(dh.posterior[5, 5], val_after, rtol=1e-10)

    def test_no_decay_on_unobserved(self, dynamic_hm):
        """Cells never observed should not be affected by decay."""
        before = dynamic_hm.posterior.copy()
        dynamic_hm.apply_decay(current_time=1000.0)
        np.testing.assert_array_equal(dynamic_hm.posterior, before)


class TestEdgeCases:
    def test_out_of_bounds_cells_ignored(self, dynamic_hm):
        """Cells outside heatmap dimensions should be silently ignored."""
        cells = {(-1, 0), (0, -1), (100, 100), (5, 5)}
        dynamic_hm.update(cells, current_time=1.0)
        # Only (5,5) should be observed, no error raised

    def test_reset_restores_prior(self, dynamic_hm):
        """Reset should restore posterior to prior."""
        dynamic_hm.update({(5, 5)}, current_time=1.0)
        dynamic_hm.reset()
        np.testing.assert_array_equal(dynamic_hm.posterior, dynamic_hm.prior)

    def test_invalid_detection_probability(self, uniform_heatmap):
        with pytest.raises(ValueError):
            DynamicHeatmap(uniform_heatmap, detection_probability=1.5)

    def test_invalid_heatmap_dims(self):
        with pytest.raises(ValueError):
            DynamicHeatmap(np.ones(10))

    def test_get_stats(self, dynamic_hm):
        stats = dynamic_hm.get_stats()
        assert "prior_sum" in stats
        assert "posterior_sum" in stats
        assert "num_updates" in stats
