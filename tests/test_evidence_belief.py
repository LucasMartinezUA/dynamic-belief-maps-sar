# tests/test_evidence_belief.py
"""Unit tests for the EvidenceBelief Bayesian map (Block B, T1-T12)."""
import numpy as np
import pytest

from sarenv.analytics.dynamic_heatmap import DynamicHeatmap
from sarenv.analytics.evidence_belief import EvidenceBelief


def _uniform(shape):
    hm = np.ones(shape, dtype=np.float64)
    hm /= hm.sum()
    return hm


def test_init():
    prior = np.full((2, 2), 0.25)
    ev = EvidenceBelief(prior)
    np.testing.assert_allclose(ev.get_current_map(), prior)
    assert np.isclose(ev.get_current_map().sum(), 1.0)


def test_one_negative_obs():
    prior = np.full((2, 2), 0.25)
    ev = EvidenceBelief(prior, detection_probability=0.8, decay_tau=np.inf)
    ev.update({(0, 0)}, current_time=1.0)
    # Z = 0.25*0.2 + 0.25*3 = 0.8
    assert np.isclose(ev.posterior[0, 0], 0.0625)
    assert np.isclose(ev.posterior[0, 1], 0.3125)
    assert np.isclose(ev.posterior[1, 0], 0.3125)
    assert np.isclose(ev.posterior[1, 1], 0.3125)
    assert np.isclose(ev.posterior.sum(), 1.0)


def test_k_simultaneous():
    prior = np.full((2, 2), 0.25)
    ev = EvidenceBelief(prior, detection_probability=0.8, decay_tau=np.inf)
    for _ in range(3):
        ev.update({(0, 0)}, current_time=1.0)
    assert np.isclose(ev.ell_anchor[0, 0], (1 - 0.8) ** 3, rtol=1e-12)


def test_order_independence():
    prior = np.full((3, 3), 1 / 9)
    a = EvidenceBelief(prior, detection_probability=0.8, decay_tau=np.inf)
    b = EvidenceBelief(prior, detection_probability=0.8, decay_tau=np.inf)
    for cell in [(0, 0), (1, 1), (2, 2)]:
        a.update({cell}, current_time=1.0)
    for cell in [(2, 2), (1, 1), (0, 0)]:
        b.update({cell}, current_time=1.0)
    np.testing.assert_allclose(a.posterior, b.posterior)


def test_pd_zero():
    prior = np.full((2, 2), 0.25)
    ev = EvidenceBelief(prior, detection_probability=0.0)
    ev.update({(0, 0)}, current_time=1.0)
    np.testing.assert_array_equal(ev.posterior, prior)


def test_legacy_equiv_no_recovery():
    prior = np.arange(4).reshape(2, 2).astype(np.float64)
    prior /= prior.sum()
    ev = EvidenceBelief(prior, 0.8, decay_tau=np.inf)
    lg = DynamicHeatmap(prior, detection_probability=0.8, decay_tau=np.inf,
                        time_zero_observations=False)
    for cells in [{(0, 0)}, {(1, 1)}, {(0, 0)}]:
        ev.update(cells, current_time=1.0)
        lg.update(cells, current_time=1.0)
    np.testing.assert_allclose(ev.posterior, lg.posterior, rtol=1e-12)


def test_convergence():
    # (a) analytic value at 10*tau (deviation from prior ~6.8e-6, so NOT atol=1e-6)
    prior = np.full((2, 2), 0.25)
    ev = EvidenceBelief(prior, 0.8, decay_tau=10.0)
    ev.update({(0, 0)}, 0.0)
    ev.apply_decay(100.0)
    ell = 1.0 - 0.8 * np.exp(-10.0)
    expected = (0.25 * ell) / (0.25 * ell + 0.75)
    assert np.isclose(ev.posterior[0, 0], expected, rtol=1e-9)
    # (b) asymptotic convergence at 20*tau (deviation ~6.6e-10)
    ev2 = EvidenceBelief(prior, 0.8, decay_tau=10.0)
    ev2.update({(0, 0)}, 0.0)
    ev2.apply_decay(200.0)
    np.testing.assert_allclose(ev2.get_current_map(), prior, atol=1e-6)


def test_reobservation_during_recovery():
    prior = np.full((2, 2), 0.25)
    ev = EvidenceBelief(prior, 0.8, decay_tau=10.0)
    ev.update({(0, 0)}, 0.0)
    ev.apply_decay(5.0)
    ev.update({(0, 0)}, 5.0)
    recovered_ell = 1.0 - 0.8 * np.exp(-0.5)  # ~0.5147755
    expect = recovered_ell * 0.2  # ~0.1029551
    assert np.isclose(ev.ell_anchor[0, 0], expect, rtol=1e-6)


def test_t0_obs_recovery():
    prior = np.full((2, 2), 0.25)
    ev = EvidenceBelief(prior, 0.8, decay_tau=10.0)
    ev.update({(0, 0)}, 0.0)
    p0 = ev.posterior[0, 0]
    assert np.isclose(p0, 0.0625)
    ev.apply_decay(10.0)
    p10 = ev.posterior[0, 0]
    assert p10 > p0          # recovers from the low point
    assert p10 < 0.25        # never overshoots the prior
    ev.apply_decay(5000.0)
    np.testing.assert_allclose(ev.get_current_map(), prior, atol=1e-6)


def test_prior_with_zeros():
    prior = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float64)
    ev = EvidenceBelief(prior, 0.8, decay_tau=np.inf)
    ev.update({(0, 0)}, current_time=1.0)
    assert np.isfinite(ev.posterior).all()
    assert ev.posterior[1, 0] == 0.0
    assert ev.posterior[1, 1] == 0.0
    assert np.isclose(ev.posterior.sum(), 1.0)


def test_stress():
    prior = _uniform((10, 10))
    ev = EvidenceBelief(prior, 0.8, decay_tau=100.0)
    rng = np.random.default_rng(0)
    for _ in range(1000):
        t = float(rng.uniform(0, 500.0))
        cells = {(int(rng.integers(0, 10)), int(rng.integers(0, 10)))}
        if rng.uniform() < 0.5:
            ev.update(cells, current_time=t)
        else:
            ev.apply_decay(t)
    assert np.isfinite(ev.posterior).all()
    assert ev.ell_anchor.min() >= 0.0
    assert ev.ell_anchor.max() <= 1.0
    assert np.isclose(ev.posterior.sum(), 1.0, rtol=1e-9)


def test_reviewer6_counterexample():
    prior = np.array([[0.5], [0.5]], dtype=np.float64)
    ev = EvidenceBelief(prior, 0.8, decay_tau=1.0)
    ev.update({(0, 0)}, 0.0)
    assert np.isclose(ev.posterior[0, 0], 1 / 6, rtol=1e-9)
    assert np.isclose(ev.posterior[1, 0], 5 / 6, rtol=1e-9)
    ev.apply_decay(1000.0)
    np.testing.assert_allclose(ev.posterior, np.array([[0.5], [0.5]]), atol=1e-6)
