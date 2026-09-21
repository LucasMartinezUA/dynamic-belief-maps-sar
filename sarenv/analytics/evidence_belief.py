# sarenv/analytics/evidence_belief.py
"""Evidence-based Bayesian belief map with exact exponential recovery to the prior.

Unlike the legacy :class:`DynamicHeatmap`, the evidence belief stores a per-cell
evidence anchor ``ell_anchor`` and computes the posterior lazily as
``posterior = prior * ell / Z`` where ``ell`` recovers toward 1 as time passes.
Recovery is pulled from the anchor (``ell -> 1 -> prior``) rather than
interpolated from the last posterior, so a cell observed at t=0 recovers
correctly. The evidence weight ``ell`` recovers toward 1 without overshooting
it, and as all evidence ages the posterior converges back to the prior (a single
posterior cell may temporarily exceed its prior value after normalization, but
the full map converges to the prior).
"""
from __future__ import annotations

import numpy as np


class EvidenceBelief:
    """
    Belief over victim location with negative-observation evidence and decay.

    Update rule (no victim found): the cell's evidence weight is multiplied by
    ``(1 - detection_probability)``; the posterior is ``prior * ell`` normalized.
    Decay rule: ``ell`` recovers exponentially from the anchor toward 1
    (``ell = 1 - (1 - anchor) * exp(-elapsed / tau)``).

    Public interface mirrors ``DynamicHeatmap`` so ``SARSimulation`` can use
    either belief interchangeably.
    """

    def __init__(
        self,
        static_heatmap: np.ndarray,
        detection_probability: float = 0.8,
        decay_tau: float = 10_000.0,
        time_zero_observations: bool = False,
    ):
        """Args:
            static_heatmap: 2D probability array (prior).
            detection_probability: P(detect | present and observed).
            decay_tau: recovery time constant in seconds. 0 or inf = no recovery.
            time_zero_observations: accepted for interface parity; unused, because
                recovery is always correct (a t=0 observation recovers after t>0).
        """
        if static_heatmap.ndim != 2:
            raise ValueError(f"Heatmap must be 2D, got {static_heatmap.ndim}D")
        if not 0.0 <= detection_probability <= 1.0:
            raise ValueError(
                f"detection_probability must be in [0,1], got {detection_probability}"
            )
        if decay_tau < 0:
            raise ValueError(f"decay_tau must be >= 0, got {decay_tau}")

        self.prior = static_heatmap.astype(np.float64).copy()
        self.posterior = self.prior.copy()
        self.detection_probability = detection_probability
        self.decay_tau = decay_tau
        self.time_zero_observations = time_zero_observations

        # Per-cell evidence weight in [0,1]; 1 = no evidence, 0 = annihilated.
        self.ell_anchor = np.ones_like(self.prior)
        self.last_evidence_time = np.zeros_like(self.prior, dtype=np.float64)
        self.observed_mask = np.zeros_like(self.prior, dtype=bool)

        self._current_time = 0.0
        self._num_updates = 0
        self._total_cells_observed = 0

    def _recovered_ell(self, t: float) -> np.ndarray:
        """Current per-cell evidence weight at time ``t`` (float64)."""
        ell = np.ones_like(self.prior)
        if self.observed_mask.any():
            if self.decay_tau > 0 and not np.isinf(self.decay_tau):
                elapsed = t - self.last_evidence_time[self.observed_mask]
                anchor = self.ell_anchor[self.observed_mask]
                factor = 1.0 - (1.0 - anchor) * np.exp(
                    -np.maximum(elapsed, 0.0) / self.decay_tau
                )
                ell[self.observed_mask] = np.clip(factor, 0.0, 1.0)
            else:
                ell[self.observed_mask] = self.ell_anchor[self.observed_mask]
        return ell

    def get_current_map(self) -> np.ndarray:
        """Return a copy of the current posterior."""
        ell = self._recovered_ell(self._current_time)
        unnorm = self.prior * ell
        s = unnorm.sum()
        posterior = unnorm / s if s > 0 else unnorm
        self.posterior = posterior
        return posterior.copy()

    def get_prior(self) -> np.ndarray:
        """Return a copy of the original prior."""
        return self.prior.copy()

    def update(
        self,
        observed_cells: set,
        current_time: float,
        victim_found: bool = False,
        detected_cells: set | None = None,
    ):
        """Apply a negative-observation update to the observed cells."""
        self._current_time = current_time
        if victim_found or not observed_cells or self.detection_probability == 0.0:
            return

        cells_to_reduce = set(observed_cells)
        if detected_cells:
            cells_to_reduce.difference_update(detected_cells)
        if not cells_to_reduce:
            return

        height, width = self.prior.shape
        rows = []
        cols = []
        for r, c in cells_to_reduce:
            if 0 <= r < height and 0 <= c < width:
                rows.append(r)
                cols.append(c)
        if not rows:
            return

        rows = np.array(rows)
        cols = np.array(cols)

        # Recover to the current time, then carry the new negative evidence.
        ell_now = self._recovered_ell(current_time)[rows, cols]
        self.ell_anchor[rows, cols] = ell_now * (1.0 - self.detection_probability)
        self.last_evidence_time[rows, cols] = current_time
        self.observed_mask[rows, cols] = True

        self._num_updates += 1
        self._total_cells_observed += len(rows)
        self.get_current_map()

    def apply_decay(self, current_time: float):
        """Advance time; recovery to the prior is implicit in ``_recovered_ell``."""
        self._current_time = current_time
        self.get_current_map()

    def reset(self):
        """Reset posterior to prior and clear observation history."""
        self.posterior = self.prior.copy()
        self.ell_anchor = np.ones_like(self.prior)
        self.last_evidence_time = np.zeros_like(self.prior, dtype=np.float64)
        self.observed_mask = np.zeros_like(self.prior, dtype=bool)
        self._current_time = 0.0
        self._num_updates = 0
        self._total_cells_observed = 0

    def get_stats(self) -> dict:
        """Return summary statistics (same keys as ``DynamicHeatmap`` plus the model)."""
        prior_sum = self.prior.sum()
        posterior_sum = self.posterior.sum()
        return {
            "num_updates": self._num_updates,
            "total_cells_observed": self._total_cells_observed,
            "prior_sum": float(prior_sum),
            "posterior_sum": float(posterior_sum),
            "reduction_factor": float(posterior_sum / prior_sum) if prior_sum > 0 else 0.0,
            "cells_ever_observed": int(self.observed_mask.sum()),
            "detection_probability": self.detection_probability,
            "decay_tau": self.decay_tau,
            "time_zero_observations": self.time_zero_observations,
            "belief_model": "evidence",
        }
