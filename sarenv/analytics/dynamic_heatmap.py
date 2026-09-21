# sarenv/analytics/dynamic_heatmap.py
"""
Dynamic Bayesian heatmap with FOV-based probability updates and exponential decay.

When a drone observes an area without finding a victim, probability is reduced
in those cells and redistributed. Over time, explored areas decay back toward
the prior (the victim may have moved there).
"""
import numpy as np


class DynamicHeatmap:
    """
    Bayesian probability map that updates with observations and decays over time.

    Update rule (no victim found):
        P_new(cell) = P_old(cell) × (1 - detection_probability)
        Then renormalize so sum(P) is conserved.

    Decay rule (exponential return to prior):
        For each previously-observed cell with elapsed time Δt:
        P(cell, t) = P_posterior + (P_prior - P_posterior) × (1 - exp(-Δt / τ))
        Then renormalize.
    """

    def __init__(
        self,
        static_heatmap: np.ndarray,
        detection_probability: float = 0.8,
        decay_tau: float = 10_000.0,
        time_zero_observations: bool = False,
    ):
        """
        Args:
            static_heatmap: 2D probability array (prior).
            detection_probability: P(detect | present ∧ observed). Range [0, 1].
            decay_tau: Time constant in seconds for exponential decay back to prior.
                       Larger = slower return. 0 or inf = no decay.
            time_zero_observations: If true, observations made at t=0 are
                eligible for later decay. False preserves the paper pipeline,
                where ``last_update_time == 0`` is the legacy sentinel.
        """
        if static_heatmap.ndim != 2:
            raise ValueError(f"Heatmap must be 2D, got {static_heatmap.ndim}D")
        if not 0.0 <= detection_probability <= 1.0:
            raise ValueError(f"detection_probability must be in [0,1], got {detection_probability}")
        if decay_tau < 0:
            raise ValueError(f"decay_tau must be >= 0, got {decay_tau}")

        self.prior = static_heatmap.astype(np.float64).copy()
        self.posterior = self.prior.copy()
        self.detection_probability = detection_probability
        self.decay_tau = decay_tau
        self.time_zero_observations = time_zero_observations

        # Track when each cell was last observed (0 = legacy never-observed
        # sentinel) and retain an explicit mask for the corrected t=0 path.
        self.last_update_time = np.zeros_like(self.prior, dtype=np.float64)
        self.observed_mask = np.zeros_like(self.prior, dtype=bool)
        # Track the posterior value right after the last observation (for decay baseline)
        self.post_observation_value = self.prior.copy()

        self._total_mass = self.prior.sum()
        self._num_updates = 0
        self._total_cells_observed = 0

    def update(
        self,
        observed_cells: set,
        current_time: float,
        victim_found: bool = False,
        detected_cells: set | None = None,
    ):
        """
        Apply Bayesian update to observed cells.

        ``detected_cells`` is used by the corrected sensing-order profile:
        cells where a victim was detected must not be treated as negative
        evidence in the same sensing event.
        """
        if victim_found or not observed_cells or self.detection_probability == 0.0:
            return

        cells_to_reduce = set(observed_cells)
        if detected_cells:
            cells_to_reduce.difference_update(detected_cells)
        if not cells_to_reduce:
            return

        rows = []
        cols = []
        height, width = self.posterior.shape
        for r, c in cells_to_reduce:
            if 0 <= r < height and 0 <= c < width:
                rows.append(r)
                cols.append(c)

        if not rows:
            return

        rows = np.array(rows)
        cols = np.array(cols)

        # Bayesian reduction: P_new = P_old × (1 - p_detect)
        reduction = 1.0 - self.detection_probability
        mass_before = self.posterior.sum()

        self.posterior[rows, cols] *= reduction

        # Renormalize to conserve probability mass
        mass_after = self.posterior.sum()
        if mass_after > 0 and mass_before > 0:
            self.posterior *= mass_before / mass_after

        # Record observation time and post-observation value for decay
        self.last_update_time[rows, cols] = current_time
        self.observed_mask[rows, cols] = True
        self.post_observation_value[rows, cols] = self.posterior[rows, cols]

        self._num_updates += 1
        self._total_cells_observed += len(rows)

    def apply_decay(self, current_time: float):
        """
        Apply exponential decay: observed cells move back toward prior.

        The legacy profile uses ``last_update_time > 0`` as its observation
        mask. The t=0 fix uses the explicit mask so an initial observation is
        not silently excluded from decay.
        """
        if self.decay_tau <= 0 or np.isinf(self.decay_tau):
            return

        if self.time_zero_observations:
            observed_mask = self.observed_mask
        else:
            observed_mask = self.last_update_time > 0
        if not observed_mask.any():
            return

        elapsed = current_time - self.last_update_time[observed_mask]
        positive_mask = elapsed > 0
        if not positive_mask.any():
            return

        obs_indices = np.where(observed_mask)
        decay_rows = obs_indices[0][positive_mask]
        decay_cols = obs_indices[1][positive_mask]
        elapsed_positive = elapsed[positive_mask]

        decay_factor = 1.0 - np.exp(-elapsed_positive / self.decay_tau)
        base_values = self.post_observation_value[decay_rows, decay_cols]
        prior_values = self.prior[decay_rows, decay_cols]

        mass_before = self.posterior.sum()
        self.posterior[decay_rows, decay_cols] = base_values + (prior_values - base_values) * decay_factor

        mass_after = self.posterior.sum()
        if mass_after > 0 and mass_before > 0:
            self.posterior *= mass_before / mass_after


    def get_current_map(self) -> np.ndarray:
        """Return a copy of the current posterior."""
        return self.posterior.copy()

    def get_prior(self) -> np.ndarray:
        """Return a copy of the original prior."""
        return self.prior.copy()

    def reset(self):
        """Reset posterior to prior, clear observation history."""
        self.posterior = self.prior.copy()
        self.post_observation_value = self.prior.copy()
        self.last_update_time[:] = 0.0
        self.observed_mask[:] = False
        self._num_updates = 0
        self._total_cells_observed = 0

    def get_stats(self) -> dict:
        """Return summary statistics."""
        prior_sum = self.prior.sum()
        posterior_sum = self.posterior.sum()
        observed_mask = self.observed_mask if self.time_zero_observations else self.last_update_time > 0
        return {
            "num_updates": self._num_updates,
            "total_cells_observed": self._total_cells_observed,
            "prior_sum": float(prior_sum),
            "posterior_sum": float(posterior_sum),
            "reduction_factor": float(posterior_sum / prior_sum) if prior_sum > 0 else 0.0,
            "cells_ever_observed": int(observed_mask.sum()),
            "detection_probability": self.detection_probability,
            "decay_tau": self.decay_tau,
            "time_zero_observations": self.time_zero_observations,
        }
