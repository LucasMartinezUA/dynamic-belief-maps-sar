#!/usr/bin/env python3
# scripts/paired_stats.py
"""
Shared helpers for *paired* non-parametric analysis of SAREnv study results.

Why this module exists
----------------------
The five planning modes are evaluated on the **same seeds** (column ``trial``)
under otherwise identical conditions (same ``dataset``, same swept parameter).
The observations are therefore matched pairs, not independent samples. The
legacy scripts pulled each mode's values independently with ``.values`` and fed
them to ``mannwhitneyu`` / ``kruskal``, which (a) assume independence and (b)
silently pair-by-array-position if naively swapped for a paired test.

This module provides the one correct primitive everything else builds on:
``paired_matrix`` pivots the long dataframe into a wide
(id) x (planning_mode) table, so that row *i* holds the same seed across modes.
Pairwise/omnibus tests then operate on aligned columns with complete cases.

It also fixes the p-value reporting: Wald/asymptotic p-values underflow to 0.0
or to meaningless magnitudes (1e-242) at n=9000-15000. We floor every reported
p at ``P_FLOOR`` (1e-16) and expose ``fmt_p`` for "< 1e-16" rendering.

Effect sizes
------------
- ``wilcoxon_rank_biserial``: the matched-pairs rank-biserial correlation
  (Kerby 2014), the effect size that actually corresponds to the Wilcoxon
  signed-rank test. Range [-1, +1]; sign follows (x - y).
- ``cohens_d`` is kept for continuity but is parametric (standardized mean
  difference) and is NOT the paired effect size; treat it as descriptive only.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Floor for reported p-values. statsmodels Wald and large-n exact tests routinely
# return 0.0 (underflow) or absurd magnitudes (1e-242); neither is informative.
P_FLOOR = 1e-16

# Columns that, when present, identify a single trial within a fixed condition.
# Any of these found in a dataframe (except 'planning_mode') become the pairing
# key. Extend this list if a study sweeps a parameter under a different name.
ID_CANDIDATES = (
    "dataset",
    "trial",
    "seed",
    "n_drones",
    "num_drones",
    "n_agents",
    "budget_km",
    "budget",
    "victim_model",
    "fov",
    "fov_deg",
)


# --------------------------------------------------------------------------- #
# p-value handling
# --------------------------------------------------------------------------- #

def cap_p(p: float) -> float:
    """Floor a p-value at P_FLOOR. NaN passes through unchanged."""
    if p is None:
        return float("nan")
    p = float(p)
    if np.isnan(p):
        return p
    return max(p, P_FLOOR)


def fmt_p(p: float) -> str:
    """Render a (capped) p-value. Values at/below the floor print as '< 1e-16'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    p = cap_p(p)
    if p <= P_FLOOR:
        return f"< {P_FLOOR:.0e}"
    return f"{p:.2e}"


# --------------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------------- #

def holm_bonferroni_adjusted(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down *adjusted* p-values (monotone, capped at 1).

    Returns an array aligned with the input order. Compare directly against
    alpha (e.g. 0.05); no separate threshold needed.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #

def cohens_d(x, y) -> float:
    """Pooled-SD Cohen's d (parametric, descriptive only — NOT paired)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled)


def wilcoxon_rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation (Kerby 2014).

    r = (W+ - W-) / (W+ + W-), where W+/W- are the sums of signed ranks of the
    nonzero paired differences d = x - y. Range [-1, +1]; r > 0 means x tends
    to exceed y. Returns (r, n_nonzero).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 0
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    total = w_plus + w_minus
    if total == 0:
        return 0.0, n
    return float((w_plus - w_minus) / total), n


def interpret_rank_biserial(r: float) -> str:
    """Magnitude labels aligned with Cliff's-delta thresholds used in the paper
    (|.|<0.147 negligible, <0.33 small, <0.474 medium, else large)."""
    if r is None or np.isnan(r):
        return "n/a"
    a = abs(r)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

def detect_id_cols(df, mode_col: str = "planning_mode") -> list:
    """Return the identifier columns present in *df* (the pairing key)."""
    return [c for c in ID_CANDIDATES if c in df.columns and c != mode_col]


def paired_matrix(df, metric, id_cols=None, modes=None, mode_col="planning_mode"):
    """Pivot long -> wide so each row is one (seed/condition), columns are modes.

    Parameters
    ----------
    df : long-format DataFrame with `mode_col` and `metric` columns.
    metric : value column to spread (e.g. 'likelihood').
    id_cols : pairing key. Auto-detected from ID_CANDIDATES if None.
    modes : optional ordering / subset of mode columns to keep.

    Raises
    ------
    ValueError
        If no identifier columns are available, or if the chosen key is not
        unique per (id, mode) — which would mean rows get silently collapsed
        and pairing would be invalid. The message tells you to widen id_cols.
    """
    if id_cols is None:
        id_cols = detect_id_cols(df, mode_col)
    if not id_cols:
        raise ValueError(
            "paired_matrix: no identifier columns found to pair on. "
            f"Expected at least one of {ID_CANDIDATES}. "
            "Cannot run a paired test without a per-trial key."
        )

    # Collision check: more than one observation per (id, mode) means the key is
    # too coarse (e.g. a swept parameter is missing from id_cols).
    counts = df.pivot_table(
        index=id_cols, columns=mode_col, values=metric, aggfunc="count"
    )
    if np.nanmax(counts.to_numpy(dtype=float)) > 1:
        raise ValueError(
            f"paired_matrix: pairing key {id_cols} is not unique per mode "
            "(>1 observation per cell). Add the swept parameter to id_cols so "
            "each row identifies a single matched trial."
        )

    wide = df.pivot_table(
        index=id_cols, columns=mode_col, values=metric, aggfunc="first"
    )
    if modes is not None:
        keep = [m for m in modes if m in wide.columns]
        wide = wide[keep]
    return wide


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def paired_wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank on aligned vectors x, y.

    Returns (statistic, p_capped, rank_biserial_r, n_pairs). Handles the
    degenerate all-equal case (p=1.0) and tiny samples gracefully.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r, n_nonzero = wilcoxon_rank_biserial(x, y)
    if n < 1 or n_nonzero == 0:
        # No nonzero differences -> no evidence of a shift.
        return float("nan"), 1.0, r, n
    try:
        res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), cap_p(res.pvalue), r, n
    except ValueError:
        return float("nan"), float("nan"), r, n


def friedman_omnibus(wide, modes=None):
    """Friedman test (paired, non-parametric omnibus) on complete cases.

    `wide` is the output of `paired_matrix`. Returns
    (chi2, p_capped, n_complete, modes_used). Needs >= 3 modes and >= ~2 blocks.
    Returns (nan, nan, n, modes) and lets the caller skip if k < 3.
    """
    if modes is not None:
        cols = [m for m in modes if m in wide.columns]
    else:
        cols = list(wide.columns)
    complete = wide[cols].dropna()
    n = len(complete)
    if len(cols) < 3 or n < 2:
        return float("nan"), float("nan"), n, cols
    chi2, p = stats.friedmanchisquare(*[complete[c].values for c in cols])
    return float(chi2), cap_p(p), n, cols
# --------------------------------------------------------------------------- #
# Block C common interval, equivalence, and validation helpers
# --------------------------------------------------------------------------- #

def _finite_values(values):
    """Return a float vector and whether every supplied value is finite."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr, bool(np.isfinite(arr).all())


def paired_mean_ci(values, confidence: float = 0.95) -> dict:
    """Student-t confidence interval for a paired effect vector.

    The helper deliberately reports a calculability flag rather than silently
    dropping invalid effects.  ``n`` is the number of finite observations.
    """
    arr, finite = _finite_values(values)
    finite_arr = arr[np.isfinite(arr)]
    n = int(finite_arr.size)
    if n == 0:
        return {
            "n": 0, "mean": float("nan"), "sd": float("nan"),
            "se": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
            "confidence": float(confidence), "finite": finite,
        }
    mean = float(np.mean(finite_arr))
    if n < 2:
        return {
            "n": n, "mean": mean, "sd": float("nan"), "se": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "confidence": float(confidence), "finite": finite,
        }
    sd = float(np.std(finite_arr, ddof=1))
    se = float(sd / np.sqrt(n))
    if sd == 0.0:
        lo = hi = mean
    else:
        quantile = stats.t.ppf((1.0 + confidence) / 2.0, n - 1)
        lo, hi = mean - quantile * se, mean + quantile * se
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "confidence": float(confidence), "finite": finite,
    }


def tost_equivalence(values, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests (TOST) for ``-margin < mean < margin``.

    The returned p-values test ``mu <= -margin`` and ``mu >= +margin`` in that
    order.  Exact boundaries are intentionally not equivalent.
    """
    arr, finite = _finite_values(values)
    margin = float(margin)
    alpha = float(alpha)
    valid_margin = np.isfinite(margin) and margin > 0.0
    n = int(arr.size)
    finite_arr = arr[np.isfinite(arr)]
    n_finite = int(finite_arr.size)
    mean = float(np.mean(finite_arr)) if n_finite else float("nan")
    sd = float(np.std(finite_arr, ddof=1)) if n_finite >= 2 else float("nan")
    ci = paired_mean_ci(finite_arr, confidence=0.90)
    p_lower = p_upper = float("nan")
    equivalent = False
    if finite and valid_margin and n >= 2:
        if sd == 0.0:
            # A constant vector has a degenerate t statistic.  Strict
            # containment is the only case where both nulls are rejected.
            p_lower = 0.0 if mean > -margin else 1.0
            p_upper = 0.0 if mean < margin else 1.0
        else:
            se = sd / np.sqrt(n)
            p_lower = float(stats.t.sf((mean + margin) / se, n - 1))
            p_upper = float(stats.t.cdf((mean - margin) / se, n - 1))
        equivalent = bool(
            np.isfinite(p_lower)
            and np.isfinite(p_upper)
            and p_lower < alpha
            and p_upper < alpha
            and mean > -margin
            and mean < margin
        )
    p_max = max(p_lower, p_upper) if np.isfinite(p_lower) and np.isfinite(p_upper) else float("nan")
    return {
        "mean": mean,
        "sd": sd,
        "n": n,
        "ci90_lo": ci["ci_lo"],
        "ci90_hi": ci["ci_hi"],
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_max": float(p_max),
        "margin": margin,
        "alpha": alpha,
        "equivalent": equivalent,
        "finite": finite and valid_margin,
    }


def summarize_paired(values, margin: float | None = None, confidence: float = 0.95) -> dict:
    """Summarize a paired effect and optionally attach its TOST result."""
    arr, finite = _finite_values(values)
    finite_arr = arr[np.isfinite(arr)]
    summary = paired_mean_ci(arr, confidence=confidence)
    summary["finite"] = bool(summary["finite"] and finite)
    summary["favorable_count"] = int(np.sum(finite_arr > 0.0))
    summary["non_adverse_count"] = int(np.sum(finite_arr >= 0.0))
    summary["n_total"] = int(arr.size)
    if margin is not None:
        summary["margin"] = float(margin)
        summary["tost"] = tost_equivalence(arr, margin)
    return summary


def validate_paired_keys(frame, keys, arm_col: str, expected_arms) -> dict:
    """Validate one-and-only-one row for every arm in every pairing cell."""
    expected = tuple(expected_arms)
    errors = []
    if not keys:
        errors.append("no pairing keys")
        return {"valid": False, "errors": errors, "duplicate_cells": [], "incomplete_cells": []}
    missing = [key for key in (set(keys) | {arm_col}) if key not in frame.columns]
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
        return {"valid": False, "errors": errors, "duplicate_cells": [], "incomplete_cells": []}
    grouped = frame.groupby(list(keys), dropna=False, sort=False)
    duplicate_cells = []
    incomplete_cells = []
    for cell, group in grouped:
        cell_key = cell if isinstance(cell, tuple) else (cell,)
        counts = group[arm_col].value_counts(dropna=False).to_dict()
        if any(count > 1 for count in counts.values()):
            duplicate_cells.append({"key": list(cell_key), "counts": {str(k): int(v) for k, v in counts.items()}})
        if set(counts) != set(expected) or any(counts.get(arm, 0) != 1 for arm in expected):
            incomplete_cells.append({"key": list(cell_key), "arms": sorted(str(k) for k in counts)})
    if duplicate_cells:
        errors.append("duplicate arm cells")
    if incomplete_cells:
        errors.append("incomplete pairs")
    return {
        "valid": not errors,
        "errors": errors,
        "duplicate_cells": duplicate_cells,
        "incomplete_cells": incomplete_cells,
        "n_cells": int(len(grouped)),
        "expected_arms": list(expected),
    }


def classify_endpoint(summary: dict, margin: float) -> str:
    """Map one endpoint summary to the preregistered four-state classification."""
    margin = float(margin)
    mean = float(summary.get("mean", np.nan))
    ci_lo = float(summary.get("ci_lo", np.nan))
    ci_hi = float(summary.get("ci_hi", np.nan))
    if np.isfinite(mean) and np.isfinite(ci_lo) and mean >= margin and ci_lo > 0.0:
        return "IMPROVED"
    if np.isfinite(mean) and np.isfinite(ci_hi) and mean <= -margin and ci_hi < 0.0:
        return "WORSENED"
    tost = summary.get("tost")
    if tost is not None and bool(tost.get("equivalent", False)):
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def classify_paired_endpoints(pdet_summary: dict, rmst_summary: dict) -> str:
    """Apply the common endpoint-to-gate transition used by every C stage."""
    states = (
        classify_endpoint(pdet_summary, pdet_summary.get("margin", np.inf)),
        classify_endpoint(rmst_summary, rmst_summary.get("margin", np.inf)),
    )
    if "WORSENED" in states:
        return "WORSENING"
    if "IMPROVED" in states:
        return "IMPROVEMENT"
    if states[0] == "EQUIVALENT" and states[1] == "EQUIVALENT":
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def cluster_bootstrap_ci(
    dataset_means,
    value_col: str,
    cluster_col: str,
    seed: int,
    iterations: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap cluster-level means, assigning equal weight to each cluster."""
    if hasattr(dataset_means, "groupby"):
        frame = dataset_means[[cluster_col, value_col]].copy()
        if frame.empty:
            values = np.empty(0, dtype=float)
        else:
            values = frame.groupby(cluster_col, sort=True)[value_col].mean().to_numpy(dtype=float)
    else:
        values = np.asarray(dataset_means, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    values = values[finite]
    n = int(values.size)
    if n == 0:
        return {
            "n_clusters": 0, "mean": float("nan"), "ci_lo": float("nan"),
            "ci_hi": float("nan"), "seed": int(seed), "iterations": int(iterations),
            "confidence": float(confidence), "finite": False,
        }
    mean = float(np.mean(values))
    if n < 2 or iterations <= 0:
        return {
            "n_clusters": n, "mean": mean, "ci_lo": float("nan"),
            "ci_hi": float("nan"), "seed": int(seed), "iterations": int(iterations),
            "confidence": float(confidence), "finite": True,
        }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(int(iterations), n))
    boot = values[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "n_clusters": n,
        "mean": mean,
        "ci_lo": float(np.quantile(boot, tail)),
        "ci_hi": float(np.quantile(boot, 1.0 - tail)),
        "seed": int(seed),
        "iterations": int(iterations),
        "confidence": float(confidence),
        "finite": True,
    }
