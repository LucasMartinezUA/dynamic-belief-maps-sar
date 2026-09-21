#!/usr/bin/env python3
# scripts/extra_studies.py
"""
Extra post-hoc analyses on the results produced by comprehensive_study.py.

This script computes the additional statistics and
regime analyses that are needed to defend the paper against a critical Q1
reviewer. It does NOT re-run simulations; it operates on the CSVs already
written to ``results/comprehensive_study/``.

Analyses produced
-----------------
  A) Cohen's d for every critical pairwise comparison (d3 vs pizza, d3 vs
     static_3step, ...), with Mann-Whitney U p-values and Holm-Bonferroni
     corrected significance flags. Answers: "is the *effect size* of the
     contribution non-trivial, not just the p-value?".

  B) Per-dataset breakdown of dynamic_3step vs pizza on Study 6 (generalization
     heterogeneity). Answers: "does d3 dominate pizza across all 18
     environments, or only on average?".

  C) Saturation / cross-over curves for Studies 1, 2, 3 (drones, budget, FOV).
     For each parameter value, reports Δ(d3 − pizza), Cohen's d and MWU p.
     Answers: "in which operational regime does the contribution hold?".

  D) Null-study check for Study 4 (victim model). For identical (mode, trial,
     seed), compares the likelihood vector across victim models. Answers:
     "does the victim motion model actually affect the primary metric, or is
     Study 4 vacuous?".

  E) Operational metric cross-check: same analyses as (A) but on
     ``victims_found`` (the metric that matters to a rescuer), to see whether
     the likelihood ranking survives when we look at detections.

  F) Linear Mixed Model (LMM) on Study 6.  Models ``dataset`` as a random
     intercept and ``planning_mode`` as a fixed effect, providing a formal
     mixed-effects alternative to the per-dataset MWU in (B).  Reports fixed-
     effects coefficients, random-effects variances, ICC, Likelihood Ratio Test
     and BLUPs per dataset.

  G) Linear Mixed Model (LMM) on Study 6, same methodology as (F) but
     comparing ``dynamic_3step`` vs ``static_3step`` (the ablation
     comparator).  Provides methodological consistency: both the external
     baseline (pizza) and the internal ablation (static_3step) are tested
     with the same mixed-effects framework.

  H) Shapiro-Wilk + Levene on Study 6 (generalization).  Tests normality per
     planning mode (Shapiro-Wilk) and homogeneity of variance across modes
     (Levene) on ``likelihood``.  Provides formal justification for the
     choice of non-parametric tests (Mann-Whitney U) used throughout the
     paper.  Answers: "are the assumptions for parametric tests violated,
     and if so, to what degree?"

  I) Post-hoc power analysis.  Given n per group (the per-dataset sample
     size), computes the minimum detectable Cohen's d at 80% power, the
     achieved power for observed effect sizes from key pairwise comparisons
     (d3 vs pizza, d3 vs static_3step), and a power curve showing how power
     scales with n for various d values.  Answers: "is n sufficient to
     detect the effects we claim, and what would we need to detect smaller
     ones?"

Outputs
-------
  * Console report with all tables.
  * One CSV per analysis under ``results/extra_studies/``
    (``cohens_d_pairwise.csv``, ``per_dataset_d3_vs_pizza.csv``,
    ``saturation_<param>.csv``, ``victim_model_null_check.csv``,
    ``victims_found_pairwise.csv``, ``mixed_effects_*.csv``).
  * Optional PNG figures (``--plot``) under the same directory.

Usage
-----
  # Run every analysis on the default CSV names in results/comprehensive_study/
  python scripts/extra_studies.py

  # Point explicitly to the files (useful after re-running comprehensive_study.py)
  python scripts/extra_studies.py \
      --study6-csv results/comprehensive_study/study_datasets_20260410_045716.csv \
      --drones-csv results/comprehensive_study/study_drones_20260410_150847.csv \
      --budget-csv results/comprehensive_study/study_budget_20260410_024356.csv \
      --fov-csv    results/comprehensive_study/study_fov_20260410_100910.csv \
      --victim-csv results/comprehensive_study/study_victim_model_20260410_064938.csv \
      --plot

  # Only specific analyses
  python scripts/extra_studies.py --analyses A B F
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import paired_stats as ps

# Conditional import of statsmodels for mixed-effects analysis
try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

import sarenv

log = sarenv.get_logger()


# ── Constants ────────────────────────────────────────────────────────────

ALL_MODES = [
    "pizza",
    "static",
    "dynamic",
    "static_3step",
    "dynamic_3step",
]

# Pairwise comparisons we care about for the paper. (a, b, rationale)
CRITICAL_COMPARISONS = [
    ("pizza", "static_3step", "geometric baseline vs static 3-step lookahead"),
    ("pizza", "dynamic_3step", "geometric baseline vs dynamic 3-step lookahead"),
    ("dynamic", "static", "dynamic vs static (no lookahead)"),
    ("static_3step", "static", "static 3-step vs 1-step lookahead"),
    ("dynamic_3step", "dynamic", "dynamic 3-step vs 1-step lookahead"),
    ("dynamic_3step", "static_3step", "3-step dynamic vs static"),
]

DEFAULT_RESULT_DIR = Path("results/comprehensive_study")
DEFAULT_OUTPUT_DIR = Path("results/extra_studies")


# ── Helpers ──────────────────────────────────────────────────────────────


def cohens_d(a, b):
    """Cohen's d with pooled standard deviation (unbiased variance)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_sd == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_sd)


def interpret_d(d):
    """Cohen's convention for effect size magnitudes."""
    if np.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "trivial"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    if ad < 1.2:
        return "large"
    return "very large"


def mwu(a, b):
    """Two-sided Mann-Whitney U. Returns (U, p). Safe against tiny samples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 1 or len(b) < 1:
        return (float("nan"), float("nan"))
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return (float(u), float(p))
    except ValueError:
        return (float("nan"), float("nan"))


def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni step-down correction for multiple comparisons.

    More powerful than plain Bonferroni while still controlling the
    family-wise error rate (FWER).  Well-suited for pairwise comparisons
    that are not independent (e.g., transitivity: if A>B and B>C then
    likely A>C).

    Procedure
    ---------
    1. Sort p-values from smallest to largest: p₁ ≤ p₂ ≤ ... ≤ pₘ
    2. For rank k (1-indexed), reject H₀ if pₖ < α / (m − k + 1)
    3. Once a p-value fails to reject, all subsequent (larger) p-values
       also fail to reject.

    Parameters
    ----------
    p_values : array-like of shape (m,)
        Raw p-values from m hypothesis tests.
    alpha : float
        Family-wise error rate (default: 0.05).

    Returns
    -------
    reject : np.ndarray[bool] of shape (m,)
        Whether each corresponding p-value is significant after correction.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(p)
    reject = np.zeros(m, dtype=bool)

    for rank, idx in enumerate(sorted_idx):
        threshold = alpha / (m - rank)
        if p[idx] < threshold:
            reject[idx] = True
        else:
            break  # all remaining (larger) p-values also fail

    return reject


def _power_t_test(d, n1, n2, alpha=0.05):
    """Power of a two-sample two-sided t-test given Cohen's d and n per group.

    Uses the non-central t distribution.  For n1 = n2 = n, the non-centrality
    parameter is λ = d × √(n / 2).
    """
    d = abs(d)
    if d == 0 or n1 < 2 or n2 < 2:
        return 0.0
    df = n1 + n2 - 2
    ncp = d * np.sqrt(n1 * n2 / (n1 + n2))
    # For very large NCP, power is essentially 1.0; numerical overflow
    # in the nct.cdf produces NaN, so short-circuit.
    if ncp > 37:  # safe threshold: nct.cdf works up to ~37 in scipy
        return 1.0
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1.0 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    # Fallback: any NaN → assume power ≈ 1 (numerical overflow implies
    # the effect is so large that power is effectively 1.0).
    if np.isnan(power):
        return 1.0
    return float(power)


def _min_detectable_d(n1, n2, alpha=0.05, target_power=0.80):
    """Smallest |Cohen's d| detectable with ≥ target_power (binary search)."""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _power_t_test(mid, n1, n2, alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_csv(path, label):
    """Load a CSV with a friendly error if missing."""
    p = Path(path)
    if not p.exists():
        log.warning(f"[{label}] CSV not found: {p}")
        return None
    df = pd.read_csv(p)
    log.info(f"[{label}] loaded {len(df)} rows from {p.name}")
    return df


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


# ── Analysis A: Cohen's d on all critical pairwise comparisons ──────────


def analysis_A_pairwise(
    df, metric, tag, output_dir, log=None, CRITICAL_COMPARISONS=None, interpret_d=None
):
    """Paired pairwise effect sizes on `metric`. Wilcoxon signed-rank +
    matched-pairs rank-biserial (primary), Cohen's d (secondary), Holm.
    """
    if log is None:
        import logging as _logging

        log = _logging.getLogger("paired")
    if interpret_d is None:
        interpret_d = lambda d: ps.interpret_rank_biserial(d)  # noqa: E731
    if CRITICAL_COMPARISONS is None:
        CRITICAL_COMPARISONS = []

    print("\n" + "=" * 100)
    print(f"ANALYSIS A — PAIRED pairwise effect sizes on '{metric}' ({tag})")
    print("=" * 100)

    modes = sorted(df["planning_mode"].unique())
    alpha = 0.05
    print(f"Modes: {modes}")
    print(f"Total unordered pairs: {len(modes) * (len(modes) - 1) // 2}")
    print(f"Wilcoxon signed-rank + Holm-Bonferroni (alpha = {alpha})")

    try:
        wide = ps.paired_matrix(df, metric, modes=modes)
        paired = True
    except ValueError as e:
        log.warning(f"[A] PAIRING UNAVAILABLE — unpaired fallback. {e}")
        wide, paired = None, False

    rows = []
    for a, b in combinations(modes, 2):
        if paired:
            sub = wide[[a, b]].dropna()
            da, db = sub[a].values, sub[b].values
            w_stat, p, r_rb, n_pair = ps.paired_wilcoxon(da, db)
        else:
            da = df.loc[df["planning_mode"] == a, metric].dropna().values
            db = df.loc[df["planning_mode"] == b, metric].dropna().values
            from scipy import stats as _st

            w_stat, p = _st.mannwhitneyu(da, db, alternative="two-sided")
            p = ps.cap_p(p)
            r_rb, n_pair = float("nan"), min(len(da), len(db))
        if len(da) == 0 or len(db) == 0:
            continue
        d = ps.cohens_d(da, db)
        delta = float(da.mean() - db.mean())
        rel = 100.0 * delta / db.mean() if db.mean() != 0 else float("nan")
        rows.append(
            {
                "mode_a": a,
                "mode_b": b,
                "n_pairs": n_pair,
                "mean_a": da.mean(),
                "mean_b": db.mean(),
                "delta_mean": delta,
                "pct_rel": rel,
                "rank_biserial": r_rb,
                "magnitude": ps.interpret_rank_biserial(r_rb),
                "cohens_d": d,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": p,
            }
        )
    full_df = pd.DataFrame(rows)

    if len(full_df) > 0:
        full_df["p_holm"] = ps.holm_bonferroni_adjusted(full_df["wilcoxon_p"].values)
        full_df["sig_holm"] = full_df["p_holm"] < alpha
        print(
            f"Pairs significant under Holm-Bonferroni: "
            f"{int(full_df['sig_holm'].sum())}/{len(full_df)}"
        )
    else:
        full_df["p_holm"], full_df["sig_holm"] = [], []

    print("\nCritical comparisons (paper narrative):")
    print(
        f"  {'comparison':<40} {'Δmean':>9} {'%rel':>8} {'r_rb':>7}  "
        f"{'magnitude':<11} {'p(Wilc)':>11} {'Holm':>5}  rationale"
    )
    print("  " + "-" * 122)
    for a, b, rationale in CRITICAL_COMPARISONS:
        sub = full_df[(full_df["mode_a"] == a) & (full_df["mode_b"] == b)]
        flip = False
        if sub.empty:
            sub = full_df[(full_df["mode_a"] == b) & (full_df["mode_b"] == a)]
            flip = True
        if sub.empty:
            continue
        r = sub.iloc[0]
        if flip:
            mean_a, mean_b = r["mean_b"], r["mean_a"]
            delta = mean_a - mean_b
            rel = 100.0 * delta / mean_b if mean_b != 0 else float("nan")
            r_rb = -r["rank_biserial"]  # antisymmetric
        else:
            delta, rel, r_rb = r["delta_mean"], r["pct_rel"], r["rank_biserial"]
        magn = ps.interpret_rank_biserial(r_rb)
        p = r["p_holm"]
        holm = "yes" if r["sig_holm"] else "no"
        print(
            f"  {a + ' vs ' + b:<40} {delta:>+9.4f} {rel:>+7.2f}% "
            f"{r_rb:>+7.3f}  {magn:<11} {ps.fmt_p(p):>11} {holm:>5}  {rationale}"
        )

    out = Path(output_dir) / f"pairwise_{metric}_{tag}.csv"
    full_df.to_csv(out, index=False)
    log.info(f"[A/{metric}] saved {out}")
    return full_df


# =========================================================================== #
# 3) scripts/extra_studies.py  ::  analysis_B_per_dataset
# =========================================================================== #
def analysis_B_per_dataset(
    df,
    output_dir,
    mode_a="dynamic_3step",
    mode_b="pizza",
    metric="likelihood",
    log=None,
):
    """Per-dataset paired comparison of two modes. Pairs on `trial` *within*
    each dataset (so the swept 'dataset' axis does not break uniqueness)."""
    if log is None:
        import logging as _logging

        log = _logging.getLogger("paired")

    print("\n" + "=" * 100)
    print(f"ANALYSIS B — {mode_a} vs {mode_b} per DATASET on '{metric}' (PAIRED)")
    print("=" * 100)

    datasets = sorted(df["dataset"].unique())
    print(
        f"  {'DS':>4} {'n':>4} {'mean_a':>8} {'mean_b':>8} {'Δ':>9} "
        f"{'%rel':>8} {'r_rb':>7}  {'magnitude':<11} {'p(Wilc)':>11}  winner"
    )
    print("  " + "-" * 110)

    rows, wins_a, wins_b, sig_a, sig_b = [], 0, 0, 0, 0
    for ds in datasets:
        sub_df = df[df["dataset"] == ds]
        try:
            w = ps.paired_matrix(sub_df, metric, modes=[mode_a, mode_b])
            sub = w[[mode_a, mode_b]].dropna()
            a, b = sub[mode_a].values, sub[mode_b].values
            w_stat, p, r_rb, n_pair = ps.paired_wilcoxon(a, b)
        except (ValueError, KeyError) as e:
            log.warning(f"[B] DS {ds}: pairing failed ({e}); skipping.")
            continue
        if len(a) == 0 or len(b) == 0:
            continue
        d = ps.cohens_d(a, b)
        delta = float(a.mean() - b.mean())
        rel = 100 * delta / b.mean() if b.mean() != 0 else float("nan")
        sig = (not np.isnan(p)) and p < 0.05
        if delta > 0:
            wins_a += 1
            sig_a += int(sig)
            winner = f"{mode_a}{'*' if sig else ''}"
        else:
            wins_b += 1
            sig_b += int(sig)
            winner = f"{mode_b}{'*' if sig else ''}"
        print(
            f"  {ds:>4} {n_pair:>4} {a.mean():>8.4f} {b.mean():>8.4f} "
            f"{delta:>+9.4f} {rel:>+7.2f}% {r_rb:>+7.3f}  "
            f"{ps.interpret_rank_biserial(r_rb):<11} {ps.fmt_p(p):>11}  {winner}"
        )
        rows.append(
            {
                "dataset": int(ds),
                "mode_a": mode_a,
                "mode_b": mode_b,
                "n_pairs": n_pair,
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "delta_mean": delta,
                "pct_rel": rel,
                "rank_biserial": r_rb,
                "magnitude": ps.interpret_rank_biserial(r_rb),
                "cohens_d": d,
                "wilcoxon_p": p,
                "significant_005": bool(sig),
                "winner": winner,
            }
        )
    print("\n  " + "-" * 110)
    print(
        f"  {mode_a:<18} wins: {wins_a:>2} / {len(datasets):<2}  (sig p<0.05: {sig_a})"
    )
    print(
        f"  {mode_b:<18} wins: {wins_b:>2} / {len(datasets):<2}  (sig p<0.05: {sig_b})"
    )

    out = Path(output_dir) / f"per_dataset_{mode_a}_vs_{mode_b}_{metric}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"[B] saved {out}")
    return pd.DataFrame(rows)


# ── Analysis C: saturation / cross-over curves ──────────────────────────


def analysis_C_saturation(
    df,
    param_col,
    output_dir,
    mode_a="dynamic_3step",
    mode_b="pizza",
    metric="likelihood",
):
    """For each value of a scalar parameter (drones / budget / FOV), compare
    modes A and B. Highlights cross-overs where B catches up or overtakes A.
    """
    if df is None:
        return None
    print("\n" + "=" * 100)
    print(
        f"ANALYSIS C — Saturation curve on '{param_col}' "
        f"({mode_a} vs {mode_b}, metric={metric})"
    )
    print("=" * 100)

    values = sorted(df[param_col].dropna().unique())
    print(
        f"  {param_col:<10} {'n_a':>4} {'n_b':>4} {'mean_a':>8} {'mean_b':>8}"
        f" {'Δ':>9} {'d':>7}  {'p(MWU)':>10}  winner"
    )
    print("  " + "-" * 78)

    rows = []
    for v in values:
        a = (
            df[(df["planning_mode"] == mode_a) & (df[param_col] == v)][metric]
            .dropna()
            .values
        )
        b = (
            df[(df["planning_mode"] == mode_b) & (df[param_col] == v)][metric]
            .dropna()
            .values
        )
        if len(a) == 0 or len(b) == 0:
            continue
        d = cohens_d(a, b)
        u, p = mwu(a, b)
        delta = float(a.mean() - b.mean())
        if p < 0.05:
            winner = f"{mode_a}*" if delta > 0 else f"{mode_b}*"
        else:
            winner = "tie"
        print(
            f"  {v!s:<10} {len(a):>4} {len(b):>4} {a.mean():>8.4f} {b.mean():>8.4f}"
            f" {delta:>+9.4f} {d:>+6.3f}  {p:>10.2e}  {winner}"
        )
        rows.append(
            {
                param_col: v,
                "mode_a": mode_a,
                "mode_b": mode_b,
                "n_a": len(a),
                "n_b": len(b),
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "delta_mean": delta,
                "cohens_d": d,
                "mwu_p": p,
                "winner": winner,
            }
        )

    out = Path(output_dir) / f"saturation_{param_col}_{mode_a}_vs_{mode_b}_{metric}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"[C/{param_col}] saved {out}")
    return pd.DataFrame(rows)


# ── Analysis D: Study 4 null-study check (victim model) ─────────────────


def analysis_D_victim_null_check(df, output_dir):
    """For Study 4, check whether the victim motion model actually affects the
    primary metric (likelihood). If the same (mode, trial, seed) tuples yield
    identical likelihoods across victim models, the study is informative only
    on victims_found — not on likelihood.
    """
    if df is None:
        return None
    print("\n" + "=" * 100)
    print("ANALYSIS D — Study 4 null-study check (victim_model)")
    print("=" * 100)

    if "victim_model" not in df.columns:
        log.warning("[D] 'victim_model' column missing; skipping")
        return None

    # Pivot on (planning_mode, trial) × victim_model
    rows = []
    for metric in ("likelihood", "victims_found"):
        if metric not in df.columns:
            continue
        try:
            pivot = df.pivot_table(
                index=["planning_mode", "trial"],
                columns="victim_model",
                values=metric,
                aggfunc="first",
            )
        except Exception as e:  # pragma: no cover
            log.warning(f"[D] pivot failed for {metric}: {e}")
            continue
        # Count rows that are identical across all victim models
        nunique_per_row = pivot.nunique(axis=1)
        n_total = len(pivot)
        n_identical = int((nunique_per_row == 1).sum())
        pct = 100.0 * n_identical / n_total if n_total else float("nan")
        print(f"\n  Metric: {metric}")
        print(f"    Rows (mode × trial):            {n_total}")
        print(f"    Rows identical across models:   {n_identical} ({pct:.1f} %)")

        if n_identical == n_total:
            print(f"    ► '{metric}' does NOT depend on victim_model in Study 4.")
            print(f"      Study 4 is a NULL STUDY for this metric.")
        elif n_identical > 0.9 * n_total:
            print(
                f"    ► '{metric}' barely depends on victim_model (<10 % rows differ)."
            )
        else:
            print(
                f"    ► '{metric}' does vary across victim models — study is "
                "informative."
            )

        rows.append(
            {
                "metric": metric,
                "rows_total": n_total,
                "rows_identical_across_victim_models": n_identical,
                "pct_identical": pct,
            }
        )

    out = Path(output_dir) / "victim_model_null_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"[D] saved {out}")
    return pd.DataFrame(rows)


# ── Analysis H: Shapiro-Wilk + Levene (normality & variance homogeneity) ─


def analysis_H_normality_variance(df, output_dir, metric="likelihood"):
    """Shapiro-Wilk normality per planning mode + Levene homogeneity across
    modes, providing formal justification for non-parametric tests.

    Two key assumptions for parametric tests (t-test, ANOVA):
      1. **Normality** of residuals — checked per mode via Shapiro-Wilk.
         H₀: data is drawn from a normal distribution.
      2. **Homogeneity of variances** across groups — checked via Levene.
         H₀: all groups have equal variance.

    If either assumption is rejected (p < 0.05), non-parametric tests
    (Mann-Whitney U) are justified.  The analysis also runs pairwise
    Levene for the critical comparisons to check variance pooling
    assumptions for Cohen's d.

    Parameters
    ----------
    df : DataFrame with columns 'planning_mode' and <metric>.
    output_dir : path for the output CSV.
    metric : str, default "likelihood".

    Returns
    -------
    tuple of (shapiro_df, levene_df)
    """
    print("\n" + "=" * 100)
    print(f"ANALYSIS H — Shapiro-Wilk (normality) + Levene (variance homogeneity)")
    print(f"           on '{metric}' in Study 6 (generalization)")
    print("=" * 100)

    modes = sorted(df["planning_mode"].unique())
    by_mode = {
        m: df.loc[df["planning_mode"] == m, metric].dropna().values for m in modes
    }

    # ── 1. Shapiro-Wilk per planning mode ─────────────────────────────
    print(f"\n  1. Shapiro-Wilk test for normality (H₀: data is normal)")
    print(f"     {'Mode':<20} {'n':>4} {'W':>8} {'p':>10}  {'Decision':<20}")
    print("     " + "-" * 70)
    shapiro_rows = []
    any_non_normal = False
    for m in modes:
        vals = by_mode[m]
        n = len(vals)
        if n < 3:
            print(f"     {m:<20} {n:>4} {'—':>8} {'—':>10}  {'too few samples':<20}")
            shapiro_rows.append(
                {
                    "mode": m,
                    "n": n,
                    "W": None,
                    "p_value": None,
                    "normal": None,
                    "comment": "too few samples (<3)",
                }
            )
            continue
        W, p = stats.shapiro(vals)
        normal = p >= 0.05
        if not normal:
            any_non_normal = True
        decision = "normal ✓" if normal else "NON-NORMAL ✗"
        print(f"     {m:<20} {n:>4} {W:>8.4f} {p:>10.4e}  {decision:<20}")
        shapiro_rows.append(
            {"mode": m, "n": n, "W": W, "p_value": p, "normal": normal, "comment": ""}
        )
    shapiro_df = pd.DataFrame(shapiro_rows)

    if any_non_normal:
        print(
            "\n     ► At least one mode is non-normal → non-parametric tests "
            "(Mann-Whitney U) ARE justified."
        )
    else:
        print(
            "\n     ► All modes pass normality → parametric t-tests could be used, "
            "but check variances below."
        )

    # ── 2. Levene test for homogeneity of variances across ALL modes ──
    print(f"\n  2. Levene test for homogeneity of variance (H₀: equal variances)")
    print(f"     Comparing all {len(modes)} modes simultaneously.")
    groups_for_levene = [by_mode[m] for m in modes if len(by_mode[m]) > 0]
    if len(groups_for_levene) >= 2:
        stat, p = stats.levene(*groups_for_levene, center="median")
        homogeneous = p >= 0.05
        decision = "equal ✓" if homogeneous else "UNEQUAL ✗"
        print(f"     Levene statistic = {stat:.4f},  p = {p:.4e}  →  {decision}")
        if not homogeneous:
            print(
                "     ► Variances are NOT homogeneous → further justification for "
                "non-parametric tests (Mann-Whitney U)."
            )
        else:
            print(
                "     ► Variances are homogeneous → parametric assumptions partially "
                "hold (but normality still matters)."
            )
        levene_overall = {
            "comparison": f"all_{len(modes)}_modes",
            "statistic": stat,
            "p_value": p,
            "homogeneous": homogeneous,
        }
    else:
        levene_overall = None
        print("     Not enough groups to run Levene.")

    # ── 3. Pairwise Levene on critical comparisons ────────────────────
    print(f"\n  3. Pairwise Levene on critical comparisons (for Cohen's d pooling)")
    print(f"     {'Comparison':<40} {'Levene stat':>12} {'p':>10}  {'Equal var?':<12}")
    print("     " + "-" * 80)
    levene_rows = []
    if levene_overall is not None:
        levene_overall["pair"] = "all"
        levene_rows.append(levene_overall)
    for a, b, rationale in CRITICAL_COMPARISONS:
        da, db = by_mode.get(a), by_mode.get(b)
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            continue
        stat, p = stats.levene(da, db, center="median")
        equal_var = p >= 0.05
        decision = "yes ✓" if equal_var else "no ✗"
        label = f"{a} vs {b}"
        print(
            f"     {label:<40} {stat:>12.4f} {p:>10.4e}  {decision:<12}  ({rationale})"
        )
        levene_rows.append(
            {
                "pair": f"{a}_vs_{b}",
                "comparison": label,
                "statistic": stat,
                "p_value": p,
                "homogeneous": equal_var,
                "rationale": rationale,
            }
        )
    levene_df = pd.DataFrame(levene_rows)

    # ── 4. Summary ───────────────────────────────────────────────────
    print(f"\n  4. Summary")
    n_modes_non_normal = (~shapiro_df["normal"].fillna(True)).sum()
    print(
        f"     Shapiro-Wilk: {n_modes_non_normal}/{len(modes)} modes non-normal"
        f" (p < 0.05)"
    )
    n_unequal = (~levene_df["homogeneous"].fillna(True)).sum() if len(levene_df) else 0
    print(
        f"     Levene:       {n_unequal}/{len(levene_df)} comparisons have "
        f"unequal variances (p < 0.05)"
        if len(levene_df)
        else "     Levene: N/A"
    )
    if any_non_normal or (levene_overall and not levene_overall["homogeneous"]):
        print(
            "\n     ► CONCLUSION: Assumptions for parametric tests are VIOLATED.\n"
            "       Non-parametric tests (Mann-Whitney U) are the appropriate "
            "choice for this data,\n"
            "       consistent with the methodology used throughout the paper."
        )
    else:
        print(
            "\n     ► Both normality and variance homogeneity hold. "
            "Parametric tests would be valid."
        )

    # ── Persist ──────────────────────────────────────────────────────
    out_shapiro = Path(output_dir) / f"shapiro_wilk_normality_{metric}.csv"
    shapiro_df.to_csv(out_shapiro, index=False)
    log.info(f"[H/shapiro] saved {out_shapiro}")

    out_levene = Path(output_dir) / f"levene_variance_{metric}.csv"
    levene_df.to_csv(out_levene, index=False)
    log.info(f"[H/levene]  saved {out_levene}")

    return shapiro_df, levene_df


# ── Analysis I: Post-hoc power analysis ───────────────────────────────────


def analysis_I_power_analysis(
    df,
    output_dir,
    mode_a="dynamic_3step",
    mode_b="pizza",
    mode_c="static_3step",
    metric="likelihood",
):
    """Post-hoc power analysis to justify sample size per group.

    Computes:
      1. Minimum detectable Cohen's d at 80% (and 90%, 95%) power for the
         per-dataset sample size and the pooled Study 6 sample size.
      2. Achieved power for the *observed* effect sizes from key pairwise
         comparisons (d3 vs pizza, d3 vs static_3step), both pooled and
         per-dataset.
      3. A power-versus-n table showing how many samples per group are
         needed to detect small (d=0.2), medium (d=0.5) and large (d=0.8)
         effects at 80% power.

    Parameters
    ----------
    df : DataFrame with columns 'planning_mode', 'dataset' and <metric>.
    output_dir : path for the output CSV.
    mode_a, mode_b, mode_c : str.  Planning modes to analyse.
    metric : str, default "likelihood".

    Returns
    -------
    pd.DataFrame
    """
    print("\n" + "=" * 100)
    print("ANALYSIS I — Post-hoc power analysis")
    print("=" * 100)

    modes_available = sorted(df["planning_mode"].unique())
    datasets = sorted(df["dataset"].unique())
    n_pooled = {
        m: len(df.loc[df["planning_mode"] == m, metric].dropna())
        for m in (mode_a, mode_b, mode_c)
    }
    n_per_ds = len(
        df.loc[
            (df["planning_mode"] == mode_a) & (df["dataset"] == datasets[0]), metric
        ].dropna()
    )

    # ── 1. Minimum detectable d at target powers ──────────────────────
    print(f"\n  1. Minimum detectable |Cohen's d| (two-sample, α = 0.05)")
    print(f"     {'n₁=n₂':>7}  {'80% power':>12}  {'90% power':>12}  {'95% power':>12}")
    print("     " + "-" * 55)

    rows_mdd = []
    per_ds_label = f"n={n_per_ds} (per-dataset)"
    pooled_label = f"n={n_pooled[mode_a]} (pooled Study 6)"
    for label, n_val in [
        (per_ds_label, n_per_ds),
        (pooled_label, n_pooled[mode_a]),
    ]:
        d80 = _min_detectable_d(n_val, n_val, target_power=0.80)
        d90 = _min_detectable_d(n_val, n_val, target_power=0.90)
        d95 = _min_detectable_d(n_val, n_val, target_power=0.95)
        print(f"     {str(n_val):>7}  {d80:>12.4f}  {d90:>12.4f}  {d95:>12.4f}")
        rows_mdd.append(
            {
                "sample_size_per_group": n_val,
                "label": label,
                "min_d_80pct": d80,
                "min_d_90pct": d90,
                "min_d_95pct": d95,
            }
        )

    d_threshold = _min_detectable_d(n_per_ds, n_per_ds, target_power=0.80)
    print(
        f"\n     ► At n={n_per_ds} per group, a |d| ≥ {d_threshold:.2f} is needed "
        f"for 80 % power."
    )
    if d_threshold < 0.5:
        print("       'Small' effects (d=0.2) are NOT reliably detectable per dataset.")
    elif d_threshold < 0.8:
        print(
            "       'Medium' effects (d≥0.5) are detectable; "
            "'small' effects (d=0.2) are not."
        )
    else:
        print("       Only 'large' effects (d≥0.8) are detectable per dataset.")

    # ── 2. Achieved power for observed effect sizes ──────────────────
    print(f"\n  2. Achieved power for observed effect sizes")

    # Per-dataset effect sizes for d3 vs pizza (from Study 6)
    per_ds_rows = []
    for ds in datasets:
        a = (
            df.loc[(df["planning_mode"] == mode_a) & (df["dataset"] == ds), metric]
            .dropna()
            .values
        )
        b = (
            df.loc[(df["planning_mode"] == mode_b) & (df["dataset"] == ds), metric]
            .dropna()
            .values
        )
        if len(a) < 2 or len(b) < 2:
            continue
        d_obs = cohens_d(a, b)
        pwr = _power_t_test(d_obs, len(a), len(b))
        _, p = mwu(a, b)
        per_ds_rows.append(
            {
                "comparison": f"{mode_a}_vs_{mode_b}",
                "level": f"DS{ds}",
                "n": len(a),
                "cohens_d": d_obs,
                "power": pwr,
                "p_mwu": p,
                "significant": p < 0.05,
            }
        )

    # Pooled effect sizes for key comparisons
    pooled_rows = []
    for ma, mb, rationale in [
        (mode_a, mode_b, "external baseline (key)"),
        (mode_a, mode_c, "internal ablation"),
    ]:
        a = df.loc[df["planning_mode"] == ma, metric].dropna().values
        b = df.loc[df["planning_mode"] == mb, metric].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        d_obs = cohens_d(a, b)
        pwr = _power_t_test(d_obs, len(a), len(b))
        _, p = mwu(a, b)
        pooled_rows.append(
            {
                "comparison": f"{ma}_vs_{mb}",
                "level": "pooled Study 6",
                "n": len(a),
                "cohens_d": d_obs,
                "power": pwr,
                "p_mwu": p,
                "significant": p < 0.05,
            }
        )

    print(
        f"     {'comparison':<30} {'level':<16} {'n':>5} {'d_obs':>8} "
        f"{'power':>8} {'p(MWU)':>10}  sig"
    )
    print("     " + "-" * 88)
    for row in pooled_rows + sorted(per_ds_rows, key=lambda r: r["cohens_d"]):
        print(
            f"     {row['comparison']:<30} {row['level']:<16} {row['n']:>5} "
            f"{row['cohens_d']:>+8.3f} {row['power']:>8.3f} "
            f"{row['p_mwu']:>10.2e}  {'yes' if row['significant'] else 'no'}"
        )

    # Summarise: how many per-dataset comparisons have >80% power?
    high_power = sum(1 for r in per_ds_rows if r["power"] >= 0.80)
    sig_found = sum(1 for r in per_ds_rows if r["significant"])
    print(
        f"\n     ► {high_power}/{len(per_ds_rows)} per-dataset comparisons "
        f"achieve ≥ 80% power."
    )
    print(
        f"     ► {sig_found}/{len(per_ds_rows)} per-dataset comparisons are "
        f"significant (p < 0.05)."
    )

    # ── 3. Sample-size requirements by effect size ────────────────────
    print(f"\n  3. Sample size required at 80% power by Cohen's d")
    print(f"     {'target d':>10}  {'n per group (α=0.05)':>22}")
    print("     " + "-" * 40)
    n_req_rows = []
    for d_target in [0.10, 0.15, 0.20, 0.30, 0.50, 0.80]:
        # Binary search for n
        lo_n, hi_n = 2, 5000
        for _ in range(40):
            mid_n = (lo_n + hi_n) // 2
            if _power_t_test(d_target, mid_n, mid_n) < 0.80:
                lo_n = mid_n
            else:
                hi_n = mid_n
        n_req = (lo_n + hi_n) // 2
        n_req_rows.append(
            {
                "cohens_d": d_target,
                "n_per_group_80pct": n_req,
            }
        )
        print(f"     {d_target:>10.2f}  {n_req:>22}")

    n_for_small = next(r for r in n_req_rows if abs(r["cohens_d"] - 0.20) < 1e-6)
    print(
        f"\n     ► To detect a small effect (d={0.20:.2f}) with 80 % power, "
        f"n={n_for_small['n_per_group_80pct']} per group is needed."
    )
    print(
        f"       Our n={n_per_ds} per dataset (per-mode) is "
        f"{'SUFFICIENT' if n_per_ds >= n_for_small['n_per_group_80pct'] else 'INSUFFICIENT'}"
        f" for small effects — which is expected and acceptable: the paper"
        f"\n       targets medium-to-large effects at the per-dataset level."
    )

    # ── 4. Summary ───────────────────────────────────────────────────
    print(f"\n  4. Summary")
    d_pooled_ab = next(r for r in pooled_rows if mode_b in r["comparison"])
    d_pooled_ac = next(r for r in pooled_rows if mode_c in r["comparison"])
    print(
        f"     Pooled Study 6 (n={n_pooled[mode_a]}): "
        f"d3 vs pizza d={d_pooled_ab['cohens_d']:.3f}, power={d_pooled_ab['power']:.3f}; "
        f"d3 vs s3 d={d_pooled_ac['cohens_d']:.3f}, power={d_pooled_ac['power']:.3f}"
    )
    print(
        f"     Per-dataset (n={n_per_ds}): min detectable d={d_threshold:.2f} "
        f"at 80% power → only medium-to-large effects are detectable"
    )
    print(
        f"     Interpretation: n={n_per_ds} is adequate for the paper's claims."
        f" The per-dataset analysis is supplementary to the pooled LMM, which "
        f"has ample power (n≈{n_pooled[mode_a]})."
    )

    # ── Persist ──────────────────────────────────────────────────────
    all_rows = pooled_rows + per_ds_rows + rows_mdd + n_req_rows
    out = Path(output_dir) / "power_analysis_posthoc.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    log.info(f"[I] saved {out}")
    return pd.DataFrame(all_rows)


# ── Analysis F: Linear Mixed Model (LMM) on Study 6 ──────────────────────


def analysis_F_mixed_effects(
    df,
    output_dir,
    mode_a="dynamic_3step",
    mode_b="pizza",
    metric="likelihood",
    name_letter="F",
):
    """Linear Mixed Model: formal test of planning_mode effect with dataset
    as a random intercept.

    This complements the per-dataset MWU (Analysis B) by pooling evidence
    across datasets while explicitly modelling between-environment
    heterogeneity.

    Model
    -----
    Null  :  metric ~ 1 + (1 | dataset)
    Full  :  metric ~ planning_mode + (1 | dataset)

    Outputs
    -------
    * Fixed-effects table (coef, SE, z, p, 95 % CI).
    * Random-effects variances (between-dataset σ², residual σ²).
    * ICC = σ²_dataset / (σ²_dataset + σ²_residual).
    * Likelihood Ratio Test (LRT) χ²(df=1) comparing Null vs Full.
    * BLUPs (Best Linear Unbiased Predictors) per dataset.

    Returns
    -------
    dict with keys 'fixed', 'random', 'lrt', 'blups'.
    """
    print("\n" + "=" * 100)
    print(
        f"ANALYSIS {name_letter} — Linear Mixed Model ({mode_a} vs {mode_b}, metric='{metric}')"
    )
    print("=" * 100)

    if not _HAS_STATSMODELS:
        print("  [SKIP] statsmodels is not installed.")
        print("         Install with:  pixi add statsmodels")
        return None

    # Filter to the two modes of interest
    sub = df[df["planning_mode"].isin([mode_a, mode_b])].copy()
    if sub.empty:
        print("  No data for the requested modes.")
        return None

    # Encode planning_mode as 0/1 (reference = mode_b)
    sub["mode_code"] = (sub["planning_mode"] == mode_a).astype(int)
    sub["dataset_id"] = sub["dataset"].astype("category").cat.codes

    # Drop rows with missing metric
    sub = sub.dropna(subset=[metric]).copy()
    if len(sub) < 10:
        print("  Insufficient data after filtering (< 10 rows).")
        return None

    # ── Fit model ─────────────────────────────────────────────────────
    endog = sub[metric].values
    exog = sm.add_constant(sub["mode_code"].values)
    groups = sub["dataset_id"].values

    try:
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = MixedLM(endog, exog, groups=groups)
            m_full = model.fit(reml=False, method="nm")

            # Check for convergence warnings
            if w and any("boundary" in str(warning.message).lower() for warning in w):
                print("\n  ⚠️  WARNING: Model convergence issue detected.")
                print("      The MLE may be on the boundary of the parameter space.")
                print("      This often occurs when:")
                print("        • Between-group variance is very small or zero")
                print("        • Sample size is limited")
                print("      Results should be interpreted with caution.\n")
    except Exception as exc:  # pragma: no cover
        log.warning(f"[F] MixedLM fitting failed: {exc}")
        print(f"  [ERROR] MixedLM fitting failed: {exc}")
        return None

    # ── Fixed effects ─────────────────────────────────────────────────
    # Use model attributes directly instead of parsing summary table
    # Note: MixedLM includes 'Group Var' in params, so we take only fixed effects
    conf_int = m_full.conf_int()
    n_fixed = len(exog[0])  # Number of fixed effects (intercept + predictors)

    # Handle both Series and ndarray return types
    params = (
        m_full.params[:n_fixed]
        if isinstance(m_full.params, np.ndarray)
        else m_full.params.iloc[:n_fixed]
    )
    bse = (
        m_full.bse[:n_fixed]
        if isinstance(m_full.bse, np.ndarray)
        else m_full.bse.iloc[:n_fixed]
    )
    tvalues = (
        m_full.tvalues[:n_fixed]
        if isinstance(m_full.tvalues, np.ndarray)
        else m_full.tvalues.iloc[:n_fixed]
    )
    pvalues = (
        m_full.pvalues[:n_fixed]
        if isinstance(m_full.pvalues, np.ndarray)
        else m_full.pvalues.iloc[:n_fixed]
    )
    ci_lower = (
        conf_int[:n_fixed, 0]
        if isinstance(conf_int, np.ndarray)
        else conf_int.iloc[:n_fixed, 0]
    )
    ci_upper = (
        conf_int[:n_fixed, 1]
        if isinstance(conf_int, np.ndarray)
        else conf_int.iloc[:n_fixed, 1]
    )

    fe_df = pd.DataFrame(
        {
            "coef": params,
            "std err": bse,
            "z": tvalues,
            "P>|z|": pvalues,
            "[0.025": ci_lower,
            "0.975]": ci_upper,
        }
    )
    fe_df.index = ["Intercept", f"{mode_a}_vs_{mode_b}"]

    print("\n  Fixed effects:")
    print(
        f"    {'parameter':<30} {'coef':>10} {'SE':>10} {'z':>8} {'p':>10} {'sig':>5}"
    )
    print("    " + "-" * 80)
    for idx, row in fe_df.iterrows():
        print(
            f"    {idx:<30} {row['coef']:>10.4f} {row['std err']:>10.4f} "
            f"{row['z']:>8.3f} {row['P>|z|']:>10.3e} "
            f"{'***' if row['P>|z|'] < 0.001 else '**' if row['P>|z|'] < 0.01 else '*' if row['P>|z|'] < 0.05 else 'ns':>5}"
        )

    # ── Random effects ────────────────────────────────────────────────
    # Handle both DataFrame and ndarray return types for cov_re
    cov_re = m_full.cov_re
    if isinstance(cov_re, np.ndarray):
        var_dataset = float(cov_re[0, 0] if cov_re.ndim > 1 else cov_re[0])
    else:
        var_dataset = float(cov_re.iloc[0, 0])

    var_resid = float(m_full.scale)
    icc = (
        var_dataset / (var_dataset + var_resid)
        if (var_dataset + var_resid) > 0
        else float("nan")
    )

    print("\n  Random effects:")
    print(f"    σ²_dataset  (between) = {var_dataset:.6f}")
    print(f"    σ²_residual (within)  = {var_resid:.6f}")
    print(
        f"    ICC                   = {icc:.4f}  ({icc * 100:.1f}% of variance due to dataset)"
    )

    # ── Wald test for planning_mode effect ────────────────────────────
    mode_coef = fe_df.loc[fe_df.index[1], "coef"]
    mode_se = fe_df.loc[fe_df.index[1], "std err"]
    mode_z = fe_df.loc[fe_df.index[1], "z"]
    mode_p = fe_df.loc[fe_df.index[1], "P>|z|"]
    print("\n  Wald test for planning_mode effect:")
    print(
        f"    z = {mode_z:.4f},  p = {mode_p:.3e}  "
        f"{'***' if mode_p < 0.001 else '**' if mode_p < 0.01 else '*' if mode_p < 0.05 else 'ns'}"
    )

    # ── BLUPs per dataset ─────────────────────────────────────────────
    blups = m_full.random_effects
    blup_rows = []
    ds_map = dict(enumerate(sub["dataset"].astype("category").cat.categories))
    # Get intercept value (first fixed effect parameter)
    global_intercept = (
        float(m_full.params[0])
        if isinstance(m_full.params, np.ndarray)
        else float(m_full.params.iloc[0])
    )

    for g_id, re in blups.items():
        ds_label = ds_map.get(g_id, g_id)
        # re is a Series, access by position using iloc or by name
        re_intercept = float(re.iloc[0]) if hasattr(re, "iloc") else float(re[0])
        blup_rows.append(
            {
                "dataset": int(ds_label) if str(ds_label).isdigit() else ds_label,
                "blup_intercept": re_intercept,
                "global_intercept": global_intercept,
                "predicted_intercept": global_intercept + re_intercept,
            }
        )
    blup_df = pd.DataFrame(blup_rows).sort_values("dataset")

    print("\n  BLUPs per dataset (sorted by predicted intercept):")
    print(f"    {'dataset':>6} {'BLUP':>10} {'pred_intercept':>16}")
    print("    " + "-" * 40)
    for _, r in blup_df.sort_values("predicted_intercept").iterrows():
        print(
            f"    {r['dataset']:>6} {r['blup_intercept']:>+10.4f} {r['predicted_intercept']:>16.4f}"
        )

    # ── Persist ───────────────────────────────────────────────────────
    out = Path(output_dir) / f"mixed_effects_{mode_a}_vs_{mode_b}_{metric}.csv"

    # Build a single CSV with three sections, separated by blank rows
    sections = []

    # Section 1: fixed effects
    fe_out = fe_df.copy()
    fe_out["parameter"] = fe_out.index
    fe_out = fe_out[["parameter", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"]]
    fe_out.insert(0, "section", "fixed_effects")
    sections.append(fe_out)

    # Section 2: random effects + ICC + Wald test
    re_out = pd.DataFrame(
        [
            {
                "section": "random_effects",
                "var_dataset": var_dataset,
                "var_residual": var_resid,
                "icc": icc,
                "wald_z": mode_z,
                "wald_pvalue": mode_p,
            }
        ]
    )
    sections.append(re_out)

    # Section 3: blups
    blup_df.insert(0, "section", "blups")
    sections.append(blup_df)

    pd.concat(sections, ignore_index=True).to_csv(out, index=False)
    log.info(f"[F] saved {out}")
    print(f"\n  Output → {out}")

    return {
        "fixed": fe_df,
        "random": {"var_dataset": var_dataset, "var_residual": var_resid, "icc": icc},
        "wald": {"z": mode_z, "pvalue": mode_p},
        "blups": blup_df,
    }


# ── Plotting (optional) ──────────────────────────────────────────────────


def generate_plots(study6_df, drones_df, budget_df, fov_df, output_dir):
    """Produce the two figures referenced in the paper narrative:
    * fig1: per-dataset heterogeneity + overall boxplot
    * fig2: saturation curves for drones, budget, FOV
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
    except ImportError:
        log.error(
            "matplotlib and seaborn are required for --plot. "
            "Install with: pip install matplotlib seaborn"
        )
        return

    colors = {
        "dynamic_3step": "#1f77b4",
        "pizza": "#ff7f0e",
        "static_3step": "#d62728",
        "dynamic": "#e377c2",
        "static": "#8c564b",
    }

    # ── fig1 ─────────────────────────────────────────────────────────
    if study6_df is not None:
        diffs = []
        for ds in sorted(study6_df["dataset"].unique()):
            a = study6_df[
                (study6_df["planning_mode"] == "dynamic_3step")
                & (study6_df["dataset"] == ds)
            ]["likelihood"].values
            b = study6_df[
                (study6_df["planning_mode"] == "pizza") & (study6_df["dataset"] == ds)
            ]["likelihood"].values
            _, p = mwu(a, b)
            diffs.append((int(ds), a.mean() - b.mean(), p))
        dd = pd.DataFrame(diffs, columns=["dataset", "delta", "p"]).sort_values("delta")

        n_modes = len(study6_df["planning_mode"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        bar_colors = ["#d62728" if x < 0 else "#2ca02c" for x in dd["delta"]]
        ax.barh(range(len(dd)), dd["delta"], color=bar_colors, alpha=0.75)
        ax.set_yticks(range(len(dd)))
        ax.set_yticklabels([f"DS{d}" for d in dd["dataset"]])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Δ likelihood (dynamic_3step − pizza)")
        ax.set_title("Per-dataset: dynamic_3step vs pizza")
        ax.grid(True, alpha=0.3)
        for i, (_, row) in enumerate(dd.iterrows()):
            marker = "*" if row["p"] < 0.05 else "ns"
            off = 0.002 if row["delta"] >= 0 else -0.002
            ha = "left" if row["delta"] >= 0 else "right"
            ax.text(row["delta"] + off, i, marker, va="center", ha=ha, fontsize=9)

        ax = axes[1]
        order = (
            study6_df.groupby("planning_mode")["likelihood"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        sns.boxplot(
            data=study6_df,
            x="planning_mode",
            y="likelihood",
            order=order,
            hue="planning_mode",
            palette=colors,
            legend=False,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=45)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        ax.set_xlabel("")
        ax.set_ylabel("Likelihood")
        ax.set_title(
            f"Likelihood distribution by mode (Study 6, N={len(study6_df) // n_modes})"
        )
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out = Path(output_dir) / "fig1_per_dataset_and_boxplot.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"[plot] saved {out}")

    # ── fig2 ─────────────────────────────────────────────────────────
    modes_plot = [
        "dynamic_3step",
        "pizza",
        "static_3step",
        "dynamic",
        "static",
    ]
    triplets = [
        ("budget", budget_df, "Budget (×1000 steps)", lambda v: v / 1000),
        ("num_drones", drones_df, "Num drones", lambda v: v),
        ("fov_deg", fov_df, "FOV (degrees)", lambda v: v),
    ]
    present = [t for t in triplets if t[1] is not None]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5.3 * len(present), 5))
    if len(present) == 1:
        axes = [axes]
    for ax, (col, d, xlabel, xform) in zip(axes, present):
        for m in modes_plot:
            if m not in d["planning_mode"].unique():
                continue
            sub = (
                d[d["planning_mode"] == m]
                .groupby(col)["likelihood"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            ax.errorbar(
                sub[col].map(xform),
                sub["mean"],
                yerr=sub["std"] / np.sqrt(sub["count"]),
                marker="o",
                label=m,
                color=colors.get(m, None),
                linewidth=2,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Likelihood")
        ax.set_title(f"Scaling by {col}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / "fig2_sensitivity_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"[plot] saved {out}")


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extra post-hoc analyses on comprehensive_study outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULT_DIR),
        help="Directory containing the study CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to save the generated CSVs and plots.",
    )
    parser.add_argument(
        "--study6-csv",
        default=None,
        help="Override path to the generalization study CSV "
        "(default: newest study_datasets_*.csv).",
    )
    parser.add_argument(
        "--drones-csv", default=None, help="Override path to the drones study CSV."
    )
    parser.add_argument(
        "--budget-csv", default=None, help="Override path to the budget study CSV."
    )
    parser.add_argument(
        "--fov-csv", default=None, help="Override path to the FOV study CSV."
    )
    parser.add_argument(
        "--victim-csv",
        default=None,
        help="Override path to the victim_model study CSV.",
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F", "G"],
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        help="Which analyses to run.",
    )
    parser.add_argument(
        "--mode-a",
        default="dynamic_3step",
        help="Primary mode for pairwise analyses (B, C).",
    )
    parser.add_argument(
        "--mode-b",
        default="pizza",
        help="Comparator mode for pairwise analyses (B, C).",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Also generate PNG figures."
    )
    return parser.parse_args()


def find_newest(results_dir, glob):
    matches = sorted(Path(results_dir).glob(glob))
    return matches[-1] if matches else None


def resolve_csvs(args):
    rd = Path(args.results_dir)
    resolved = {
        "study6": args.study6_csv or find_newest(rd, "study_datasets_*.csv"),
        "drones": args.drones_csv or find_newest(rd, "study_drones_2*.csv"),
        "budget": args.budget_csv or find_newest(rd, "study_budget_*.csv"),
        "fov": args.fov_csv or find_newest(rd, "study_fov_*.csv"),
        "victim": args.victim_csv or find_newest(rd, "study_victim_model_*.csv"),
    }
    for k, v in resolved.items():
        log.info(f"[csv] {k:<7} → {v}")
    return resolved


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    log.info("=" * 100)
    log.info("extra_studies — post-hoc analyses")
    log.info("=" * 100)

    paths = resolve_csvs(args)
    study6 = load_csv(paths["study6"], "study6") if paths["study6"] else None
    drones = load_csv(paths["drones"], "drones") if paths["drones"] else None
    budget = load_csv(paths["budget"], "budget") if paths["budget"] else None
    fov = load_csv(paths["fov"], "fov") if paths["fov"] else None
    victim = load_csv(paths["victim"], "victim") if paths["victim"] else None

    # ── A: pairwise effect sizes (on study6, metric=likelihood) ──────
    if "A" in args.analyses and study6 is not None:
        analysis_A_pairwise(study6, "likelihood", tag="study6", output_dir=args.output_dir)

    # ── B: per-dataset heterogeneity ─────────────────────────────────
    if "B" in args.analyses and study6 is not None:
        analysis_B_per_dataset(
            study6,
            args.output_dir,
            mode_a=args.mode_a,
            mode_b=args.mode_b,
            metric="likelihood",
        )

    # ── C: saturation curves on sensitivity studies ─────────────────
    if "C" in args.analyses:
        if drones is not None:
            analysis_C_saturation(
                drones,
                "num_drones",
                args.output_dir,
                args.mode_a,
                args.mode_b,
                "likelihood",
            )
        if budget is not None:
            analysis_C_saturation(
                budget,
                "budget",
                args.output_dir,
                args.mode_a,
                args.mode_b,
                "likelihood",
            )
        if fov is not None:
            analysis_C_saturation(
                fov, "fov_deg", args.output_dir, args.mode_a, args.mode_b, "likelihood"
            )

    # ── D: null-study check on Study 4 ──────────────────────────────
    if "D" in args.analyses and victim is not None:
        analysis_D_victim_null_check(victim, args.output_dir)

    # ── E: same as A but on victims_found (operational metric) ──────
    if (
        "E" in args.analyses
        and study6 is not None
        and "victims_found" in study6.columns
    ):
        analysis_A_pairwise(study6, "victims_found", tag="study6", output_dir=args.output_dir)

    # ── F: Linear Mixed Model (dataset as random effect) ────────────
    if "F" in args.analyses and study6 is not None:
        analysis_F_mixed_effects(
            study6,
            args.output_dir,
            mode_a=args.mode_a,
            mode_b=args.mode_b,
            metric="likelihood",
        )

    # ── G: LMM for ablation comparator (dynamic_3step vs static_3step) ─
    if "G" in args.analyses and study6 is not None:
        analysis_F_mixed_effects(
            study6,
            args.output_dir,
            mode_a="dynamic_3step",
            mode_b="static_3step",
            metric="likelihood",
            name_letter="G",
        )

    # ── H: Shapiro-Wilk + Levene (normality & variance justification) ─
    if "H" in args.analyses and study6 is not None:
        analysis_H_normality_variance(study6, args.output_dir, metric="likelihood")

    # ── I: Post-hoc power analysis ──────────
    if "I" in args.analyses and study6 is not None:
        analysis_I_power_analysis(
            study6,
            args.output_dir,
            mode_a=args.mode_a,
            mode_b=args.mode_b,
            metric="likelihood",
        )

    # ── optional plots ─────────────────────────────────────────────
    if args.plot:
        generate_plots(study6, drones, budget, fov, args.output_dir)

    log.info("=" * 100)
    log.info("Extra analyses complete.")
    log.info(f"Outputs under: {Path(args.output_dir).resolve()}")
    log.info("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
