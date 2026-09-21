#!/usr/bin/env python3
# scripts/analyze_study_results.py
"""
Statistical analysis of comprehensive study results.

Computes:
- Descriptive statistics (mean, std, median, IQR)
- Statistical tests (Mann-Whitney U, Kruskal-Wallis)
- Effect sizes (Cohen's d, percentage improvements)
- Confidence intervals (95% CI)
- Bonferroni correction for multiple comparisons

Usage:
  python scripts/analyze_study_results.py results/comprehensive_study/study_main_*.csv
  python scripts/analyze_study_results.py results/comprehensive_study/all_studies_*.csv --groupby study
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import paired_stats as ps

import sarenv

log = sarenv.get_logger()
# ---------------------------------------------------------------------------
# Study family discovery helpers
# ---------------------------------------------------------------------------

STUDY_FAMILIES = {
    "main": "study_main_*.csv",
    "datasets": "study_datasets_*.csv",
    "drones": "study_drones_*.csv",
    "budget": "study_budget_*.csv",
    "fov": "study_fov_*.csv",
    "victim_model": "study_victim_model_*.csv",
}

DEFAULT_RESULTS_DIR = "results/comprehensive_study"


def _sorted_matches(results_dir: Path, pattern: str) -> list:
    """Return sorted list of Paths matching *pattern* inside *results_dir*."""
    return sorted(results_dir.glob(pattern))


def resolve_csvs(results_dir: Path, families: list, versions: str) -> list:
    """Return list of (family_name, Path) tuples to analyse.

    Parameters
    ----------
    results_dir : Path
        Directory that contains the study CSVs.
    families : list of str
        Family names to include (keys of STUDY_FAMILIES).
    versions : str
        ``"newest"`` → one CSV per family (latest timestamp);
        ``"all"``    → every CSV per family.
    """
    result = []
    for fam in families:
        pattern = STUDY_FAMILIES[fam]
        matches = _sorted_matches(results_dir, pattern)
        # Exclude files with OBSOLETE in the name
        matches = [m for m in matches if "OBSOLETE" not in m.name]
        if not matches:
            log.warning(f"No CSV found for family '{fam}' in {results_dir}")
            continue
        if versions == "newest":
            result.append((fam, matches[-1]))
        else:
            result.extend((fam, m) for m in matches)
    return result


def cohens_d(x, y):
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )


def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence interval."""
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrapped_means, alpha * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha) * 100)
    return lower, upper


def analyze_single_study(
    df,
    study_name="main",
    metric="likelihood",
    log=None,
    bootstrap_ci=None,
    cohens_d=None,
):
    """Paired analysis for a single study. `log`, `bootstrap_ci`, `cohens_d`
    are the module-level objects in this script; passed in here so this file is
    importable standalone. In the real script call as analyze_single_study(df,
    ...) — the defaults below resolve to the script's globals."""
    if log is None:
        import logging as _logging

        log = _logging.getLogger("paired")
    if cohens_d is None:
        cohens_d = ps.cohens_d
    if bootstrap_ci is None:

        def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
            boot = [
                np.mean(np.random.choice(data, size=len(data), replace=True))
                for _ in range(n_bootstrap)
            ]
            a = (1 - confidence) / 2
            return np.percentile(boot, a * 100), np.percentile(boot, (1 - a) * 100)

    log.info(f"\n{'=' * 100}")
    log.info(f"STUDY: {study_name.upper()}")
    log.info(f"METRIC: {metric}")
    log.info(f"{'=' * 100}\n")

    modes = sorted(df["planning_mode"].unique())
    log.info(f"Planning modes tested: {len(modes)}")
    log.info(f"Modes: {', '.join(modes)}\n")

    # ---- Descriptive statistics (unchanged) -----------------------------
    log.info("-" * 100)
    log.info("DESCRIPTIVE STATISTICS")
    log.info("-" * 100)
    stats_data = []
    for mode in modes:
        d = df[df["planning_mode"] == mode][metric].values
        q25, q75 = np.percentile(d, [25, 75])
        lo, hi = bootstrap_ci(d)
        stats_data.append(
            {
                "mode": mode,
                "n": len(d),
                "mean": np.mean(d),
                "std": np.std(d, ddof=1),
                "median": np.median(d),
                "Q25": q25,
                "Q75": q75,
                "IQR": q75 - q25,
                "CI_lower": lo,
                "CI_upper": hi,
            }
        )
    stats_df = pd.DataFrame(stats_data).sort_values("mean", ascending=False)
    print(
        f"\n{'Mode':<18s} {'n':>4s}  {'Mean':>10s}  {'Std':>10s}  "
        f"{'Median':>10s}  {'IQR':>10s}  {'95% CI':>22s}"
    )
    print("-" * 100)
    for _, r in stats_df.iterrows():
        print(
            f"{r['mode']:<18s} {int(r['n']):>4d}  {r['mean']:>10.4f}  "
            f"{r['std']:>10.4f}  {r['median']:>10.4f}  {r['IQR']:>10.4f}  "
            f"[{r['CI_lower']:>9.4f}, {r['CI_upper']:>9.4f}]"
        )

    # ---- Build the paired (id x mode) matrix ----------------------------
    paired = True
    wide = None
    try:
        wide = ps.paired_matrix(df, metric, modes=modes)
    except ValueError as e:
        paired = False
        log.warning(
            f"[!] PAIRING UNAVAILABLE — falling back to UNPAIRED tests. Reason: {e}"
        )

    def _vectors(mode_a, mode_b):
        """Aligned complete-case vectors for a pair (paired) or raw arrays."""
        if paired:
            sub = wide[[mode_a, mode_b]].dropna()
            return sub[mode_a].values, sub[mode_b].values
        a = df[df["planning_mode"] == mode_a][metric].values
        b = df[df["planning_mode"] == mode_b][metric].values
        return a, b

    # ---- Omnibus: Friedman (paired) or Kruskal-Wallis (fallback) --------
    log.info(f"\n{'-' * 100}")
    if paired:
        log.info("FRIEDMAN TEST (paired omnibus)")
        log.info("-" * 100)
        chi2, p_om, n_blocks, used = ps.friedman_omnibus(wide, modes=modes)
        if np.isnan(chi2):
            log.info(
                f"Friedman skipped (needs >=3 modes and >=2 complete "
                f"blocks; have {len(used)} modes, {n_blocks} blocks)."
            )
        else:
            log.info(
                f"chi^2({len(used) - 1}) = {chi2:.4f}   "
                f"p = {ps.fmt_p(p_om)}   (complete blocks: {n_blocks})"
            )
    else:
        from scipy import stats as _st

        log.info("KRUSKAL-WALLIS H-TEST (omnibus, UNPAIRED fallback)")
        log.info("-" * 100)
        groups = [df[df["planning_mode"] == m][metric].values for m in modes]
        h, p_om = _st.kruskal(*groups)
        log.info(f"H = {h:.4f}   p = {ps.fmt_p(p_om)}")

    # ---- Pairwise vs baseline (worst mode): paired Wilcoxon -------------
    log.info(f"\n{'-' * 100}")
    log.info(
        "PAIRWISE WILCOXON SIGNED-RANK vs baseline"
        + ("" if paired else "  (UNPAIRED fallback)")
    )
    log.info("-" * 100)
    baseline_mode = stats_df.iloc[-1]["mode"]
    baseline_mean = stats_df.iloc[-1]["mean"]
    log.info(f"Baseline: {baseline_mode} (mean={baseline_mean:.4f})\n")

    raw_p, recs = [], []
    for _, row in stats_df.iterrows():
        mode = row["mode"]
        if mode == baseline_mode:
            continue
        x, y = _vectors(mode, baseline_mode)
        if paired:
            w_stat, p_val, r_rb, n_pair = ps.paired_wilcoxon(x, y)
        else:
            from scipy import stats as _st

            w_stat, p_val = _st.mannwhitneyu(x, y, alternative="two-sided")
            p_val = ps.cap_p(p_val)
            r_rb, n_pair = (
                ps.wilcoxon_rank_biserial(x, y)
                if len(x) == len(y)
                else (float("nan"), min(len(x), len(y)))
            )
        recs.append(
            {
                "mode": mode,
                "baseline": baseline_mode,
                "mean": row["mean"],
                "delta_pct": 100 * (row["mean"] - baseline_mean) / baseline_mean
                if baseline_mean
                else 0.0,
                "w_stat": w_stat,
                "p_value": p_val,
                "rank_biserial": r_rb,
                "n_pairs": n_pair,
                "cohens_d": cohens_d(x, y),
            }
        )
        raw_p.append(p_val)

    adj = ps.holm_bonferroni_adjusted(raw_p) if raw_p else []
    comparison_results = []
    print(
        f"{'Mode':<18s} {'Mean':>9s} {'Δ%':>7s} {'n':>4s} {'r_rb':>7s} "
        f"{'(d)':>7s} {'p_Holm':>11s} {'sig':>5s}"
    )
    print("-" * 100)
    for rec, ap in zip(recs, adj):
        rec["p_holm"] = ap
        sig = (
            "***" if ap < 0.001 else "**" if ap < 0.01 else "*" if ap < 0.05 else "n.s."
        )
        rec["significant"] = sig  # consumed by SUMMARY block
        comparison_results.append(rec)
        print(
            f"{rec['mode']:<18s} {rec['mean']:>9.4f} {rec['delta_pct']:>+6.1f}% "
            f"{rec['n_pairs']:>4d} {rec['rank_biserial']:>+7.3f} "
            f"{rec['cohens_d']:>+7.3f} {ps.fmt_p(ap):>11s} {sig:>5s}"
        )

    # ---- Pairwise matrix among top-5 modes: paired Wilcoxon -------------
    log.info(f"\n{'-' * 100}")
    log.info("PAIRWISE WILCOXON MATRIX (top 5 modes, Holm-adjusted)")
    log.info("-" * 100)
    top_modes = list(stats_df.head(5)["mode"].values)
    pairs = list(combinations(top_modes, 2))
    pair_p = {}
    raw = []
    for a, b in pairs:
        x, y = _vectors(a, b)
        if paired:
            _, p, _, _ = ps.paired_wilcoxon(x, y)
        else:
            from scipy import stats as _st

            _, p = _st.mannwhitneyu(x, y, alternative="two-sided")
            p = ps.cap_p(p)
        pair_p[(a, b)] = p
        raw.append(p)
    adj_m = ps.holm_bonferroni_adjusted(raw) if raw else []
    for (a, b), ap in zip(pairs, adj_m):
        pair_p[(a, b)] = ap
    print(f"\n{'':>18s}" + "".join(f"  {m[:16]:>16s}" for m in top_modes))
    print("-" * (20 + 18 * len(top_modes)))
    for m1 in top_modes:
        print(f"{m1:<18s}", end="")
        for m2 in top_modes:
            if m1 == m2:
                print(f"  {'—':>16s}", end="")
            else:
                ap = pair_p.get((m1, m2), pair_p.get((m2, m1)))
                mark = (
                    "***"
                    if ap < 0.001
                    else "**"
                    if ap < 0.01
                    else "*"
                    if ap < 0.05
                    else "n.s."
                )
                print(f"  {ap:>13.2e}{mark:>3s}", end="")
        print()

    # ---- Summary (unchanged contract) -----------------------------------
    log.info(f"\n{'=' * 100}")
    log.info("SUMMARY")
    log.info("=" * 100)
    best, worst = stats_df.iloc[0], stats_df.iloc[-1]
    log.info(f"\nBest mode: {best['mode']}  (mean {metric}={best['mean']:.4f})")
    log.info(f"Worst mode: {worst['mode']}  (mean {metric}={worst['mean']:.4f})")
    n_sig = sum(1 for r in comparison_results if r["significant"] == "***")
    log.info(
        f"Modes significantly better than {baseline_mode} "
        f"(Holm-adjusted ***): {n_sig}/{len(comparison_results)}"
    )
    return stats_df, comparison_results


def analyze_computational_overhead(df):
    """Analyze computational time (wall_time)."""
    log.info(f"\n{'=' * 100}")
    log.info("COMPUTATIONAL OVERHEAD ANALYSIS")
    log.info("=" * 100)

    stats_data = []
    for mode in sorted(df["planning_mode"].unique()):
        mode_data = df[df["planning_mode"] == mode]["wall_time"].values
        stats_data.append(
            {
                "mode": mode,
                "mean_time": np.mean(mode_data),
                "std_time": np.std(mode_data, ddof=1),
                "median_time": np.median(mode_data),
            }
        )

    stats_df = pd.DataFrame(stats_data).sort_values("mean_time")

    print(
        f"\n{'Mode':<18s}  {'Mean (s)':>10s}  {'Std (s)':>10s}  {'Median (s)':>10s}  {'Overhead':>10s}"
    )
    print("-" * 80)

    baseline_time = stats_df.iloc[0]["mean_time"]

    for _, row in stats_df.iterrows():
        overhead = (
            100 * (row["mean_time"] - baseline_time) / baseline_time
            if baseline_time > 0
            else 0
        )
        print(
            f"{row['mode']:<18s}  {row['mean_time']:>10.2f}  {row['std_time']:>10.2f}  "
            f"{row['median_time']:>10.2f}  {overhead:>+9.1f}%"
        )

    log.info("")


def _study_name_from_path(csv_path: Path) -> str:
    """Derive a human-readable study name from a CSV filename."""
    stem = csv_path.stem  # e.g. "study_main_20260409_183752"
    # Strip leading "study_" prefix
    name = stem.removeprefix("study_")
    # Keep only the family part (before the timestamp "_YYYYMMDD")
    parts = name.split("_")
    # Find first numeric token (timestamp start)
    family_parts = []
    for p in parts:
        if p.isdigit() and len(p) == 8:
            break
        family_parts.append(p)
    return "_".join(family_parts) if family_parts else name


def _analyze_one_csv(csv_path: Path, metric: str, groupby: str | None) -> None:
    """Load *csv_path* and run analysis (groupby-aware)."""
    if not csv_path.exists():
        log.error(f"File not found: {csv_path}")
        sys.exit(1)

    log.info(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)

    log.info(f"Total rows: {len(df)}")
    log.info(f"Columns: {', '.join(df.columns)}")

    if groupby and groupby in df.columns:
        for group_name, group_df in df.groupby(groupby):
            analyze_single_study(group_df, study_name=str(group_name), metric=metric)
            if "wall_time" in group_df.columns:
                analyze_computational_overhead(group_df)
    else:
        study_name = _study_name_from_path(csv_path)
        analyze_single_study(df, study_name=study_name, metric=metric)
        if "wall_time" in df.columns:
            analyze_computational_overhead(df)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze comprehensive study results with statistical tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover newest CSV per family (default)
  python scripts/analyze_study_results.py

  # Only analyse 'main' and 'drones' families
  python scripts/analyze_study_results.py --studies main drones

  # Analyse all versions of the 'budget' family
  python scripts/analyze_study_results.py --studies budget --versions all

  # Legacy: explicit CSV path (unchanged behaviour)
  python scripts/analyze_study_results.py results/comprehensive_study/study_main_*.csv
  python scripts/analyze_study_results.py results/comprehensive_study/all_studies_*.csv --groupby study
        """,
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=None,
        help="Path to a specific results CSV file (legacy mode). "
        "Omit to auto-discover CSVs in --results-dir.",
    )
    parser.add_argument(
        "--metric",
        default="likelihood",
        choices=["likelihood", "victims_found", "cells_observed", "path_length"],
        help="Primary metric to analyse (default: likelihood).",
    )
    parser.add_argument(
        "--groupby",
        default=None,
        help="Column to group by, e.g. 'study' for combined CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory to search for study CSVs (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=list(STUDY_FAMILIES),
        default=list(STUDY_FAMILIES),
        metavar="FAMILY",
        help=(
            "Study families to analyse. Choices: "
            + ", ".join(STUDY_FAMILIES)
            + "  (default: all families)."
        ),
    )
    parser.add_argument(
        "--versions",
        choices=["newest", "all"],
        default="newest",
        help="Which CSV versions to use per family: 'newest' (default) or 'all'.",
    )
    args = parser.parse_args()

    if args.csv_file:
        # ------------------------------------------------------------------ #
        # Legacy mode: user provided an explicit CSV path                     #
        # ------------------------------------------------------------------ #
        _analyze_one_csv(Path(args.csv_file), args.metric, args.groupby)
    else:
        # ------------------------------------------------------------------ #
        # Auto-discovery mode                                                 #
        # ------------------------------------------------------------------ #
        results_dir = Path(args.results_dir)
        if not results_dir.is_dir():
            log.error(f"Results directory not found: {results_dir}")
            sys.exit(1)

        csv_list = resolve_csvs(results_dir, args.studies, args.versions)
        if not csv_list:
            log.error(
                f"No study CSVs found in {results_dir} "
                f"for families: {', '.join(args.studies)}"
            )
            sys.exit(1)

        log.info(f"Resolved {len(csv_list)} CSV(s) to analyse:")
        for fam, path in csv_list:
            log.info(f"  [{fam:>12s}]  {path.name}")

        for _fam, csv_path in csv_list:
            _analyze_one_csv(csv_path, args.metric, args.groupby)

    log.info(f"\n{'=' * 100}")
    log.info("Analysis complete.")
    log.info("=" * 100)


if __name__ == "__main__":
    main()
