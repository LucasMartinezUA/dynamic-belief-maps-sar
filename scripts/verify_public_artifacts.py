#!/usr/bin/env python3
"""Verify the published Block A/B/C artifacts against the public manifest.

The manifest (``results/audit_block_C/public_manifest.json``) is the only place
where the published cut is described: the critical-source hash of the
implementation, the identity each shipped campaign recorded for itself, and a
sha256 index of every published file. This CLI proves that the tree on disk
matches that description.

  python scripts/verify_public_artifacts.py            verify; exit 0 if OK, 1 if not
  python scripts/verify_public_artifacts.py --update   regenerate the manifest, then verify

The frozen Block C implementation is reused rather than restated: ``_source_hash``
and ``_CRITICAL_SOURCES`` come from ``scripts/audit_block_c.py``, the frozen C7
parent campaigns come from ``scripts/c7_stage.py``. Re-deriving either here would
let the checks drift away from the runtime they are supposed to pin.

Every failure is reported on stderr; the exit status is 1 if anything failed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_block_c as cmod  # noqa: E402
import c7_stage as c7mod  # noqa: E402

MANIFEST_PATH = cmod.OUTPUT_DIR / "public_manifest.json"
MANIFEST_SCHEMA_VERSION = "public-artifact-v1"
CUT = "anonymous"
RESULT_TREES = ("audit_block_A", "audit_block_B", "audit_block_C")
C7_RUNS = {
    "c7-midflight": "c7_midflight_runs.csv.gz",
    "c7-state-matched": "c7_state_matched_runs.csv.gz",
    "c7-timing": "c7_timing_runs.csv.gz",
}
C7_FLAGS = (
    "common_horizon_ok",
    "dead_uav_silent",
    "hold_inactive_for_search",
    "step_bound_ok",
    "no_parked_repeats",
)
# c7_stage bounds one movement step by `drone_speed * dt` with a 1e-6 absolute
# tolerance (its gate literal is 5.000001 for speed 5.0 m/s and dt 1.0 s), and
# accepts a mission as budget-exact while `actual_distance_total - budget` stays
# inside `[-1e-6, drone_speed * drone_speed * dt + 1e-6]` (c7_stage._fairness_c7a).
# Both tolerances are reproduced here because the same rules must hold for the
# C7B and C7C tables, which that C7A-shaped helper does not cover.
STEP_TOL = 1e-6
# The C7A end-to-end quantities the manuscript promises are "reported with the
# released artifacts": retention under a distance budget, retention at a fixed
# horizon, and the repartitioning overhead. Only the prose documents of that
# analysis stay forbidden; the data tables ship.
FORBIDDEN_ROOT_DOCS = (
    "C7A_*.md",
    "C7_FREEZE.md",
    "C7C_GATE_RED_INCIDENT.md",
    "block_C_closure.md",
    "block_C_manifest_final*.json",
    "c_final_checks.json",
)
# Interpretive verdict documents: the reading of the results lives in the
# manuscript, so no campaign ships a second prose source for it. Machine gates,
# audit dumps and test logs stay; those are evidence, not interpretation.
FORBIDDEN_CAMPAIGN_DOCS = (
    "*_verdict.md",
    "b*_decision_report.md",
    "b*_gate_verdict.md",
)
# C7A end-to-end quantities promised by the manuscript: retention under a
# distance budget, retention at a fixed horizon, and the repartitioning
# overhead. The first two are derivable from c7_midflight_runs.csv.gz; the
# overhead exists only in the diagnostic trials table, which is why these
# tables ship. Their producers are scripts/c7a_*.py.
C7A_TABLES = {
    "C7A_TREF_GLOBAL.csv": (5, {"arm", "retention_Tref"}),
    "C7A_TREF_TRIALS.csv": (900, {"retention_budget", "retention_Tref"}),
    "C7A_PIZZA_DIAGNOSTIC_TRIALS.csv": (180, {"deadhead_fraction_remaining_budget", "prefix_removed_total"}),
    "C7A_DYNAMIC_VS_PIZZA_FIXED_DATASETS.csv": (18, {"mean_dPdet", "mean_dRMST_rel"}),
}


class Report:
    """Collects check outcomes; failures are printed to stderr as they happen."""

    def __init__(self) -> None:
        self.failures = 0
        self.checks = 0

    def check(self, ok: bool, message: str) -> bool:
        self.checks += 1
        if not ok:
            self.failures += 1
            print(f"FAIL: {message}", file=sys.stderr)
        return ok


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_files() -> dict[str, Path]:
    """Every published file under the three result trees, keyed by repo-relative path."""
    files: dict[str, Path] = {}
    for tree in RESULT_TREES:
        for path in sorted((cmod.OUTPUT_DIR.parent / tree).rglob("*")):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = path
    return files


def campaign_dirs() -> list[Path]:
    """Directories holding a campaign ``config.json``, across Block B and Block C."""
    return sorted(p.parent for p in sorted(cmod.OUTPUT_DIR.parent.rglob("*/config.json")))


def runs_tables(directory: Path) -> dict[str, Path]:
    """Raw runs tables by base name.

    The frozen preflight reads the plain ``*_runs.csv`` name, so a table may be
    shipped both plain and gzipped; either copy counts once here, and
    :func:`check_compressed_copies` proves the two carry the same bytes.
    """
    tables: dict[str, Path] = {}
    for path in sorted(directory.glob("*_runs.csv")) + sorted(directory.glob("*_runs.csv.gz")):
        tables.setdefault(path.name[:-3] if path.suffix == ".gz" else path.name, path)
    return tables


def fractions(runs: pd.DataFrame) -> pd.Series:
    """``failure_fraction`` as numbers, NaN for the runs without a mid-flight failure."""
    return pd.to_numeric(runs["failure_fraction"], errors="coerce")


def budget_reference(runs: pd.DataFrame) -> pd.Series:
    """The movement budget each row is measured against.

    The state-matched prefix arm stops at the fault budget; every other arm is
    measured against the full mission budget. Both bounds are the frozen
    ``c7_stage`` constants, so a row cannot loosen its own budget.
    """
    reference = pd.Series(float(c7mod.C7_BUDGET), index=runs.index)
    reference[runs["condition"] == "state_matched_prefix"] = float(c7mod.C7_PREFIX_BUDGET)
    return reference


def build_manifest() -> dict:
    critical_sources = {
        rel: (sha256_file(ROOT / rel) if (ROOT / rel).is_file() else "")
        for rel in cmod._CRITICAL_SOURCES
    }
    identity_map: dict[str, str] = {}
    for directory in campaign_dirs():
        config = json.loads((directory / "config.json").read_text())
        stage, source_hash = config.get("stage"), config.get("source_hash")
        if stage and source_hash:
            identity_map[str(stage)] = str(source_hash)
    artifacts = {
        rel: sha256_file(path)
        for rel, path in result_files().items()
        if path != MANIFEST_PATH
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cut": CUT,
        "critical_source_hash": cmod._source_hash(),
        "critical_sources": critical_sources,
        "identity_map": dict(sorted(identity_map.items())),
        "artifacts": dict(sorted(artifacts.items())),
    }


def write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_critical_hash(report: Report, manifest: dict) -> None:
    recomputed = cmod._source_hash()
    report.check(
        recomputed == manifest.get("critical_source_hash"),
        "critical source hash mismatch: manifest "
        f"{manifest.get('critical_source_hash')}, recomputed {recomputed}",
    )


def check_critical_sources(report: Report, manifest: dict) -> None:
    recorded = manifest.get("critical_sources") or {}
    report.check(
        set(recorded) == set(cmod._CRITICAL_SOURCES),
        "critical_sources keys differ from audit_block_c._CRITICAL_SOURCES: "
        f"missing {sorted(set(cmod._CRITICAL_SOURCES) - set(recorded))}, "
        f"extra {sorted(set(recorded) - set(cmod._CRITICAL_SOURCES))}",
    )
    for rel, expected in sorted(recorded.items()):
        path = ROOT / rel
        if not path.is_file():
            report.check(False, f"critical source missing: {rel}")
        elif sha256_file(path) != expected:
            report.check(False, f"critical source changed: {rel}")


def check_artifacts(report: Report, manifest: dict) -> None:
    recorded = manifest.get("artifacts") or {}
    on_disk = {rel: path for rel, path in result_files().items() if path != MANIFEST_PATH}
    for rel in sorted(set(recorded) - set(on_disk)):
        report.check(False, f"published artifact missing: {rel}")
    for rel in sorted(set(on_disk) - set(recorded)):
        report.check(False, f"unrecorded artifact: {rel}")
    for rel in sorted(set(recorded) & set(on_disk)):
        if sha256_file(on_disk[rel]) != recorded[rel]:
            report.check(False, f"artifact changed: {rel}")


def check_campaigns(report: Report, manifest: dict) -> None:
    identity_map = manifest.get("identity_map") or {}
    for directory in campaign_dirs():
        rel = directory.relative_to(ROOT).as_posix()
        try:
            config = json.loads((directory / "config.json").read_text())
        except json.JSONDecodeError as exc:
            report.check(False, f"{rel}/config.json is not valid JSON: {exc}")
            continue
        block_c = directory.parent.name == "audit_block_C"
        report.check(bool(config.get("stage")), f"{rel}/config.json has no stage")
        report.check(bool(config.get("config_hash")), f"{rel}/config.json has no config_hash")
        if not block_c:
            continue
        for field in ("source_hash", "expected_rows"):
            report.check(bool(config.get(field)), f"{rel}/config.json has no {field}")
        report.check(
            config.get("schema_version") == cmod.SCHEMA_VERSION,
            f"{rel}/config.json schema_version is "
            f"{config.get('schema_version')!r}, expected {cmod.SCHEMA_VERSION!r}",
        )
        report.check(
            identity_map.get(str(config.get("stage"))) == config.get("source_hash"),
            f"{rel} records source_hash {config.get('source_hash')}, manifest identity_map "
            f"has {identity_map.get(str(config.get('stage')))}",
        )
        tables = runs_tables(directory)
        report.check(bool(tables), f"{rel} has no raw runs table")
        if not tables:
            continue
        rows = sum(len(pd.read_csv(table)) for table in tables.values())
        expected = config.get("expected_rows")
        report.check(
            rows == expected,
            f"{rel} raw runs rows {rows} != expected_rows {expected}",
        )


def check_c7_parents(report: Report) -> None:
    for key, campaign in sorted(c7mod.C7_FROZEN_PARENTS.items()):
        report.check(
            (cmod.OUTPUT_DIR / campaign).is_dir(),
            f"frozen C7 parent {key} missing: results/audit_block_C/{campaign}",
        )


def check_c7a_tables(report: Report) -> None:
    for name, (expected_rows, required_columns) in sorted(C7A_TABLES.items()):
        path = cmod.OUTPUT_DIR / name
        if not path.is_file():
            report.check(False, f"C7A end-to-end table missing: {name}")
            continue
        frame = pd.read_csv(path)
        report.check(
            len(frame) == expected_rows,
            f"{name} has {len(frame)} rows, expected {expected_rows}",
        )
        report.check(
            required_columns <= set(frame.columns),
            f"{name} lacks columns {sorted(required_columns - set(frame.columns))}",
        )


def load_c7(stage: str) -> tuple[Path, pd.DataFrame]:
    filename = C7_RUNS[stage]
    matches = sorted(cmod.OUTPUT_DIR.glob(f"{stage}_*/{filename}"))
    return (matches[0], pd.read_csv(matches[0])) if matches else (Path(), pd.DataFrame())


def check_row_flags(report: Report, rel: str, runs: pd.DataFrame) -> None:
    for flag in C7_FLAGS:
        if flag not in runs.columns:
            report.check(False, f"{rel} has no {flag} column")
            continue
        offenders = int((~runs[flag].astype(bool)).sum())
        report.check(offenders == 0, f"{rel} {flag} false in {offenders} rows")
    step = runs["max_step_distance"] - runs["drone_speed"] * runs["dt"]
    report.check(
        bool((step <= STEP_TOL).all()),
        f"{rel} step bound violated in {int((step > STEP_TOL).sum())} rows",
    )
    overshoot = runs["actual_distance_total"] - budget_reference(runs)
    allowance = runs["drone_speed"] * runs["drone_speed"] * runs["dt"]
    report.check(
        bool((runs["budget"] == float(c7mod.C7_BUDGET)).all()),
        f"{rel} mission budget differs from C7_BUDGET in "
        f"{int((runs['budget'] != float(c7mod.C7_BUDGET)).sum())} rows",
    )
    report.check(
        bool((overshoot >= -STEP_TOL).all() and (overshoot <= allowance + STEP_TOL).all()),
        f"{rel} budget violated: overshoot range "
        f"[{overshoot.min():.6f}, {overshoot.max():.6f}]",
    )
    fault = runs[fractions(runs).notna()]
    if not fault.empty:
        report.check(
            bool((fault["failed_uav_post_fault_distance"] == 0).all()),
            f"{rel} failed UAV moved after the fault in "
            f"{int((fault['failed_uav_post_fault_distance'] != 0).sum())} rows",
        )
        report.check(
            bool((fault["failed_uav_post_fault_exposures"] == 0).all()),
            f"{rel} failed UAV sensed after the fault in "
            f"{int((fault['failed_uav_post_fault_exposures'] != 0).sum())} rows",
        )
        report.check(
            bool((fault["search_inactive_post_fault_exposures"] <= 1).all()),
            f"{rel} search-inactive UAV exposed more than once in "
            f"{int((fault['search_inactive_post_fault_exposures'] > 1).sum())} rows",
        )


def check_c7a(report: Report, manifest: dict) -> None:
    path, runs = load_c7("c7-midflight")
    if runs.empty:
        report.check(False, "c7_midflight_runs.csv.gz not found")
        return
    rel = path.relative_to(ROOT).as_posix()
    report.check(len(runs) == 1620, f"{rel} has {len(runs)} rows, expected 1620")
    check_row_flags(report, rel, runs)
    critical = manifest.get("critical_source_hash")
    report.check(
        bool((runs["source_hash"] == critical).all()),
        f"{rel} source_hash differs from the critical hash in "
        f"{int((runs['source_hash'] != critical).sum())} rows",
    )
    grouped = runs.groupby(["dataset", "planning_seed"])
    sizes = grouped.size()
    report.check(
        bool((sizes == 9).all()),
        f"{rel} arms per (dataset, planning_seed) differ from 9: {sorted(set(sizes))}",
    )
    spread = grouped["initial_positions_hash"].nunique()
    report.check(
        bool((spread == 1).all()),
        f"{rel} arms do not share initial_positions_hash in "
        f"{int((spread != 1).sum())} jobs",
    )


def check_c7b(report: Report) -> None:
    path, runs = load_c7("c7-state-matched")
    if runs.empty:
        report.check(False, "c7_state_matched_runs.csv.gz not found")
        return
    rel = path.relative_to(ROOT).as_posix()
    report.check(len(runs) == 900, f"{rel} has {len(runs)} rows, expected 900")
    check_row_flags(report, rel, runs)
    branches = runs[runs["condition"] == "state_matched"]
    grouped = branches.groupby(["dataset", "planning_seed"])
    sizes = grouped.size()
    report.check(
        bool((sizes == 4).all()),
        f"{rel} state-matched branches per job differ from 4: {sorted(set(sizes))}",
    )
    spread = grouped["prefailure_trajectory_hash"].nunique()
    report.check(
        bool((spread == 1).all()),
        f"{rel} branches do not share prefailure_trajectory_hash in "
        f"{int((spread != 1).sum())} jobs",
    )


def check_c7c(report: Report) -> None:
    path, runs = load_c7("c7-timing")
    if runs.empty:
        report.check(False, "c7_timing_runs.csv.gz not found")
        return
    rel = path.relative_to(ROOT).as_posix()
    report.check(len(runs) == 450, f"{rel} has {len(runs)} rows, expected 450")
    check_row_flags(report, rel, runs)

    _, midflight = load_c7("c7-midflight")
    if midflight.empty:
        return
    keys = ["dataset", "planning_seed", "planning_mode"]
    left = runs[fractions(runs) == float(c7mod.C7_PRIMARY_FRACTION)]
    right = midflight[fractions(midflight) == float(c7mod.C7_PRIMARY_FRACTION)]
    joined = left.merge(right, on=keys, suffixes=("_c7c", "_c7a"))
    report.check(
        len(joined) == len(left) and len(left) > 0,
        f"{rel} f=0.5 rows ({len(left)}) did not all pair with c7_midflight_runs "
        f"({len(joined)} paired)",
    )
    if joined.empty:
        return
    delta_p = (joined["P_detect_c7c"] - joined["P_detect_c7a"]).abs()
    delta_t = joined["T_c7c"] - joined["T_c7a"]
    report.check(
        bool((delta_p <= 1e-12).all()),
        f"{rel} paired |dP_detect| > 1e-12 in {int((delta_p > 1e-12).sum())} rows",
    )
    report.check(
        bool((delta_t == 0).all()),
        f"{rel} paired dT != 0 in {int((delta_t != 0).sum())} rows",
    )


def check_absent_artifacts(report: Report) -> None:
    report.check(
        not list(cmod.OUTPUT_DIR.glob("c7-severity*")),
        "c7-severity campaign shipped, but C7D was never run",
    )
    checkpoints = sorted((cmod.OUTPUT_DIR.parent).rglob("ckpt_*"))
    report.check(
        not checkpoints,
        "checkpoint shards shipped: " + ", ".join(p.name for p in checkpoints),
    )
    unit_reports = sorted((cmod.OUTPUT_DIR.parent).rglob("c7_unit_test_report.txt"))
    report.check(
        not unit_reports,
        "unit test reports shipped: "
        + ", ".join(p.relative_to(ROOT).as_posix() for p in unit_reports),
    )


def check_no_analysis_documents(report: Report) -> None:
    for pattern in FORBIDDEN_ROOT_DOCS:
        found = sorted(cmod.OUTPUT_DIR.glob(pattern))
        report.check(
            not found,
            f"analysis/narrative document in results/audit_block_C: "
            f"{', '.join(p.name for p in found)}",
        )
    for directory in campaign_dirs():
        rel = directory.relative_to(ROOT).as_posix()
        for pattern in FORBIDDEN_CAMPAIGN_DOCS:
            found = sorted(directory.glob(pattern))
            report.check(
                not found,
                f"verdict document in {rel}: {', '.join(p.name for p in found)}",
            )


def check_compressed_copies(report: Report) -> None:
    """Every shipped ``*.gz`` must be the gzip of the plain table beside it.

    The frozen preflights read the plain runs tables, so those tables ship in
    both forms; this proves the two copies carry the same bytes.
    """
    for path in sorted(result_files().values()):
        if path.suffix != ".gz":
            continue
        plain = path.with_suffix("")
        if not plain.is_file():
            continue
        report.check(
            gzip.decompress(path.read_bytes()) == plain.read_bytes(),
            f"{plain.relative_to(ROOT).as_posix()} differs from the decompressed "
            f"{path.relative_to(ROOT).as_posix()}",
        )


def run_checks(manifest: dict) -> int:
    report = Report()
    check_critical_hash(report, manifest)
    check_critical_sources(report, manifest)
    check_artifacts(report, manifest)
    check_campaigns(report, manifest)
    check_c7_parents(report)
    check_c7a_tables(report)
    check_c7a(report, manifest)
    check_c7b(report)
    check_c7c(report)
    check_absent_artifacts(report)
    check_no_analysis_documents(report)
    check_compressed_copies(report)
    if report.failures:
        print(f"{report.failures}/{report.checks} checks failed", file=sys.stderr)
        return 1
    print(f"{report.checks} checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="regenerate results/audit_block_C/public_manifest.json, then verify",
    )
    args = parser.parse_args()

    if args.update:
        manifest = build_manifest()
        write_manifest(manifest)
        print(f"wrote {MANIFEST_PATH.relative_to(ROOT).as_posix()}")
    else:
        if not MANIFEST_PATH.is_file():
            print(f"FAIL: manifest not found: {MANIFEST_PATH}", file=sys.stderr)
            return 1
        manifest = json.loads(MANIFEST_PATH.read_text())
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            print(
                f"FAIL: manifest schema_version is {manifest.get('schema_version')!r}, "
                f"expected {MANIFEST_SCHEMA_VERSION!r}",
                file=sys.stderr,
            )
            return 1
    return run_checks(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
