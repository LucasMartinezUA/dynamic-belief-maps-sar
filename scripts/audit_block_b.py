#!/usr/bin/env python3
"""Block B audit orchestrator: evidence-belief + operational-validity experiments.

Mirrors ``scripts/audit_block_a.py``. Each stage is a *campaign* with its own
``experiment_id`` under ``results/audit_block_B/<experiment_id>/`` and a frozen
``config.json``. Raw runs and their summaries share that campaign folder; the
``*-summary`` stages are pure post-hoc and require ``--input-experiment-id``.

The tree MUST be clean of tracked edits before a run so the recorded ``git_commit``
identifies the implementation exactly; untracked files are allowed only under
``results/audit_block_B/``.

  python scripts/audit_block_b.py belief-tests --jobs 1
  python scripts/audit_block_b.py b1-differential --jobs 8
  python scripts/audit_block_b.py b1-summary --input-experiment-id <id>
  python scripts/audit_block_b.py b2-stationary --jobs 1
  python scripts/audit_block_b.py b3-operational --jobs 8
  python scripts/audit_block_b.py b3-summary --input-experiment-id <id>
  python scripts/audit_block_b.py b5-contrasts --jobs 8
  python scripts/audit_block_b.py b5-summary --input-experiment-id <id>
  python scripts/audit_block_b.py b4-moving --jobs 8
  python scripts/audit_block_b.py b4-summary --input-experiment-id <id>
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import sarenv
from sarenv.analytics.simulation import SARSimulation
from sarenv.analytics.detection_metrics import (
    exposure_counts,
    exposure_events,
    expected_detection,
    expected_rmst,
    stationary_victim_cell,
)
from sarenv.audit import stable_hash
from sarenv.core.loading import DatasetLoader

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "audit_block_B"

P_D = 0.8
TAU = 10_000.0
BUDGET = 200_000.0
FOV_DEG = 45.0
ALTITUDE = 80.0
NUM_DRONES = 5
VICTIM_SPEED = 0.5
SIZE = "xlarge"

MARGIN_PDET = 0.01
MARGIN_RMST = 0.02
L_IMPROVE_MIN = 0.01


# ── provenance helpers ────────────────────────────────────────────────────
def _audit_commit_sha() -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


_CRITICAL_SOURCES = [
    "sarenv/analytics/detection_metrics.py",
    "sarenv/analytics/evidence_belief.py",
    "sarenv/analytics/simulation.py",
    "sarenv/audit.py",
    "sarenv/__init__.py",
    "tests/test_evidence_belief.py",
    "tests/test_detection_metrics.py",
    "scripts/audit_block_b.py",
    "scripts/run_block_b.sh",
]


def _source_hash() -> str:
    """Content hash of the Block B source bundle; identifies the exact code even
    when the source is rsync'd and not committed (remote rsync-only runs)."""
    payload = {}
    for rel in _CRITICAL_SOURCES:
        try:
            payload[rel] = (ROOT / rel).read_text()
        except FileNotFoundError:
            payload[rel] = ""
    return stable_hash(payload)


def _require_clean_tree() -> None:
    """Reject tracked edits AND untracked source/config so the recorded
    ``git_commit`` identifies the implementation exactly. Untracked artifacts
    (result CSVs, plan docs, B output) are allowed; untracked Python/config
    source is not (it would change the code without being pinned by the SHA).

    ``SARENV_B_NO_GATE=1`` skips the gate (remote rsync-only runs, where source
    is not committed; ``SARENV_B_IMPL_VERSION`` + ``source_hash`` then pin the
    recorded source).
    """
    if os.environ.get("SARENV_B_NO_GATE") == "1":
        return
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if tracked:
        raise SystemExit("Tracked working-tree changes; git_commit cannot identify the implementation")
    untracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for p in untracked:
        if p.endswith((".py", ".toml", ".yaml", ".yml")):
            raise SystemExit(f"Untracked source/config file not committed: {p}")


def _make_campaign(stage: str, base_config: dict, jobs: int):
    """Create the campaign folder (microsecond id, no-overwrite), freeze config.json."""
    config_hash = stable_hash(base_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_id = f"{stage}_{timestamp}_{config_hash[:8]}"
    config = {
        **base_config,
        "timestamp": datetime.now().isoformat(timespec="microseconds"),
        "experiment_id": experiment_id,
        "config_hash": config_hash,
    }
    outdir = OUTPUT_DIR / experiment_id
    outdir.mkdir(parents=True, exist_ok=False)   # fail if id reused -> no overwrite
    (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config, outdir


_PROVENANCE_COLS = [
    "implementation_version", "git_commit", "experiment_id", "config_hash", "source_hash",
    "belief_model", "planning_mode", "revisit_weight", "simulation_num_victims",
    "evaluation_num_victims", "victim_model", "simulation_victim_model", "victim_sampling_model",
    "detection_probability", "decay_tau", "dataset", "planning_seed", "victim_seed",
    "sensor_seed", "budget", "size",
]


def _prov(config: dict) -> dict:
    return {k: config.get(k) for k in _PROVENANCE_COLS}


def _with_provenance(frame, config):
    """Attach provenance columns to a runs frame (fill from the frozen config)."""
    frame = frame.copy()
    if "w" in frame.columns:
        frame["revisit_weight"] = frame["w"]   # per-row w overrides the global 0.5
    for col in _PROVENANCE_COLS:
        if col not in frame.columns:
            frame[col] = config.get(col)
    return frame
# ── shared helpers ────────────────────────────────────────────────────────
def _load_item(dataset_id: int):
    return DatasetLoader(f"sarenv_dataset/{dataset_id}").load_environment(SIZE)


def _initial_positions(dataset_item, num_drones: int, seed: int):
    cx = (dataset_item.bounds[0] + dataset_item.bounds[2]) / 2
    cy = (dataset_item.bounds[1] + dataset_item.bounds[3]) / 2
    minx, miny, maxx, maxy = dataset_item.bounds
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(num_drones):
        angle = rng.uniform(0, 2 * np.pi)
        r = 100.0 * np.sqrt(rng.uniform())
        x = np.clip(cx + r * np.cos(angle), minx, maxx)
        y = np.clip(cy + r * np.sin(angle), miny, maxy)
        positions.append((float(x), float(y)))
    return positions


def _build_sim(dataset_item, *, planning_mode, revisit_weight, belief_model, seed,
               num_victims, dataset_id, implementation_version):
    return SARSimulation(
        dataset_item=dataset_item,
        num_drones=NUM_DRONES,
        num_victims=num_victims,
        fov_deg=FOV_DEG,
        altitude=ALTITUDE,
        detection_probability=P_D,
        decay_tau=TAU,
        victim_speed=VICTIM_SPEED,
        victim_model="random_walk",
        budget=BUDGET,
        init_strategy="random",
        init_radius=100.0,
        drone_speed=5.0,
        planning_mode=planning_mode,
        revisit_weight=revisit_weight,
        seed=seed,
        dataset_id=dataset_id,
        implementation_version=implementation_version,
        fix_profile="revision_B_base",
        belief_model=belief_model,
    )


def _run(sim: SARSimulation, dataset_id: int, seed: int):
    initial = _initial_positions(sim.dataset, sim.num_drones, seed)
    sim.setup(initial_positions=initial, victim_positions=[])
    result = sim.run_from_state(dt=1.0, snapshot_interval=50, heatmap_interval=500)
    return result


def _detection_metrics(sim: SARSimulation, prior_flat, T=None):
    n_c = sim.cell_exposure_counts()
    P = expected_detection(prior_flat, n_c.ravel(), P_D)
    cell, ts, offsets = exposure_events(sim.drone_paths, sim._get_visible_cells_world,
                                        sim.heatmap_shape)
    if T is None:
        T = max(len(p) for p in sim.drone_paths) - 1
    R = expected_rmst(prior_flat, cell, ts, offsets, P_D, T)
    return P, R, T


def _revisit_fraction(sim: SARSimulation) -> float:
    total = int(sim.cell_exposure_counts().sum())
    unique = len(sim.globally_observed_cells)
    return (total - unique) / total if total else 0.0


def _new_coverage_fraction(sim: SARSimulation) -> float:
    unique = len(sim.globally_observed_cells)
    domain = int(sim._valid_domain_mask.sum())
    return unique / domain if domain else 0.0


def _posterior_diagnostics(heatmap):
    p = np.asarray(heatmap, dtype=np.float64)
    lp = np.zeros_like(p)
    np.log(p, out=lp, where=p > 0)
    entropy = float(-np.sum(p * lp))
    return entropy, float(p.max())


def _trace_json(sim: SARSimulation) -> str:
    return json.dumps(
        [{"replan_id": r["replan_id"], "drone": r["drone"],
          "target_row": r.get("target_row"), "target_col": r.get("target_col")}
         for r in getattr(sim, "_target_trace", [])],
        separators=(",", ":"),
    )


def _aligned_action_disagreement(left: str, right: str):
    def parse(value):
        try:
            return {(it["replan_id"], it["drone"]): (it["target_row"], it["target_col"])
                    for it in json.loads(value)}
        except (TypeError, json.JSONDecodeError, KeyError):
            return {}

    la, ra = parse(left), parse(right)
    keys = sorted(la.keys() & ra.keys())
    if not keys:
        return float("nan"), 0
    changed = sum(la[k] != ra[k] for k in keys)
    return changed / len(keys), len(keys)


def _tv(legacy_sim, evidence_sim):
    """Total-variation diagnostics over aligned heatmap snapshots."""
    def by_time(sim):
        return {s.time: s.heatmap for s in sim.snapshots if s.heatmap is not None}

    lg = by_time(legacy_sim)
    ev = by_time(evidence_sim)
    common = sorted(lg.keys() & ev.keys())
    if not common:
        return float("nan"), float("nan")
    tvs = [0.5 * np.sum(np.abs(lg[t] - ev[t])) for t in common]
    return tvs[-1], max(tvs)


def _paired(cfg: dict, dataset_id: int, seed: int, mode: str):
    """Run one (dataset, seed, mode) with both legacy and evidence; return a row."""
    item = _load_item(dataset_id)
    impl = cfg["implementation_version"]
    lg = _build_sim(item, planning_mode=mode, revisit_weight=cfg.get("revisit_weight", 0.5),
                    belief_model="legacy", seed=seed, num_victims=0,
                    dataset_id=dataset_id, implementation_version=impl)
    ev = _build_sim(item, planning_mode=mode, revisit_weight=cfg.get("revisit_weight", 0.5),
                    belief_model="evidence", seed=seed, num_victims=0,
                    dataset_id=dataset_id, implementation_version=impl)
    _run(lg, dataset_id, seed)
    _run(ev, dataset_id, seed)
    tv_final, tv_max = _tv(lg, ev)
    disagree, comparable = _aligned_action_disagreement(_trace_json(lg), _trace_json(ev))
    ent_lg, max_lg = _posterior_diagnostics(lg.dynamic_heatmap.get_current_map())
    ent_ev, max_ev = _posterior_diagnostics(ev.dynamic_heatmap.get_current_map())
    row = {
        "dataset": dataset_id, "planning_seed": seed, "planning_mode": mode,
        "w": cfg.get("revisit_weight", 0.5),
        "L_legacy": lg.snapshots[-1].likelihood_score,
        "L_evidence": ev.snapshots[-1].likelihood_score,
        "revisit_fraction_legacy": _revisit_fraction(lg),
        "revisit_fraction_evidence": _revisit_fraction(ev),
        "new_coverage_fraction_legacy": _new_coverage_fraction(lg),
        "new_coverage_fraction_evidence": _new_coverage_fraction(ev),
        "action_disagreement_rate": disagree,
        "comparable_action_decisions": comparable,
        "number_of_replans_legacy": lg._number_of_replans,
        "number_of_replans_evidence": ev._number_of_replans,
        "planner_wall_time_legacy": lg._planner_wall_time,
        "planner_wall_time_evidence": ev._planner_wall_time,
        "tv_final": tv_final, "tv_max": tv_max,
        "entropy_legacy_final": ent_lg, "entropy_evidence_final": ent_ev,
        "max_cell_legacy_final": max_lg, "max_cell_evidence_final": max_ev,
        "belief_model": "paired",
    }
    return row


def _write_frame(frame: pd.DataFrame, outdir: Path, name: str):
    frame.to_csv(outdir / name, index=False)


# ── stages ────────────────────────────────────────────────────────────────
def run_belief_tests(args, config, outdir):
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_evidence_belief.py", "-q"],
                       capture_output=True, text=True, cwd=str(ROOT))
    (outdir / "b1_belief_unit_tests.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(1)


def run_b1_differential(args, config, outdir):
    datasets = args.datasets or (1, 7, 10)
    seeds = [args.base_seed + i for i in range(args.runs)]
    modes = list(args.modes) if args.modes else ["dynamic", "dynamic_3step", "online_static", "online_static_3step"]
    jobs = [(d, s, m) for d in datasets for s in seeds for m in modes]
    if args.jobs > 1:
        rows = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(_paired)(config, d, s, m) for (d, s, m) in jobs)
    else:
        rows = [_paired(config, d, s, m) for (d, s, m) in jobs]
    frame = pd.DataFrame(rows)
    for col in _PROVENANCE_COLS:
        if col not in frame.columns:
            frame[col] = config.get(col)
    _write_frame(frame, outdir, "b1_legacy_vs_evidence_runs.csv")


def run_b1_summary(args, config, outdir):
    src = (OUTPUT_DIR / args.input_experiment_id / "b1_legacy_vs_evidence_runs.csv")
    df = pd.read_csv(src)
    df["dL"] = df["L_evidence"] - df["L_legacy"]
    grp = df.groupby("planning_mode").agg(
        n=("dL", "size"), mean_dL=("dL", "mean"),
        mean_abs_dL=("dL", lambda s: s.abs().mean()),
        mean_tv_final=("tv_final", "mean"), mean_abs_tv_max=("tv_max", lambda s: s.abs().mean()),
    ).round(6)
    grp.to_csv(outdir / "b1_legacy_vs_evidence_summary.csv")

    ev = df[df["belief_model"] == "paired"]
    lev = df.groupby(["dataset", "planning_seed"]).apply(
        lambda g: pd.Series({
            "D1_B": g.loc[g["planning_mode"] == "dynamic", "L_evidence"].iloc[0]
                   - g.loc[g["planning_mode"] == "online_static", "L_evidence"].iloc[0],
            "D3_B": g.loc[g["planning_mode"] == "dynamic_3step", "L_evidence"].iloc[0]
                   - g.loc[g["planning_mode"] == "online_static_3step", "L_evidence"].iloc[0],
        })).reset_index()
    d1 = lev["D1_B"].mean()
    d3 = lev["D3_B"].mean()
    pd.DataFrame({"metric": ["D1_B", "D3_B"],
                  "mean": [d1, d3],
                  "n": [lev["D1_B"].notna().sum(), lev["D3_B"].notna().sum()]}).round(6).to_csv(
        outdir / "b1_contrasts_summary.csv")

    diag = df.groupby(["dataset", "planning_seed", "planning_mode"]).agg(
        tv_final=("tv_final", "first"), tv_max=("tv_max", "first"),
        entropy_legacy_final=("entropy_legacy_final", "first"),
        entropy_evidence_final=("entropy_evidence_final", "first"),
        max_cell_legacy_final=("max_cell_legacy_final", "first"),
        max_cell_evidence_final=("max_cell_evidence_final", "first"),
        action_disagreement_rate=("action_disagreement_rate", "first"),
    ).reset_index().round(6)
    diag.to_csv(outdir / "b1_posterior_diagnostics.csv", index=False)

    verdict = "GREEN" if (d1 >= 0.01 and d3 >= 0.01) else (
        "AMBER" if (d1 > 0 and d3 > 0) else "RED")
    (outdir / "b1_decision_report.md").write_text(
        f"# B1 decision\n\nD1_B (mini) = {d1:.6f}\nD3_B (mini) = {d3:.6f}\n"
        f"Gate verdict: {verdict}\n\nTV and action_disagreement are diagnostics "
        f"(no handoff-defined threshold); see the summary CSVs.\n")
    (outdir / "b1_gate_status.txt").write_text(f"B1_VERDICT={verdict}\n")
    print(f"B1 D1_B={d1:.6f} D3_B={d3:.6f} verdict={verdict}")


def run_b2_stationary(args, config, outdir):
    item = _load_item(1)
    sim = _build_sim(item, planning_mode="dynamic_3step", revisit_weight=0.5,
                     belief_model="evidence", seed=args.base_seed, num_victims=0,
                     dataset_id=1, implementation_version=config["implementation_version"])
    _run(sim, 1, args.base_seed)
    prior_flat = item.heatmap.ravel()
    prior_flat = prior_flat / prior_flat.sum()   # datasets sum to ~0.9997
    P, R, T = _detection_metrics(sim, prior_flat)

    M = 100000
    sample = np.random.default_rng(123).choice(prior_flat.size, size=M, p=prior_flat)
    n_c = sim.cell_exposure_counts().ravel()
    h = 1 - (1 - P_D) ** n_c[sample]
    det_hits = np.random.default_rng(7).random(M) < h
    mc_P = det_hits.mean()
    mc_R = _mc_rmst(sample, sim, prior_flat)
    rows = [
        {"metric": "P_detect", "exact": P, "mc_mean": mc_P, "abs_error": abs(mc_P - P),
         "allowed_3sigma": 3 * np.sqrt(0.25 / M),
         "dataset": 1, "planning_seed": args.base_seed, "planning_mode": "dynamic_3step"},
        {"metric": "RMST", "exact": R, "mc_mean": mc_R, "abs_error": abs(mc_R - R),
         "allowed_3sigma": 3 * (T + 1) / np.sqrt(M),
         "dataset": 1, "planning_seed": args.base_seed, "planning_mode": "dynamic_3step"},
    ]
    _write_frame(_with_provenance(pd.DataFrame(rows), config), outdir, "b2_stationary_metric_validation.csv")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_detection_metrics.py", "-q"],
                       capture_output=True, text=True, cwd=str(ROOT))
    (outdir / "b2_detection_overlay_tests.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"B2 detection-metrics tests FAILED (rc={r.returncode})")
    for row in rows:
        if row["abs_error"] > row["allowed_3sigma"]:
            raise SystemExit(f"B2 MC validation out of tolerance for {row['metric']}: "
                             f"abs_error {row['abs_error']:.6f} > allowed {row['allowed_3sigma']:.6f}")
    (outdir / "b2_detection_metric_definition.md").write_text(
        "P_detect = sum_c P_prior(c) * (1 - (1-p_d)^n_c)\n"
        "RMST = sum_c P_prior(c) * sum_{t=0}^{T} q^{sum_{s<=t} k_s(c)}\n")
    print(f"B2 exact P={P:.6f} (MC {mc_P:.6f}) exact R={R:.6f} (MC {mc_R:.6f})")


def _mc_rmst(sample, sim, prior_flat):
    """Realized RMST MC (full pipeline): simulate Bernoulli per exposed timestep.

    Iterates only a cell's exposure timesteps (grouped by multiplicity), skipping
    the O(T) zero-hazard gaps, so it is fast on sparse exposure. The realized
    variable is Y = min(T_detect, T+1); E[Y] = sum_t S_t = RMST.
    """
    cell, ts, offsets = exposure_events(sim.drone_paths, sim._get_visible_cells_world,
                                        sim.heatmap_shape)
    T = max(len(p) for p in sim.drone_paths) - 1
    q = 1 - P_D
    ys = []
    rng = np.random.default_rng(8)
    for cidx in sample:
        lo, hi = int(offsets[cidx]), int(offsets[cidx + 1])
        if lo == hi:
            ys.append(T + 1)   # never exposed -> never detected -> censored at T+1
            continue
        i = lo
        detected = False
        while i < hi:
            tau = int(ts[i])
            m = 0
            while i < hi and int(ts[i]) == tau:
                m += 1
                i += 1
            if rng.random() < (1 - q ** m):
                ys.append(tau)
                detected = True
                break
        if not detected:
            ys.append(T + 1)
    return float(np.mean(ys))


def run_b3_operational(args, config, outdir):
    datasets = args.datasets or (1, 7, 10)
    seeds = [args.base_seed + i for i in range(args.seeds)]

    def one(d, s, w):
        item = _load_item(d)
        prior_flat = item.heatmap.ravel()
        prior_flat = prior_flat / prior_flat.sum()
        out = {"dataset": d, "planning_seed": s, "w": w}
        sims = {}
        T_static = T_dynamic = None
        for tag, mode in (("static3", "online_static_3step"), ("dyn3", "dynamic_3step")):
            sim = _build_sim(item, planning_mode=mode, revisit_weight=w,
                             belief_model="evidence", seed=s, num_victims=0,
                             dataset_id=d, implementation_version=config["implementation_version"])
            _run(sim, d, s)
            sims[tag] = sim
            out[f"L_{tag}"] = sim.snapshots[-1].likelihood_score
            out[f"revisit_{tag}"] = _revisit_fraction(sim)
            out[f"newcov_{tag}"] = _new_coverage_fraction(sim)
            out[f"replans_{tag}"] = sim._number_of_replans
            out[f"wall_{tag}"] = sim._planner_wall_time
            T_s = max(len(p) for p in sim.drone_paths) - 1
            if tag == "static3":
                T_static = T_s
            else:
                T_dynamic = T_s
        T_common = max(T_static, T_dynamic)
        for tag in ("static3", "dyn3"):
            P, R, _ = _detection_metrics(sims[tag], prior_flat, T=T_common)
            out[f"Pdet_{tag}"] = P
            out[f"RMST_{tag}"] = R
        out["horizon"] = T_common + 1
        out["T_static"] = T_static
        out["T_dynamic"] = T_dynamic
        out["T_common"] = T_common
        return out

    jobs = [(d, s, w) for d in datasets for s in seeds for w in args.w]
    rows = (Parallel(n_jobs=args.jobs, backend="loky")(delayed(one)(d, s, w) for (d, s, w) in jobs)
            if args.jobs > 1 else [one(d, s, w) for (d, s, w) in jobs])
    _write_frame(_with_provenance(pd.DataFrame(rows), config), outdir, "b3_operational_stationary_runs.csv")


def _ci(series, conf):
    n = len(series)
    if n < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(series))
    sd = float(np.std(series, ddof=1))
    se = sd / np.sqrt(n)
    try:
        from scipy import stats
        z = stats.t.ppf(0.975 if conf >= 0.95 else 0.95, df=n - 1)
    except Exception:
        z = 1.96 if conf >= 0.95 else 1.645
    return mean - z * se, mean + z * se


def _b3_verdict_one(g, H):
    """Four-state verdict for one (dataset, w) group from the paired deltas.

    RMST equivalence/improvement is assessed on the relative delta
    ``dRMST_rel = dRMST / horizon`` with the fixed margin MARGIN_RMST, so seeds
    whose ``T_common`` differ are compared on a common scale.
    """
    margin_rm = MARGIN_RMST   # fixed relative margin (dRMST_rel is normalized)
    det = (float(g["dPdet"].mean()), *_ci(g["dPdet"], 0.95), *_ci(g["dPdet"], 0.90))
    rm = (float(g["dRMST_rel"].mean()), *_ci(g["dRMST_rel"], 0.95), *_ci(g["dRMST_rel"], 0.90))
    imp_det = det[0] >= MARGIN_PDET and det[1] > 0
    wor_det = det[0] <= -MARGIN_PDET and det[2] < 0
    eq_det = det[3] > -MARGIN_PDET and det[4] < MARGIN_PDET
    imp_rm = rm[0] >= margin_rm and rm[1] > 0
    wor_rm = rm[0] <= -margin_rm and rm[2] < 0
    eq_rm = rm[3] > -margin_rm and rm[4] < margin_rm
    if (wor_det or wor_rm) and not (imp_det or imp_rm):
        return "RED"
    if (imp_det or imp_rm) and not (wor_det or wor_rm):
        return "GREEN"
    if (imp_det or imp_rm) and (wor_det or wor_rm):
        return "MIXED"
    if eq_det and eq_rm and float(g["dL"].mean()) >= L_IMPROVE_MIN:
        return "INTERMEDIATE"
    if eq_det and eq_rm:
        return "NO_BENEFIT"
    return "INCONCLUSIVE"


def run_b3_summary(args, config, outdir):
    src = (OUTPUT_DIR / args.input_experiment_id / "b3_operational_stationary_runs.csv")
    df = pd.read_csv(src)
    df["dPdet"] = df["Pdet_dyn3"] - df["Pdet_static3"]
    df["dRMST"] = df["RMST_static3"] - df["RMST_dyn3"]
    df["dRMST_rel"] = df["dRMST"] / df["horizon"]
    df["dL"] = df["L_dyn3"] - df["L_static3"]
    H = df.groupby("dataset")["horizon"].max().to_dict()
    summary = []
    for (d, w), g in df.groupby(["dataset", "w"]):
        det95 = _ci(g["dPdet"], 0.95)
        rm95 = _ci(g["dRMST"], 0.95)
        rrel95 = _ci(g["dRMST_rel"], 0.95)
        row = {"dataset": d, "w": w, "n": len(g),
               "mean_dPdet": g["dPdet"].mean(), "sd_dPdet": g["dPdet"].std(ddof=1),
               "ci95_lo_dPdet": det95[0], "ci95_hi_dPdet": det95[1],
               "mean_dRMST": g["dRMST"].mean(), "sd_dRMST": g["dRMST"].std(ddof=1),
               "ci95_lo_dRMST": rm95[0], "ci95_hi_dRMST": rm95[1],
               "mean_dRMST_rel": g["dRMST_rel"].mean(), "sd_dRMST_rel": g["dRMST_rel"].std(ddof=1),
               "ci95_lo_dRMST_rel": rrel95[0], "ci95_hi_dRMST_rel": rrel95[1],
               "mean_dL": g["dL"].mean(),
               "n_fav_dPdet": int((g["dPdet"] > 0).sum()),
               "n_fav_dRMST": int((g["dRMST"] > 0).sum())}
        summary.append(row)
    sdf = pd.DataFrame(summary).round(6)
    sdf.to_csv(outdir / "b3_operational_stationary_summary.csv", index=False)
    df.to_csv(outdir / "b3_paired_contrasts.csv", index=False)
    df["drevisit"] = df["revisit_dyn3"] - df["revisit_static3"]
    tradeoff = df.groupby(["dataset", "w"]).agg(
        mean_revisit_delta=("drevisit", "mean"), mean_dPdet=("dPdet", "mean")).round(6)
    tradeoff.to_csv(outdir / "b3_revisit_detection_tradeoff.csv")

    stats_rows = []
    for (d, w), g in df.groupby(["dataset", "w"]):
        stats_rows.append({"dataset": d, "w": w, "horizon": int(H[d]),
                           "verdict": _b3_verdict_one(g, H[d])})
    status = pd.DataFrame(stats_rows)
    status.to_csv(outdir / "b3_gate_status.csv", index=False)
    w05 = status[status["w"] == 0.5]
    red = int((w05["verdict"] == "RED").sum())
    ok = int(w05["verdict"].isin(["GREEN", "INTERMEDIATE"]).sum())
    overall = "RED" if red > 0 else ("OK" if ok >= 2 else "NOT_OK")
    (outdir / "b3_gate_status.txt").write_text(f"OVERALL_W05={overall}\n")
    (outdir / "b3_gate_verdict.md").write_text(
        "Margins (study decisions): MARGIN_PDET=0.01 absolute; MARGIN_RMST=2% of (T+1).\n"
        "w=0.5 is the confirmatory condition; w=0/1 are sensitivity.\n"
        + status.to_string(index=False) + f"\n\nOVERALL_W05={overall}\n")
    print(f"B3 overall w=0.5 verdict: {overall}")


def run_b5_contrasts(args, config, outdir):
    datasets = args.datasets or (1, 5, 7, 9, 10, 12)
    seeds = [args.base_seed + i for i in range(args.seeds)]

    def one(d, s):
        item = _load_item(d)
        out = {"dataset": d, "planning_seed": s}
        for tag, mode in (("os", "online_static"), ("os3", "online_static_3step"),
                          ("dyn", "dynamic"), ("dyn3", "dynamic_3step")):
            sim = _build_sim(item, planning_mode=mode, revisit_weight=0.5,
                             belief_model="evidence", seed=s, num_victims=0,
                             dataset_id=d, implementation_version=config["implementation_version"])
            _run(sim, d, s)
            out[f"L_{tag}"] = sim.snapshots[-1].likelihood_score
        return out

    jobs = [(d, s) for d in datasets for s in seeds]
    rows = (Parallel(n_jobs=args.jobs, backend="loky")(delayed(one)(d, s) for (d, s) in jobs)
            if args.jobs > 1 else [one(d, s) for (d, s) in jobs])
    _write_frame(_with_provenance(pd.DataFrame(rows), config), outdir, "b5_contrasts_runs.csv")


def run_b5_summary(args, config, outdir):
    src = (OUTPUT_DIR / args.input_experiment_id / "b5_contrasts_runs.csv")
    df = pd.read_csv(src)
    df["D1_B"] = df["L_dyn"] - df["L_os"]
    df["D3_B"] = df["L_dyn3"] - df["L_os3"]
    df["I_B"] = df["D3_B"] - df["D1_B"]
    df["H_S_B"] = df["L_os3"] - df["L_os"]
    df["H_D_B"] = df["L_dyn3"] - df["L_dyn"]
    df.to_csv(outdir / "b5_causal_contrasts.csv", index=False)
    print(df[["D1_B", "D3_B", "I_B", "H_S_B", "H_D_B"]].mean().round(6))
    print("pos/%d:" % len(df), {c: int((df[c] > 0).sum()) for c in
                                ("D1_B", "D3_B", "I_B", "H_S_B", "H_D_B")})


def _moving_metrics(sim, prior_flat, victim_seed, T=None):
    """Single moving-victim overlay: exact hazard integration along the trajectory.

    Samples the victim start cell from the matched prior, advances a
    ``RandomWalkModel`` for the mission, and at each timestep integrates the
    sensor hazard ``h_t = 1 - (1-p_d)^k_t`` where ``k_t`` is the number of drones
    whose FOV contains the victim. Detects via the SAME continuous rule as
    ``SARSimulation`` (``dist(drone, victim) <= detection_radius``), so the overlay
    is an exact realisation of the simulator's sensor, not a cell approximation.
    """
    from shapely.geometry import Point
    from sarenv.core.victim_models import RandomWalkModel

    cell = stationary_victim_cell(prior_flat, victim_seed)
    row, col = divmod(int(cell), sim.heatmap_shape[1])
    wx0, wy0 = sim._grid_to_world(row, col)
    vm = RandomWalkModel(speed=VICTIM_SPEED, seed=victim_seed)
    pos = Point(wx0, wy0)
    if T is None:
        T = max(len(p) for p in sim.drone_paths) - 1
    if T < 0:
        return 0.0, 0.0

    vpos = [pos]
    for _ in range(T):
        pos = vm.move(pos, 1.0, sim.dataset.features, sim.bounds)
        vpos.append(pos)

    r = sim.detection_radius
    q = 1 - P_D
    S = 1.0
    sum_S = 0.0
    for t in range(T + 1):
        vx, vy = vpos[t].x, vpos[t].y
        k = 0
        for dp in sim.drone_paths:
            if t < len(dp):
                if np.hypot(dp[t][0] - vx, dp[t][1] - vy) <= r:
                    k += 1
        h = 1 - q ** k
        S *= (1 - h)
        sum_S += S
    return 1.0 - S, sum_S  # P_detect, RMST


def run_b4_moving(args, config, outdir):
    datasets = args.datasets or (1, 7, 10)
    seeds = [args.base_seed + i for i in range(args.seeds)]

    def one(d, s, w):
        item = _load_item(d)
        prior_flat = item.heatmap.ravel()
        prior_flat = prior_flat / prior_flat.sum()
        victim_seed = s ^ 0x5DEECE66D
        sensor_seed = s ^ 0x5DEECE66D ^ 0x9E3779B9
        row = {"dataset": d, "planning_seed": s, "victim_seed": victim_seed,
               "sensor_seed": sensor_seed, "w": w}
        sims = {}
        T_static = T_dynamic = None
        for tag, mode in (("static3", "online_static_3step"), ("dyn3", "dynamic_3step")):
            sim = _build_sim(item, planning_mode=mode, revisit_weight=w,
                             belief_model="evidence", seed=s, num_victims=0,
                             dataset_id=d, implementation_version=config["implementation_version"])
            _run(sim, d, s)
            sims[tag] = sim
            row[f"L_{tag}"] = sim.snapshots[-1].likelihood_score
            T_s = max(len(p) for p in sim.drone_paths) - 1
            if tag == "static3":
                T_static = T_s
            else:
                T_dynamic = T_s
        T_common = max(T_static, T_dynamic)
        for tag in ("static3", "dyn3"):
            P, R = _moving_metrics(sims[tag], prior_flat, victim_seed, T=T_common)
            row[f"Pdet_{tag}"] = P
            row[f"RMST_{tag}"] = R
        row["T_static"] = T_static
        row["T_dynamic"] = T_dynamic
        row["T_common"] = T_common
        row["horizon"] = T_common + 1
        return row

    jobs = [(d, s, w) for d in datasets for s in seeds for w in args.w]
    rows = (Parallel(n_jobs=args.jobs, backend="loky")(delayed(one)(d, s, w) for (d, s, w) in jobs)
            if args.jobs > 1 else [one(d, s, w) for (d, s, w) in jobs])
    _write_frame(_with_provenance(pd.DataFrame(rows), config), outdir, "b4_moving_victim_runs.csv")


def run_b4_summary(args, config, outdir):
    src = (OUTPUT_DIR / args.input_experiment_id / "b4_moving_victim_runs.csv")
    df = pd.read_csv(src)
    df["dPdet"] = df["Pdet_dyn3"] - df["Pdet_static3"]
    df["dRMST"] = df["RMST_static3"] - df["RMST_dyn3"]
    df["dRMST_rel"] = df["dRMST"] / df["horizon"]
    rows = []
    for (d, w), g in df.groupby(["dataset", "w"]):
        det95 = _ci(g["dPdet"], 0.95)
        rm95 = _ci(g["dRMST"], 0.95)
        rrel95 = _ci(g["dRMST_rel"], 0.95)
        rows.append({"dataset": d, "w": w, "n": len(g),
                     "mean_dPdet": g["dPdet"].mean(), "sd_dPdet": g["dPdet"].std(ddof=1),
                     "ci95_lo_dPdet": det95[0], "ci95_hi_dPdet": det95[1],
                     "mean_dRMST": g["dRMST"].mean(), "sd_dRMST": g["dRMST"].std(ddof=1),
                     "ci95_lo_dRMST": rm95[0], "ci95_hi_dRMST": rm95[1],
                     "mean_dRMST_rel": g["dRMST_rel"].mean(), "sd_dRMST_rel": g["dRMST_rel"].std(ddof=1),
                     "ci95_lo_dRMST_rel": rrel95[0], "ci95_hi_dRMST_rel": rrel95[1]})
    pd.DataFrame(rows).round(6).to_csv(outdir / "b4_moving_victim_summary.csv", index=False)


def run_regression(args, config, outdir):
    """Legacy regression gate: must stay green on the full repo before B1."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_dynamic_heatmap.py",
         "tests/test_audit_heatmap.py", "tests/test_simulation.py", "-q"],
        capture_output=True, text=True, cwd=str(ROOT))
    (outdir / "b0_regression.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(1)


def _stage_design(stage, args) -> dict:
    """Resolved design actually executed by a stage (for the frozen config)."""
    ds = args.datasets
    dsl = (ds or (1, 7, 10)) if stage in ("b1-differential", "b3-operational", "b4-moving") else (
        (ds or (1, 5, 7, 9, 10, 12)) if stage == "b5-contrasts" else (ds or [1]))
    design = {"datasets": list(dsl), "jobs": args.jobs,
              "base_seed": args.base_seed, "runs": args.runs, "seeds": args.seeds,
              "w": list(args.w), "simulation_victim_model": "random_walk"}
    if stage == "b1-differential":
        design["modes"] = list(args.modes) if args.modes else ["dynamic", "dynamic_3step", "online_static", "online_static_3step"]
        design["beliefs"] = ["legacy", "evidence"]
        design["seeds_used"] = [args.base_seed + i for i in range(args.runs)]
        design["victim_sampling_model"] = "none_planning_only"
        design["w"] = [0.5]
    elif stage == "b2-stationary":
        design["planning_mode"] = "dynamic_3step"
        design["revisit_weight"] = 0.5
        design["evaluation"] = "stationary"
        design["victim_sampling_model"] = "matched_prior_stationary_exact"
        design["w"] = [0.5]
    elif stage in ("b3-operational", "b4-moving"):
        design["planners"] = ["online_static_3step", "dynamic_3step"]
        design["seeds_used"] = [args.base_seed + i for i in range(args.seeds)]
        design["evaluation"] = "stationary" if stage == "b3-operational" else "moving"
        design["victim_sampling_model"] = ("matched_prior_stationary_exact"
                                           if stage == "b3-operational" else "matched_prior_random_walk")
    elif stage == "b5-contrasts":
        design["modes"] = ["online_static", "online_static_3step", "dynamic", "dynamic_3step"]
        design["seeds_used"] = [args.base_seed + i for i in range(args.seeds)]
        design["w"] = [0.5]
        design["victim_sampling_model"] = "none_planning_only"
    return design


STAGES = {
    "regression": run_regression,
    "belief-tests": run_belief_tests,
    "b1-differential": run_b1_differential,
    "b1-summary": run_b1_summary,
    "b2-stationary": run_b2_stationary,
    "b3-operational": run_b3_operational,
    "b3-summary": run_b3_summary,
    "b5-contrasts": run_b5_contrasts,
    "b5-summary": run_b5_summary,
    "b4-moving": run_b4_moving,
    "b4-summary": run_b4_summary,
}


def main():
    _require_clean_tree()
    git_commit = os.environ.get("SARENV_B_IMPL_VERSION") or _audit_commit_sha()
    impl = f"revision_B@{git_commit[:12]}"
    base = {
        "implementation_version": impl,
        "git_commit": git_commit,
        "source_hash": _source_hash(),
        "belief_model": "evidence",
        "revisit_weight": 0.5,
        "simulation_num_victims": 0,
        "evaluation_num_victims": 1,
        "victim_model": "random_walk",
        "detection_probability": P_D,
        "decay_tau": TAU,
        "budget": BUDGET,
        "size": SIZE,
        "planning_seed": 42,
        "victim_seed": None,
        "sensor_seed": None,
    }
    parser = argparse.ArgumentParser(description="Block B audit orchestrator")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--input-experiment-id", default=None)
    parser.add_argument("--datasets", nargs="+", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--w", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    args = parser.parse_args()

    if args.input_experiment_id:
        # post-hoc summary: reuse the campaign folder so artifacts stay together
        outdir = OUTPUT_DIR / args.input_experiment_id
        config = json.loads((outdir / "config.json").read_text())
        STAGES[args.stage](args, config, outdir)
        print(f"summary written to {outdir}")
        return

    stage_config = dict(base)
    stage_config["stage"] = args.stage
    if args.datasets:
        stage_config["datasets"] = args.datasets
    if args.modes:
        stage_config["modes"] = args.modes
    stage_config["design"] = _stage_design(args.stage, args)
    stage_config["victim_sampling_model"] = stage_config["design"].get("victim_sampling_model")
    stage_config["simulation_victim_model"] = stage_config["design"].get("simulation_victim_model")
    config, outdir = _make_campaign(args.stage, stage_config, args.jobs)
    print(f"campaign {config['experiment_id']} -> {outdir}")
    STAGES[args.stage](args, config, outdir)


if __name__ == "__main__":
    main()
