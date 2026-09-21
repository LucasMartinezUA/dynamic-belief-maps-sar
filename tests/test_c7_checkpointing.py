"""C7 campaign checkpointing machinery (audit harness level).

Verifies the durable per-job shard protocol behind ``_parallel_jobs_ckpt``:
atomic shard write, resume re-runs only missing jobs, fail-closed on corrupt
shards, and campaign resume reuses the same campaign identity.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_block_c as cmod  # noqa: E402

KEY_COLS = ["dataset", "planning_seed"]


def _worker(dataset, seed):
    row = {"dataset": dataset, "planning_seed": seed,
           "value": float(dataset * 100 + seed)}
    return pd.DataFrame([row]), {"extra": dataset}


def _args(jobs=2):
    return SimpleNamespace(jobs=jobs, no_progress=True)


@pytest.fixture()
def tmp_output(monkeypatch, tmp_path):
    monkeypatch.setattr(cmod, "OUTPUT_DIR", tmp_path)
    return tmp_path


def test_ckpt_writes_shards_and_returns_ordered(tmp_output):
    outdir = tmp_output / "cam_x"
    outdir.mkdir()
    jobs = [(1, 42), (1, 43), (5, 42), (5, 43)]
    frames = cmod._parallel_jobs_ckpt(
        _args(), jobs, _worker, "t", outdir, "tag", KEY_COLS
    )
    assert len(frames) == 4
    for frame, job in zip(frames, jobs):
        assert int(frame[0]["dataset"].iloc[0]) == job[0]
        assert int(frame[0]["planning_seed"].iloc[0]) == job[1]
        assert frame[1] == {"extra": job[0]}
    assert len(list((outdir / "ckpt_tag").glob("job_*.json"))) == 4
    assert not list((outdir / "ckpt_tag").glob("*.tmp"))


def test_ckpt_resume_runs_only_missing_jobs(tmp_output):
    outdir = tmp_output / "cam_y"
    outdir.mkdir()
    jobs = [(1, 42), (1, 43), (5, 42), (5, 43)]
    cmod._parallel_jobs_ckpt(_args(), jobs, _worker, "t", outdir, "tag", KEY_COLS)
    (outdir / "ckpt_tag" / "job_1_43.json").unlink()
    (outdir / "ckpt_tag" / "job_5_42.json").unlink()

    def stamp_worker(dataset, seed):
        row = {"dataset": dataset, "planning_seed": seed,
               "value": 9999.0 + float(dataset * 100 + seed)}
        return pd.DataFrame([row]), {"extra": dataset}

    frames = cmod._parallel_jobs_ckpt(
        _args(), jobs, stamp_worker, "t", outdir, "tag", KEY_COLS
    )
    assert len(frames) == 4
    values = {
        (int(f[0]["dataset"].iloc[0]), int(f[0]["planning_seed"].iloc[0])):
        float(f[0]["value"].iloc[0])
        for f in frames
    }
    # missing jobs were re-run (stamp marker present); completed jobs were
    # loaded from their shards (original values preserved).
    assert values[(1, 43)] == pytest.approx(9999.0 + 143.0)
    assert values[(5, 42)] == pytest.approx(9999.0 + 542.0)
    assert values[(1, 42)] == pytest.approx(142.0)
    assert values[(5, 43)] == pytest.approx(543.0)


def test_ckpt_corrupt_shard_fails_closed(tmp_output):
    outdir = tmp_output / "cam_z"
    outdir.mkdir()
    shard_dir = outdir / "ckpt_tag"
    shard_dir.mkdir()
    (shard_dir / "job_garbage.json").write_text("{not json")
    with pytest.raises(SystemExit):
        cmod._parallel_jobs_ckpt(_args(), [(1, 42)], _worker, "t", outdir, "tag", KEY_COLS)


def test_ckpt_key_mismatch_fails_closed(tmp_output):
    outdir = tmp_output / "cam_w"
    outdir.mkdir()
    shard_dir = outdir / "ckpt_tag"
    shard_dir.mkdir()
    (shard_dir / "job_mismatch.json").write_text(
        '{"key": [1, 42], "payload": [{"__dataframe__": true, '
        '"columns": ["dataset", "planning_seed"], "rows": [{"dataset": 9, "planning_seed": 99}]}]}'
    )
    with pytest.raises(SystemExit):
        cmod._parallel_jobs_ckpt(_args(), [(1, 42)], _worker, "t", outdir, "tag", KEY_COLS)


def test_make_campaign_resume_reuses_identity(tmp_output):
    base = {"stage": "c7-midflight", "implementation_version": "work", "w": 0.5}
    cfg1, outdir1 = cmod._make_campaign("c7-midflight", base, 4, allow_resume=True)
    (outdir1 / "ckpt_test").mkdir()  # durable shards exist -> resumable
    cfg2, outdir2 = cmod._make_campaign("c7-midflight", base, 4, allow_resume=True)
    assert outdir1 == outdir2
    assert cfg2["experiment_id"] == cfg1["experiment_id"]
    # without shards the same call must still fail (incomplete campaign)
    cfg3, outdir3 = cmod._make_campaign("c7-midflight", base, 4, allow_resume=True)
    assert outdir3 == outdir1  # unchanged identity on plain resume too
    assert cfg3["experiment_id"] == cfg1["experiment_id"]
