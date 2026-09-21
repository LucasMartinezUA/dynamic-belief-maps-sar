#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Two archived v5 runners import scripts/paired_stats.py by module name; the v5
# layout kept them in scripts/, so keep that directory on the import path.
export PYTHONPATH="${ROOT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

pixi install

PY=(pixi run python)

run() {
    printf '\n==> %s\n' "$*"
    "$@"
}

# The v5 runners write under results/; the archived figure scripts create the
# data directory they read from (legacy_v5/data) themselves.
mkdir -p results

run "${PY[@]}" scripts/download_datasets.py

run "${PY[@]}" scripts/legacy_v5/comprehensive_study.py --studies 5 6 --runs 30 --budget 200000 -j 8
run "${PY[@]}" scripts/legacy_v5/comprehensive_study.py --studies 3 --runs 100 --budget 200000 -j 8
run "${PY[@]}" scripts/legacy_v5/nstep_sweep.py -j 10
run "${PY[@]}" scripts/legacy_v5/tree_search_baseline.py -j 10
run "${PY[@]}" scripts/legacy_v5/tree_crossover.py
run "${PY[@]}" scripts/legacy_v5/drone_failure_resilience.py
run "${PY[@]}" scripts/legacy_v5/analyze_study_results.py

run "${PY[@]}" legacy_v5/figure_scripts/analysis_cliff_delta_lmm.py
run "${PY[@]}" legacy_v5/figure_scripts/capture_teaser_data.py

run "${PY[@]}" legacy_v5/figure_scripts/fig_teaser.py
run "${PY[@]}" legacy_v5/figure_scripts/fig_pairwise_combined.py
run "${PY[@]}" legacy_v5/figure_scripts/fig_nstep_sweep.py

run "${PY[@]}" legacy_v5/experiments/4_victim_model/scripts/plot_victim_model.py