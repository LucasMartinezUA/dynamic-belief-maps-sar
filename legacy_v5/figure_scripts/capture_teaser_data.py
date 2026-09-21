#!/usr/bin/env python3
"""Capture observed cells for teaser figure: 1-drone failure scenario.

Pizza: 5-drone paths generated, drone 2 (central sector) fails → orphaned sector.
dynamic_3step: 4 surviving drones replan online → no orphaned areas.
Both use 160,000 m total budget (4 × 40,000 m per drone).

Output: ../data/teaser_cells_pizza_fail1_seed43.npy
        ../data/teaser_cells_dyn3_fail1_seed49.npy
        ../data/teaser_perdrone_pizza_fail1_seed43.npz
        ../data/teaser_perdrone_dyn3_fail1_seed49.npz
        ../data/teaser_likelihoods.txt
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sarenv.core.loading import DatasetLoader
from sarenv.analytics.simulation import SARSimulation
from sarenv.analytics.paths import generate_pizza_zigzag_path

DATA_DIR = REPO_ROOT / "legacy_v5" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = REPO_ROOT / "sarenv_dataset" / "1"
BUDGET_5 = 200_000
BUDGET_4 = 160_000  # 4 surviving drones × 40,000 m each
FAILED_DRONE = 2    # central sector (144°–216°)

common = dict(
    num_victims=5,
    fov_deg=45.0,
    victim_model="random_walk",
)

loader = DatasetLoader(DATASET_PATH)


def extract_per_drone_cells(sim):
    per_drone = {}
    for d in range(sim.num_drones):
        cells = set()
        for wx, wy in sim.drone_paths[d]:
            visible = sim._get_visible_cells_world(wx, wy)
            cells.update(visible)
        per_drone[f"drone_{d}"] = np.array(sorted(cells))
    return per_drone


# ── Pizza with 1-drone failure ───────────────────────────────────────────────
print("Running pizza with drone failure (seed=43)...")
item_p = loader.load_environment("xlarge")
minx, miny, maxx, maxy = item_p.bounds
cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
det_r = 80.0 * np.tan(np.radians(45.0 / 2))
spacing = det_r * 0.5

pizza_paths_5 = generate_pizza_zigzag_path(
    center_x=cx, center_y=cy,
    max_radius=item_p.radius_km * 1000,
    num_drones=5, fov_deg=45.0, altitude=80.0,
    overlap=0.1, path_point_spacing_m=spacing,
    border_gap_m=10.0, budget=BUDGET_5,
)
pizza_paths_4 = [p for i, p in enumerate(pizza_paths_5) if i != FAILED_DRONE]

sim_p = SARSimulation(
    dataset_item=item_p, planning_mode="static",
    static_paths=pizza_paths_4, num_drones=4,
    budget=BUDGET_4, seed=43, **common,
)
result_p = sim_p.run()

cells_p = np.array(sorted(sim_p.globally_observed_cells))
L_p = round(result_p.final_likelihood, 4)
print(f"  pizza (4d): L={L_p}, cells={len(cells_p)}")
np.save(DATA_DIR / "teaser_cells_pizza_fail1_seed43.npy", cells_p)
np.savez(DATA_DIR / "teaser_perdrone_pizza_fail1_seed43.npz", **extract_per_drone_cells(sim_p))

# ── dynamic_3step with 1-drone failure ───────────────────────────────────────
print("Running dynamic_3step with drone failure (seed=49)...")
item_d = loader.load_environment("xlarge")
sim_d = SARSimulation(
    dataset_item=item_d, planning_mode="dynamic_3step",
    num_drones=4, budget=BUDGET_4, seed=49, **common,
)
result_d = sim_d.run()

cells_d = np.array(sorted(sim_d.globally_observed_cells))
L_d = round(result_d.final_likelihood, 4)
print(f"  dynamic_3step (4d): L={L_d}, cells={len(cells_d)}")
np.save(DATA_DIR / "teaser_cells_dyn3_fail1_seed49.npy", cells_d)
np.savez(DATA_DIR / "teaser_perdrone_dyn3_fail1_seed49.npz", **extract_per_drone_cells(sim_d))

# ── Likelihoods ──────────────────────────────────────────────────────────────
with open(DATA_DIR / "teaser_likelihoods.txt", "w") as f:
    f.write(f"pizza_fail1 L={L_p}\n")
    f.write(f"dynamic_3step_fail1 L={L_d}\n")

# ── Copy heatmap for fig_teaser reproducibility ──────────────────────────────
import shutil
heatmap_src = DATASET_PATH / "heatmap.npy"
heatmap_dst = DATA_DIR / "heatmap.npy"
shutil.copy2(heatmap_src, heatmap_dst)
print(f"  heatmap copied to {heatmap_dst}")

print(f"\nSaved to: {DATA_DIR}")
