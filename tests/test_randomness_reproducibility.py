"""
Tests verifying random position generation and seed reproducibility
for SAR simulation runs.
"""
import numpy as np
import pytest

from sarenv.analytics.paths import generate_greedy_path


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_heatmap(size=50):
    """Create a small heatmap with a hotspot for testing."""
    hm = np.random.default_rng(0).random((size, size))
    hm /= hm.sum()
    return hm


def _make_bounds(size=50, resolution=10.0):
    return (0.0, 0.0, size * resolution, size * resolution)


def _initial_positions_random(num_drones, center_x, center_y, radius, seed):
    """Same 'random' strategy used in 10_comprehensive_study.py."""
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(num_drones):
        angle = rng.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(rng.uniform())
        positions.append((center_x + r * np.cos(angle), center_y + r * np.sin(angle)))
    return positions


def _initial_positions_circle(num_drones, center_x, center_y, radius):
    """Same 'circle' strategy used in 10_comprehensive_study.py."""
    positions = []
    for i in range(num_drones):
        angle = 2 * np.pi * i / num_drones
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        positions.append((x, y))
    return positions


# ── Test: random positions differ between runs ───────────────────────────

class TestRandomPositionsDiffer:
    """Verify that random init_strategy produces different positions for
    different seeds, and that this leads to different simulation paths."""

    def test_random_positions_vary_with_seed(self):
        """Different seeds must produce different positions."""
        center_x, center_y = 250.0, 250.0
        radius = 100.0
        num_drones = 5

        positions_per_seed = {}
        for seed in range(42, 52):  # 10 different seeds
            pos = _initial_positions_random(num_drones, center_x, center_y, radius, seed)
            positions_per_seed[seed] = pos

        # All must be mutually different
        seeds = list(positions_per_seed.keys())
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                pos_a = positions_per_seed[seeds[i]]
                pos_b = positions_per_seed[seeds[j]]
                assert pos_a != pos_b, (
                    f"Seed {seeds[i]} and {seeds[j]} produced identical positions"
                )

    def test_circle_positions_are_deterministic(self):
        """Circle strategy must produce the SAME positions regardless of seed."""
        center_x, center_y = 250.0, 250.0
        radius = 100.0
        num_drones = 5

        pos_a = _initial_positions_circle(num_drones, center_x, center_y, radius)
        pos_b = _initial_positions_circle(num_drones, center_x, center_y, radius)
        assert pos_a == pos_b

    def test_random_positions_produce_different_greedy_paths(self):
        """Different random starting positions should lead to different
        greedy 1-step paths (different coordinates)."""
        heatmap = _make_heatmap(50)
        bounds = _make_bounds(50, 10.0)
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        max_radius = 250.0
        num_drones = 3
        budget = 5_000

        path_signatures = set()
        for seed in range(42, 47):
            init_pos = _initial_positions_random(num_drones, center_x, center_y, 80.0, seed)
            paths = generate_greedy_path(
                center_x=center_x, center_y=center_y,
                num_drones=num_drones,
                probability_map=heatmap, bounds=bounds,
                max_radius=max_radius,
                fov_deg=45.0, altitude=30.0,
                budget=budget,
                initial_positions=init_pos,
                seed=seed,
            )
            # Use WKT representation of paths as a signature
            sig = tuple(p.wkt for p in paths)
            path_signatures.add(sig)

        # At least some paths must differ in geometry
        assert len(path_signatures) > 1, (
            "All 5 seeds produced identical path geometries"
        )


# ── Test: reproducibility with seed ──────────────────────────────────────

class TestSeedReproducibility:
    """Verify that using the same seed produces exactly the same results."""

    def test_random_positions_reproducible(self):
        """Same seed must produce identical positions every time."""
        center_x, center_y = 250.0, 250.0
        radius = 100.0
        num_drones = 5
        seed = 777

        pos_a = _initial_positions_random(num_drones, center_x, center_y, radius, seed)
        pos_b = _initial_positions_random(num_drones, center_x, center_y, radius, seed)

        for i in range(num_drones):
            assert pos_a[i][0] == pytest.approx(pos_b[i][0]), f"Drone {i} x mismatch"
            assert pos_a[i][1] == pytest.approx(pos_b[i][1]), f"Drone {i} y mismatch"

    def test_greedy_path_reproducible_with_seed(self):
        """generate_greedy_path with same seed and positions must produce
        identical paths."""
        heatmap = _make_heatmap(50)
        bounds = _make_bounds(50, 10.0)
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        max_radius = 250.0
        num_drones = 3
        budget = 5_000
        seed = 42

        init_pos = _initial_positions_random(num_drones, center_x, center_y, 80.0, seed)

        kwargs = dict(
            center_x=center_x, center_y=center_y,
            num_drones=num_drones,
            probability_map=heatmap, bounds=bounds,
            max_radius=max_radius,
            fov_deg=45.0, altitude=30.0,
            budget=budget,
            initial_positions=init_pos,
            seed=seed,
        )

        paths_a = generate_greedy_path(**kwargs)
        paths_b = generate_greedy_path(**kwargs)

        for i in range(num_drones):
            coords_a = list(paths_a[i].coords)
            coords_b = list(paths_b[i].coords)
            assert len(coords_a) == len(coords_b), f"Drone {i} path length mismatch"
            for j, (a, b) in enumerate(zip(coords_a, coords_b)):
                assert a[0] == pytest.approx(b[0]), f"Drone {i} point {j} x mismatch"
                assert a[1] == pytest.approx(b[1]), f"Drone {i} point {j} y mismatch"

    def test_greedy_path_different_seed_different_result(self):
        """generate_greedy_path with different seeds should produce
        different paths (when tie-breaking triggers)."""
        heatmap = _make_heatmap(50)
        bounds = _make_bounds(50, 10.0)
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        max_radius = 250.0
        num_drones = 3
        budget = 5_000

        all_lengths = []
        for seed in [42, 43, 44, 45, 46]:
            init_pos = _initial_positions_random(num_drones, center_x, center_y, 80.0, seed)
            paths = generate_greedy_path(
                center_x=center_x, center_y=center_y,
                num_drones=num_drones,
                probability_map=heatmap, bounds=bounds,
                max_radius=max_radius,
                fov_deg=45.0, altitude=30.0,
                budget=budget,
                initial_positions=init_pos,
                seed=seed,
            )
            # Use first drone's first few coords as a fingerprint
            coords = list(paths[0].coords)[:5]
            all_lengths.append(tuple(round(c[0], 2) for c in coords))

        unique = set(all_lengths)
        assert len(unique) > 1, (
            f"All seeds produced identical path starts: {unique}"
        )
