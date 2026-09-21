# sarenv/analytics/simulation.py
"""
Discrete-timestep SAR simulation engine.

Each timestep:
  1. Move victims according to their pedestrian model
  2. Move drones (greedy 1-step on current posterior)
  3. Update heatmap with observed cells (Bayesian reduction)
  4. Check victim detection (FOV ∩ victim position + probability roll)
  5. Apply exponential decay toward prior
  6. Track budget consumed
  7. Record snapshot
"""
import re
import time as _time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve, maximum_filter
from shapely.geometry import Point, LineString

from sarenv.analytics.dynamic_heatmap import DynamicHeatmap
from sarenv.audit import AuditFixes, make_audit_record, stable_hash
from sarenv.core.loading import SARDatasetItem
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.core.victim_models import VictimModel, RandomWalkModel, VICTIM_MODELS
from sarenv.utils.logging_setup import get_logger

log = get_logger()


@dataclass
class SimulationSnapshot:
    """State captured at a single timestep."""
    time: float
    drone_positions: list[tuple[float, float]]
    victim_positions: list[tuple[float, float]]
    victims_found: list[bool]
    likelihood_score: float
    cells_observed: int
    heatmap: np.ndarray | None = None  # only stored at snapshot intervals

@dataclass
class SimulationResult:
    """Final results from a completed simulation."""
    params: dict
    snapshots: list[SimulationSnapshot]
    final_drone_paths: list[list[tuple[float, float]]]
    total_time: float
    total_distance: float
    victims_found_count: int
    final_likelihood: float
    final_heatmap: np.ndarray = field(default_factory=lambda: np.array([]))
    wall_time: float = 0.0
    planner_wall_time: float = 0.0
    shadow_wall_time: float = 0.0
    number_of_replans: int = 0
    number_of_target_conflicts: int = 0
    reservation_blocked: int = 0
    first_action_sequence_hash: str = ""
    planning_trace: list[dict] = field(default_factory=list)
    target_trace: list[dict] = field(default_factory=list)
    audit_record: dict | None = None


class SARSimulation:
    """
      - "random_uniform_n8_uncoordinated": uninformed memoryless per-UAV RNG walk
        over the physically valid N8 actions (C6 exploration floor; the policy
        consults no probability map and performs no target deconfliction)

    The N-step planner uses Bellman's equation with vectorised operations:
      1. score_map = convolve(weighted_heatmap, FOV_kernel)   (once per step)
      2. V = score_map
         for k in range(N-1):
             V = score_map + maximum_filter(V, 3×3)
      3. Each drone picks its best 8-neighbour in V
    """

    _NSTEP_RE = re.compile(r'^(static|dynamic|online_static|tree)_(\d+)step$')

    def __init__(
        self,
        dataset_item: SARDatasetItem,
        num_drones: int = 3,
        num_victims: int = 5,
        fov_deg: float = 45.0,
        altitude: float = 80.0,
        detection_probability: float = 0.8,
        decay_tau: float = 10_000.0,
        victim_speed: float = 0.5,
        victim_model: str | VictimModel = "random_walk",
        budget: float = 100_000.0,
        init_strategy: str = "circle",
        init_radius: float = 100.0,
        drone_speed: float = 5.0,
        revisit_weight: float = 0.0,
        planning_mode: str = "dynamic",
        static_paths: list[LineString] | None = None,
        seed: int = 42,
        dataset_id: int | str | None = None,
        implementation_version: str = "working_tree",
        fix_profile: str = "original",
        belief_model: str = "legacy",
        policy_seed: int | None = None,
        drone_identity: list[int] | None = None,
        midflight_failure_fraction: float | None = None,
        midflight_failed_uav_ids: list[int] | None = None,
        shadow_diagnostics: bool = True,
    ):
        base_mode, lookahead = self._parse_planning_mode(planning_mode)
        if base_mode in ("static", "static_nstep", "pizza_replan", "pizza_fixed") and not static_paths:
            raise ValueError(f"planning_mode='{planning_mode}' requires static_paths")
        if base_mode == "random_n8" and policy_seed is None:
            raise ValueError(
                "planning_mode='random_uniform_n8_uncoordinated' requires policy_seed"
            )
        self._base_mode = base_mode
        self._lookahead_steps = lookahead
        self.dataset = dataset_item
        self.dataset_id = dataset_id
        self.num_drones = num_drones
        self.num_victims = num_victims
        self.fov_deg = fov_deg
        self.altitude = altitude
        self.detection_probability = detection_probability
        self.decay_tau = decay_tau
        self.victim_speed = victim_speed
        self.budget = budget
        self.init_strategy = init_strategy
        self.init_radius = init_radius
        self.drone_speed = drone_speed
        self.revisit_weight = revisit_weight
        self.planning_mode = planning_mode
        self.static_paths = static_paths
        self.seed = seed
        self.implementation_version = implementation_version
        self.fix_profile = fix_profile
        if belief_model not in ("legacy", "evidence"):
            raise ValueError(f"belief_model must be 'legacy' or 'evidence', got {belief_model!r}")
        self.belief_model = belief_model
        self.shadow_diagnostics = bool(shadow_diagnostics)
        self.audit_fixes = AuditFixes.from_profile(fix_profile)

        # Path-following state for static mode
        self._path_progress: list[float] = []

        # Resolve victim model
        if isinstance(victim_model, str):
            model_cls = VICTIM_MODELS.get(victim_model)
            if model_cls is None:
                raise ValueError(f"Unknown victim model '{victim_model}'. Choose from {list(VICTIM_MODELS)}")
            self._victim_model = model_cls(speed=victim_speed, seed=seed)
        else:
            self._victim_model = victim_model

        self.rng = np.random.default_rng(seed)

        # C6 random policy streams: one independent RNG per persistent UAV
        # identity, keyed only on (policy_seed, uav_id). Removing another UAV
        # never shifts the stream of a survivor, and there is no shared
        # sequential consumption across UAVs.
        self.policy_seed = policy_seed
        if drone_identity is None:
            self.drone_identity = list(range(self.num_drones))
        elif len(drone_identity) != self.num_drones:
            raise ValueError("drone_identity length must equal num_drones")
        else:
            self.drone_identity = [int(identity) for identity in drone_identity]
        self._policy_rngs = (
            [self._make_policy_rng(identity) for identity in self.drone_identity]
            if base_mode == "random_n8"
            else []
        )

        # C7 mid-mission failure configuration. A single failure event per run:
        # after the consumed fleet movement crosses B_fail = fraction * budget,
        # the configured UAV ids become failed (silent; no move/sense/budget).
        if midflight_failure_fraction is not None:
            if not 0.0 < float(midflight_failure_fraction) <= 1.0:
                raise ValueError(
                    f"midflight_failure_fraction must be in (0, 1], "
                    f"got {midflight_failure_fraction!r}"
                )
            if midflight_failed_uav_ids is None:
                raise ValueError(
                    "midflight_failed_uav_ids is required when "
                    "midflight_failure_fraction is set"
                )
            raw_ids = list(midflight_failed_uav_ids)
            if not all(isinstance(item, (int, np.integer)) for item in raw_ids):
                raise ValueError("midflight_failed_uav_ids must contain integers")
            ids = [int(item) for item in raw_ids]
            if not (1 <= len(ids) < self.num_drones):
                raise ValueError(
                    "midflight_failed_uav_ids length must be >= 1 and < num_drones"
                )
            if any(item < 0 or item >= self.num_drones for item in ids):
                raise ValueError(
                    "midflight_failed_uav_ids entries must be within range(num_drones)"
                )
            if len(set(ids)) != len(ids):
                raise ValueError("midflight_failed_uav_ids entries must be distinct")
            self._midflight_failure_fraction = float(midflight_failure_fraction)
            self._midflight_failed_uav_ids = ids
            self._midflight_B_fail = self._midflight_failure_fraction * self.budget
        else:
            if midflight_failed_uav_ids is not None:
                raise ValueError(
                    "midflight_failed_uav_ids requires midflight_failure_fraction"
                )
            self._midflight_failure_fraction = None
            self._midflight_failed_uav_ids = None
            self._midflight_B_fail = None
        self._failed_uav_ids: list[int] | None = None
        self._search_inactive_uav_ids: list[int] = []
        # A survivor that has just reached the end of its assigned task in the
        # current timestep: it senses the final waypoint ONCE (this timestep)
        # and is promoted to search-inactive right after the sensing block.
        self._pending_search_inactive_uav_ids: list[int] = []
        self._midflight_triggered = False
        self._midflight_info: dict | None = None

        # Derived
        self.detection_radius = altitude * np.tan(np.radians(fov_deg / 2))
        self.bounds = dataset_item.bounds
        self.heatmap_shape = dataset_item.heatmap.shape
        height, width = self.heatmap_shape
        minx, miny, maxx, maxy = self.bounds
        self.dx = (maxx - minx) / width
        self.dy = (maxy - miny) / height
        self.x_offset = minx + self.dx / 2
        self.y_offset = miny + self.dy / 2
        self.detection_radius_cells_x = int(np.ceil(self.detection_radius / self.dx))
        self.detection_radius_cells_y = int(np.ceil(self.detection_radius / self.dy))
        self.max_radius = dataset_item.radius_km * 1000
        self.center_x = (minx + maxx) / 2
        self.center_y = (miny + maxy) / 2
        rows, cols = np.indices(self.heatmap_shape)
        world_x = self.x_offset + cols * self.dx
        world_y = self.y_offset + rows * self.dy
        self._valid_domain_mask = (
            (world_x - self.center_x) ** 2
            + (world_y - self.center_y) ** 2
            < self.max_radius ** 2
        )

        # State (initialized in setup)
        self.dynamic_heatmap: DynamicHeatmap | None = None
        self.drone_positions: list[tuple[float, float]] = []
        self.drone_paths: list[list[tuple[float, float]]] = []
        self.victim_positions: list[Point] = []
        self.victims_found: list[bool] = []
        self.globally_observed_cells: set = set()
        # Float mirror of ``globally_observed_cells`` (1.0 for observed cells);
        # maintained incrementally so ``_compute_score_map`` never pays a
        # Python pass over the whole observed set at replan time. It is rebuilt
        # only when the set is mutated out-of-band (detected via length).
        self._observed_mask_np = np.zeros(self.heatmap_shape, dtype=np.float64)
        self._observed_mask_count = 0
        self.total_distance: float = 0.0
        self.sim_time: float = 0.0
        self.snapshots: list[SimulationSnapshot] = []
        self._done = False
        self._planner_wall_time = 0.0
        self._shadow_wall_time = 0.0
        self._number_of_replans = 0
        self._number_of_target_conflicts = 0
        self._reservation_blocked = 0
        self._first_actions: list[tuple[int, int, int]] = []
        self._planning_trace: list[dict] = []

        # Neighbor offsets for greedy movement (8-connected)
        self._neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Pre-compute circular FOV kernel for convolution-based planning
        self._fov_kernel = self._build_fov_kernel()

    def setup(
        self,
        initial_positions: list[tuple[float, float]] | None = None,
        victim_positions: list[Point] | None = None,
    ):
        """Initialize simulation state and observe the initial FOV exactly once."""
        if self.belief_model == "evidence":
            from sarenv.analytics.evidence_belief import EvidenceBelief
            self.dynamic_heatmap = EvidenceBelief(
                self.dataset.heatmap,
                detection_probability=self.detection_probability,
                decay_tau=self.decay_tau,
            )
        else:
            self.dynamic_heatmap = DynamicHeatmap(
                self.dataset.heatmap,
                detection_probability=self.detection_probability,
                decay_tau=self.decay_tau,
                time_zero_observations=self.audit_fixes.time_zero_observations,
            )

        # External positions are accepted for paired experiments. Static paths
        # still use grid-aligned starts so path-following and path generation
        # share the same first cell.
        wrld_positions = (
            list(initial_positions)
            if initial_positions is not None
            else self._generate_initial_positions()
        )
        if self.static_paths is not None:
            self.drone_positions = [
                self._grid_to_world(*self._world_to_grid(wx, wy))
                for wx, wy in wrld_positions
            ]
        else:
            self.drone_positions = wrld_positions
        self.drone_paths = [[pos] for pos in self.drone_positions]

        self.victim_positions = (
            list(victim_positions)
            if victim_positions is not None
            else self._generate_victim_positions()
        )
        self.victims_found = [False] * self.num_victims

        self.globally_observed_cells = set()
        self._observed_mask_np[:] = 0.0
        self._observed_mask_count = 0
        initial_cells = set()
        for pos in self.drone_positions:
            cells = self._get_visible_cells_world(pos[0], pos[1])
            self.globally_observed_cells.update(cells)
            self._sync_observed_mask(cells)
            initial_cells.update(cells)
            if not self.audit_fixes.union_observations_per_timestep:
                self.dynamic_heatmap.update(cells, current_time=0.0)
        if self.audit_fixes.union_observations_per_timestep:
            self.dynamic_heatmap.update(initial_cells, current_time=0.0)

        self.total_distance = 0.0
        self.sim_time = 0.0
        self._done = False
        self.snapshots = []
        self._planner_wall_time = 0.0
        self._shadow_wall_time = 0.0
        self._number_of_replans = 0
        self._number_of_target_conflicts = 0
        self._reservation_blocked = 0
        self._first_actions = []
        self._planning_trace = []
        self._target_trace = []
        self._current_replan_id = 0

        if (
            self.planning_mode.startswith("static")
            or self._base_mode in ("pizza_replan", "pizza_fixed")
        ) and self.static_paths:
            self._path_progress = [0.0] * len(self.static_paths)

        self._drone_targets: list[tuple[float, float] | None] = [None] * self.num_drones
        self._cached_V: np.ndarray | None = None
        self._record_snapshot(store_heatmap=True)


    def _generate_initial_positions(self) -> list[tuple[float, float]]:
        """Generate drone starting positions using the configured strategy."""
        cx, cy = self.center_x, self.center_y
        minx, miny, maxx, maxy = self.bounds
        positions = []

        if self.init_strategy == "circle":
            for i in range(self.num_drones):
                angle = 2 * np.pi * i / self.num_drones
                x = cx + self.init_radius * np.cos(angle)
                y = cy + self.init_radius * np.sin(angle)
                x = np.clip(x, minx, maxx)
                y = np.clip(y, miny, maxy)
                positions.append((float(x), float(y)))
        elif self.init_strategy == "center":
            for i in range(self.num_drones):
                x = cx + i * 5.0
                y = cy + i * 5.0
                positions.append((np.clip(x, minx, maxx), np.clip(y, miny, maxy)))
        elif self.init_strategy == "grid":
            side = int(np.ceil(np.sqrt(self.num_drones)))
            spacing = 2 * self.init_radius / max(side - 1, 1)
            idx = 0
            for r in range(side):
                for c in range(side):
                    if idx >= self.num_drones:
                        break
                    x = cx - self.init_radius + c * spacing
                    y = cy - self.init_radius + r * spacing
                    positions.append((np.clip(x, minx, maxx), np.clip(y, miny, maxy)))
                    idx += 1
        elif self.init_strategy == "random":
            for _ in range(self.num_drones):
                angle = self.rng.uniform(0, 2 * np.pi)
                r = self.init_radius * np.sqrt(self.rng.uniform())
                x = np.clip(cx + r * np.cos(angle), minx, maxx)
                y = np.clip(cy + r * np.sin(angle), miny, maxy)
                positions.append((float(x), float(y)))
        else:
            raise ValueError(f"Unknown init_strategy '{self.init_strategy}'")

        return positions

    def _generate_victim_positions(self) -> list[Point]:
        """Generate initial victim positions using LostPersonLocationGenerator."""
        gen = LostPersonLocationGenerator(self.dataset, seed=self.seed)
        return gen.generate_locations(n=self.num_victims, percent_random_samples=0)

    def _get_visible_cells_world(self, wx: float, wy: float) -> set[tuple[int, int]]:
        """Get visible grid cells from a world-coordinate position.

        Vectorized but bit-exact with the per-cell formulation: the same
        subtractions, squares, ``sqrt`` and comparisons are performed (numpy
        elementwise IEEE operations on the same doubles), so the resulting
        cell sets are identical.
        """
        height, width = self.heatmap_shape
        center_col = int((wx - self.bounds[0]) / self.dx)
        center_row = int((wy - self.bounds[1]) / self.dy)

        r0 = max(0, center_row - self.detection_radius_cells_y)
        r1 = min(height, center_row + self.detection_radius_cells_y + 1)
        c0 = max(0, center_col - self.detection_radius_cells_x)
        c1 = min(width, center_col + self.detection_radius_cells_x + 1)
        if r0 >= r1 or c0 >= c1:
            return set()

        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        cell_x = self.x_offset + cols * self.dx
        cell_y = self.y_offset + rows * self.dy
        dist = np.sqrt(
            (cell_x - wx)[np.newaxis, :] ** 2 + (cell_y - wy)[:, np.newaxis] ** 2
        )
        rr, cc = np.nonzero(dist <= self.detection_radius)
        return {
            (int(r0 + int(r)), int(c0 + int(c)))
            for r, c in zip(rr.tolist(), cc.tolist())
        }

    def _world_to_grid(self, wx, wy):
        col = int((wx - self.bounds[0]) / self.dx)
        row = int((wy - self.bounds[1]) / self.dy)
        h, w = self.heatmap_shape
        return (np.clip(row, 0, h - 1), np.clip(col, 0, w - 1))

    def _grid_to_world(self, row, col):
        return (self.x_offset + col * self.dx, self.y_offset + row * self.dy)

    def step(self, dt: float = 1.0) -> bool:
        """
        Advance simulation by one timestep.

        Returns:
            True if simulation should continue, False if done.
        """
        if self._done:
            return False

        self.sim_time += dt

        # 1. Move victims
        for i, pos in enumerate(self.victim_positions):
            if not self.victims_found[i]:
                self.victim_positions[i] = self._victim_model.move(
                    pos, dt, self.dataset.features, self.bounds
                )

        # 2. Move drones
        if self._base_mode in ("static", "static_nstep", "pizza_replan", "pizza_fixed"):
            self._move_drones_static(dt)
        elif self._base_mode == "tree_nstep":
            self._move_drones_tree_nstep(dt)
        elif self._base_mode == "random_n8":
            self._move_drones_random(dt)
        elif self._lookahead_steps >= 2:
            self._move_drones_greedy_nstep(dt)
        else:
            self._move_drones_greedy(dt)

        # 3–4. Sense and update belief. The original ordering is preserved
        # unless the isolated sensing-order fix is explicitly selected.
        if self.audit_fixes.sensing_order:
            detected_cells = self._detect_victims()
            self._update_observations(detected_cells=detected_cells)
        else:
            self._update_observations()
            self._detect_victims()

        # 4b. Task-finished survivors sensed their final waypoint this step;
        # now they become search-inactive (hold): no movement, no new
        # exposures from the next step on.
        if self._pending_search_inactive_uav_ids:
            self._search_inactive_uav_ids.extend(
                item
                for item in self._pending_search_inactive_uav_ids
                if item not in self._search_inactive_uav_ids
            )
            self._pending_search_inactive_uav_ids = []

        # 5. Apply decay
        self.dynamic_heatmap.apply_decay(self.sim_time)

        # 6. Check budget
        if self.total_distance >= self.budget:
            self._done = True

        # 7. In static mode, stop if all drones have finished their paths
        if self._base_mode in ("static", "static_nstep", "pizza_replan", "pizza_fixed") and not self._done:
            all_done = True
            for i in range(self.num_drones):
                if not self._is_active(i):
                    continue
                if i < len(self.static_paths):
                    path = self.static_paths[i]
                    if not path.is_empty and path.length > 0 and self._path_progress[i] < path.length:
                        all_done = False
                        break
            if all_done:
                self._done = True

        # 8. Mid-mission failure trigger. Fires at the end of the first
        # timestep whose completed fleet movement distance crosses B_fail:
        # that step's moves and sensing are already complete (and include the
        # soon-failed UAV). Fires at most once per run.
        if (
            not self._midflight_triggered
            and self._midflight_failure_fraction is not None
            and self.total_distance >= self._midflight_B_fail
        ):
            self._trigger_midflight_failure()

        return not self._done

    # ------------------------------------------------------------------
    # Drone movement strategies
    # ------------------------------------------------------------------

    def _score_position(self, nwx: float, nwy: float, current_map: np.ndarray) -> float:
        """Score a candidate position using hybrid new/revisit weighting."""
        visible = self._get_visible_cells_world(nwx, nwy)
        new_cells = visible - self.globally_observed_cells
        score = sum(current_map[r, c] for r, c in new_cells)
        if self.revisit_weight > 0:
            old_cells = visible - new_cells
            if old_cells:
                score += self.revisit_weight * sum(
                    current_map[r, c] for r, c in old_cells
                )
        return score

    def _is_active(self, index: int) -> bool:
        """UAV is not lost (failed)."""
        return self._failed_uav_ids is None or index not in self._failed_uav_ids

    def _is_sensing(self, index: int) -> bool:
        """UAV senses: active and not search-inactive (finished its task)."""
        return self._is_active(index) and index not in self._search_inactive_uav_ids

    def _sync_observed_mask(self, cells: set[tuple[int, int]]) -> None:
        """Record freshly observed cells in the cached mask mirror."""
        for row, col in cells:
            if self._observed_mask_np[row, col] == 0.0:
                self._observed_mask_np[row, col] = 1.0
                self._observed_mask_count += 1

    def _rebuild_observed_mask(self) -> None:
        """Rebuild the mask mirror from the observed set (out-of-band edits)."""
        observed_mask = np.zeros(self.heatmap_shape, dtype=np.float64)
        for row, col in self.globally_observed_cells:
            observed_mask[row, col] = 1.0
        self._observed_mask_np = observed_mask
        self._observed_mask_count = len(self.globally_observed_cells)

    def _update_observations(self, detected_cells: set[tuple[int, int]] | None = None):
        """Record the current FOV union and update the Bayesian map."""
        observed_cells = set()
        per_drone_cells = []
        for i, pos in enumerate(self.drone_positions):
            if not self._is_sensing(i):
                continue
            cells = self._get_visible_cells_world(pos[0], pos[1])
            per_drone_cells.append(cells)
            observed_cells.update(cells)
            self.globally_observed_cells.update(cells)
            self._sync_observed_mask(cells)

        if self.audit_fixes.union_observations_per_timestep:
            self.dynamic_heatmap.update(
                observed_cells,
                current_time=self.sim_time,
                detected_cells=detected_cells,
            )
        else:
            for cells in per_drone_cells:
                self.dynamic_heatmap.update(
                    cells,
                    current_time=self.sim_time,
                    detected_cells=detected_cells,
                )

    def _detect_victims(self) -> set[tuple[int, int]]:
        """Apply temporal Bernoulli detection and return detected cells."""
        detected_cells: set[tuple[int, int]] = set()
        for vi in range(self.num_victims):
            if self.victims_found[vi]:
                continue
            vpos = self.victim_positions[vi]
            for di, dpos in enumerate(self.drone_positions):
                if not self._is_sensing(di):
                    continue
                dist = np.sqrt((dpos[0] - vpos.x) ** 2 + (dpos[1] - vpos.y) ** 2)
                if dist <= self.detection_radius and self.rng.random() < self.detection_probability:
                    self.victims_found[vi] = True
                    detected_cells.add(self._world_to_grid(vpos.x, vpos.y))
                    break
        return detected_cells

    def _get_valid_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Return in-bounds, in-radius neighbor grid positions."""
        height, width = self.heatmap_shape
        max_radius_sq = self.max_radius ** 2
        valid = []
        for dr, dc in self._neighbor_offsets:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            nwx, nwy = self._grid_to_world(nr, nc)
            if (nwx - self.center_x) ** 2 + (nwy - self.center_y) ** 2 < max_radius_sq:
                valid.append((nr, nc))
        return valid

    def _get_planning_map(self) -> np.ndarray:
        """Return the map visible to the online planner."""
        if self._base_mode in ("online_static", "online_static_nstep"):
            return self.dynamic_heatmap.get_prior()
        return self.dynamic_heatmap.get_current_map()

    def _begin_replan(self):
        self._number_of_replans += 1
        self._current_replan_id += 1
        self._reserved_target_cells: set[tuple[int, int]] = set()

    def _register_target(
        self,
        drone_index: int,
        row: int,
        col: int,
        score: float,
        start_row: int,
        start_col: int,
        *,
        reserve: bool | None = None,
    ):
        row, col = int(row), int(col)
        start_row, start_col = int(start_row), int(start_col)
        target = (row, col)
        if reserve is None:
            reserve = bool(self.audit_fixes.target_reservation)
        if reserve and target in self._reserved_target_cells:
            self._number_of_target_conflicts += 1
        if reserve:
            self._reserved_target_cells.add(target)
        if not any(action[0] == drone_index for action in self._first_actions):
            self._first_actions.append((int(drone_index), row, col))
        trace_row = {
            "replan_id": self._current_replan_id,
            "sim_time": float(self.sim_time),
            "drone": int(drone_index),
            "start_row": start_row,
            "start_col": start_col,
            "target_row": row,
            "target_col": col,
            "score": float(score),
        }
        self._target_trace.append(trace_row.copy())
        self._planning_trace.append({
            "planner": self.planning_mode,
            **trace_row,
        })

    def _argmax_candidates(
        self,
        candidates: list[tuple[int, int]],
        current_map: np.ndarray | None = None,
        use_value_map: bool = False,
    ) -> tuple[tuple[int, int], float]:
        """Best candidate by score; returns (position, score)."""
        best_score = -1.0
        best_pos = candidates[0] if candidates else None
        for nr, nc in candidates:
            if use_value_map:
                score = self._cached_V[nr, nc]
            else:
                nwx, nwy = self._grid_to_world(nr, nc)
                score = self._score_position(nwx, nwy, current_map)
            if score > best_score:
                best_score = score
                best_pos = (nr, nc)
        return best_pos, best_score

    def _move_drones_greedy(self, dt: float):
        """Greedy 1-step movement on the selected online planning map."""
        current_map = self._get_planning_map()
        max_step = self.drone_speed * dt
        needs_reeval = any(
            self._is_active(i) and self._drone_targets[i] is None
            for i in range(self.num_drones)
        )

        if needs_reeval:
            planning_start = _time.perf_counter()
            self._begin_replan()
            for i in range(self.num_drones):
                if not self._is_active(i):
                    continue
                wx, wy = self.drone_positions[i]
                if self._drone_targets[i] is None:
                    row, col = self._world_to_grid(wx, wy)
                    valid = self._get_valid_neighbors(row, col)
                    if self.audit_fixes.target_reservation:
                        available = [
                            (nr, nc) for nr, nc in valid
                            if (nr, nc) not in self._reserved_target_cells
                        ]
                        unreserved_pos, _ = self._argmax_candidates(
                            valid, current_map
                        )
                        if unreserved_pos in self._reserved_target_cells:
                            self._reservation_blocked += 1
                    else:
                        available = valid
                    candidates = available or valid

                    best_pos, best_score = self._argmax_candidates(
                        candidates, current_map
                    )
                    if best_score <= 1e-12 and candidates:
                        best_pos = candidates[self.rng.integers(len(candidates))]
                    if (
                        self.audit_fixes.target_reservation
                        and len(available) < len(valid)
                        and best_pos != unreserved_pos
                    ):
                        self._number_of_target_conflicts += 1
                    self._register_target(
                        i, best_pos[0], best_pos[1], best_score, row, col
                    )
                    self._drone_targets[i] = self._grid_to_world(*best_pos)
            self._planner_wall_time += _time.perf_counter() - planning_start

        for i in range(self.num_drones):
            if not self._is_active(i):
                continue
            wx, wy = self.drone_positions[i]
            tx, ty = self._drone_targets[i]
            dx_t = tx - wx
            dy_t = ty - wy
            dist_to_target = np.sqrt(dx_t ** 2 + dy_t ** 2)

            if dist_to_target <= max_step:
                new_world = (tx, ty)
                move_dist = dist_to_target
                self._drone_targets[i] = None
            else:
                scale = max_step / dist_to_target
                new_world = (wx + scale * dx_t, wy + scale * dy_t)
                move_dist = max_step

            self.total_distance += move_dist
            self.drone_positions[i] = new_world
            self.drone_paths[i].append(new_world)

    def _make_policy_rng(self, uav_id: int) -> np.random.Generator:
        """Return the independent reproducible RNG stream for one UAV.

        The stream depends only on ``(policy_seed, uav_id)`` and never on fleet
        size or on the other UAVs: removing another UAV does not shift the
        stream of a survivor, and there is no shared sequential consumption
        across UAVs (each stream is created directly from its own derived
        SeedSequence).
        """
        child = np.random.SeedSequence([
            int(self.policy_seed) % (2**32),
            int(uav_id) % (2**32),
        ])
        derived_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        return np.random.default_rng(derived_seed)

    def _move_drones_random(self, dt: float) -> None:
        """Uninformed, memoryless, uncoordinated N8 uniform random walk.

        At every timestep each UAV draws a direction uniformly from the
        physically valid N8 actions of its current grid cell using its own
        independent policy RNG:

            A_t^i = {a in N8(x_t^i) : a valid}
            a_t^i ~ Uniform(A_t^i)

        The policy consults no probability map, no belief, no observed mask, no
        global observed set, no revisit score, no other UAV targets, no cycle
        reservations and no own history. Physical validity / safety rules are
        unchanged (_get_valid_neighbors: in-bounds and within the search
        radius). There is no target deconfliction: two UAVs may select the same
        target during the same cycle and both decisions are kept.
        """
        max_step = self.drone_speed * dt
        self._begin_replan()
        for i in range(self.num_drones):
            if not self._is_active(i):
                continue
            wx, wy = self.drone_positions[i]
            row, col = self._world_to_grid(wx, wy)
            valid = self._get_valid_neighbors(row, col)
            if not valid:
                # Degenerate domain without any in-radius N8 neighbour: hold the
                # current cell. Physical validity is preserved and the standard
                # budget/termination rules still apply.
                self._register_target(
                    i, row, col, float("nan"), row, col, reserve=False
                )
                self._drone_targets[i] = (wx, wy)
                continue
            pick = int(self._policy_rngs[i].integers(len(valid)))
            nr, nc = valid[pick]
            self._register_target(i, nr, nc, float("nan"), row, col, reserve=False)
            self._drone_targets[i] = self._grid_to_world(nr, nc)

        for i in range(self.num_drones):
            if not self._is_active(i):
                continue
            wx, wy = self.drone_positions[i]
            tx, ty = self._drone_targets[i]
            dx_t = tx - wx
            dy_t = ty - wy
            dist_to_target = np.sqrt(dx_t ** 2 + dy_t ** 2)
            if dist_to_target <= max_step:
                new_world = (tx, ty)
                move_dist = dist_to_target
            else:
                scale = max_step / dist_to_target
                new_world = (wx + scale * dx_t, wy + scale * dy_t)
                move_dist = max_step
            self.total_distance += move_dist
            self.drone_positions[i] = new_world
            self.drone_paths[i].append(new_world)

    @classmethod
    def _parse_planning_mode(cls, mode: str) -> tuple[str, int]:
        """Parse planning mode string into (base_mode, lookahead_steps)."""
        if mode == "static":
            return "static", 0
        if mode == "dynamic":
            return "dynamic", 1
        if mode == "online_static":
            return "online_static", 1
        if mode == "pizza_replan":
            return "pizza_replan", 1
        if mode == "pizza_fixed":
            return "pizza_fixed", 1
        if mode == "random_uniform_n8_uncoordinated":
            return "random_n8", 1
        m = cls._NSTEP_RE.match(mode)
        if m:
            base, n = m.group(1), int(m.group(2))
            if n < 2:
                raise ValueError(f"N-step lookahead requires N >= 2, got {n}")
            if base == "static":
                return "static_nstep", n
            if base == "tree":
                return "tree_nstep", n
            if base == "online_static":
                return "online_static_nstep", n
            return "dynamic_nstep", n
        raise ValueError(
            f"Unknown planning_mode '{mode}'. "
            "Choose 'static', 'dynamic', 'online_static', 'static_Nstep', "
            "'online_static_Nstep', or 'dynamic_Nstep' "
            "(e.g. online_static_3step, dynamic_3step)"
        )

    def _build_fov_kernel(self) -> np.ndarray:
        """Build circular FOV kernel for convolution-based score maps."""
        rx = self.detection_radius_cells_x
        ry = self.detection_radius_cells_y
        Y, X = np.ogrid[-ry:ry + 1, -rx:rx + 1]
        dist = np.sqrt((X * self.dx) ** 2 + (Y * self.dy) ** 2)
        return (dist <= self.detection_radius).astype(np.float64)

    def _compute_score_map(self, current_map: np.ndarray) -> np.ndarray:
        """Compute per-cell score via convolution (vectorised _score_position).

        This is equivalent to calling _score_position for every cell in the grid
        but runs as a single numpy convolution — orders of magnitude faster.
        """
        # Observed mask. The cached mirror avoids a Python pass over the whole
        # observed set at every replan; a length mismatch detects out-of-band
        # set mutations (tests manipulate the set directly) and rebuilds it.
        if len(self.globally_observed_cells) != self._observed_mask_count:
            self._rebuild_observed_mask()
        observed_mask = self._observed_mask_np

        # Score from new (unobserved) cells
        new_values = current_map * (1.0 - observed_mask)
        score_map = convolve(new_values, self._fov_kernel, mode='constant', cval=0.0)

        if self.revisit_weight > 0:
            old_values = current_map * observed_mask
            score_map += self.revisit_weight * convolve(
                old_values, self._fov_kernel, mode='constant', cval=0.0
            )
        if self.audit_fixes.bellman_domain_mask:
            score_map = score_map.astype(np.float64, copy=False)
            score_map[~self._valid_domain_mask] = -np.inf
        return score_map

    def _maximum_neighbor_filter(self, values: np.ndarray) -> np.ndarray:
        """Maximum over the eight moves, excluding the center cell."""
        padded = np.pad(values, 1, mode="constant", constant_values=-np.inf)
        neighbor_values = []
        for dr, dc in self._neighbor_offsets:
            row_start = 1 + dr
            col_start = 1 + dc
            neighbor_values.append(
                padded[
                    row_start:row_start + values.shape[0],
                    col_start:col_start + values.shape[1],
                ]
            )
        return np.maximum.reduce(neighbor_values)

    def _bellman_implied_path(self, row: int, col: int) -> list[tuple[int, int]]:
        """Follow Bellman's cached value greedily for diagnostics."""
        path: list[tuple[int, int]] = []
        current = (int(row), int(col))
        for _ in range(self._lookahead_steps):
            neighbors = self._get_valid_neighbors(*current)
            if not neighbors:
                break
            next_position = max(neighbors, key=lambda pos: self._cached_V[pos])
            current = (int(next_position[0]), int(next_position[1]))
            path.append(current)
        return path

    def _move_drones_greedy_nstep(self, dt: float):
        """N-step lookahead via Bellman value iteration."""
        max_step = self.drone_speed * dt
        needs_reeval = any(
            self._is_active(i) and self._drone_targets[i] is None
            for i in range(self.num_drones)
        )

        if needs_reeval:
            planning_start = _time.perf_counter()
            self._begin_replan()
            current_map = self._get_planning_map()
            score_map = self._compute_score_map(current_map)
            self._cached_V = score_map
            for _ in range(self._lookahead_steps - 1):
                if self.audit_fixes.bellman_no_center:
                    next_value = self._maximum_neighbor_filter(self._cached_V)
                else:
                    next_value = maximum_filter(
                        self._cached_V,
                        size=3,
                        mode="constant",
                        cval=-np.inf if self.audit_fixes.bellman_domain_mask else 0.0,
                    )
                self._cached_V = score_map + next_value

            shadow_elapsed = 0.0
            for i in range(self.num_drones):
                if not self._is_active(i):
                    continue
                if self._drone_targets[i] is not None:
                    continue
                wx, wy = self.drone_positions[i]
                row, col = self._world_to_grid(wx, wy)
                neighbors = self._get_valid_neighbors(row, col)
                if self.audit_fixes.target_reservation:
                    available = [
                        (nr, nc) for nr, nc in neighbors
                        if (nr, nc) not in self._reserved_target_cells
                    ]
                    unreserved_pos, _ = self._argmax_candidates(
                        neighbors, use_value_map=True
                    )
                    if unreserved_pos in self._reserved_target_cells:
                        self._reservation_blocked += 1
                else:
                    available = neighbors
                candidates = available or neighbors

                best_pos, best_score = self._argmax_candidates(
                    candidates, use_value_map=True
                )
                if best_score <= 1e-12 and candidates:
                    best_pos = candidates[self.rng.integers(len(candidates))]
                if (
                    self.audit_fixes.target_reservation
                    and len(available) < len(neighbors)
                    and best_pos != unreserved_pos
                ):
                    self._number_of_target_conflicts += 1
                self._register_target(
                    i, best_pos[0], best_pos[1], best_score, row, col
                )
                if self.audit_fixes.bellman_no_center and self.shadow_diagnostics:
                    shadow_start = _time.perf_counter()
                    shadow = self._shadow_tree_diagnostics(
                        row,
                        col,
                        current_map,
                        best_pos,
                    )
                    shadow_elapsed += _time.perf_counter() - shadow_start
                    self._planning_trace[-1].update(shadow)
                    self._target_trace[-1].update(shadow)
                else:
                    implied_path = self._bellman_implied_path(row, col)
                    self._planning_trace[-1]["surrogate_exact_reward"] = (
                        self._exact_path_reward(implied_path, current_map)
                    )
                    self._planning_trace[-1]["surrogate_path"] = implied_path
                self._drone_targets[i] = self._grid_to_world(*best_pos)
            self._planner_wall_time += (
                _time.perf_counter() - planning_start - shadow_elapsed
            )
            self._shadow_wall_time += shadow_elapsed

        for i in range(self.num_drones):
            if not self._is_active(i):
                continue
            wx, wy = self.drone_positions[i]
            tx, ty = self._drone_targets[i]
            dx_t = tx - wx
            dy_t = ty - wy
            dist_to_target = np.sqrt(dx_t ** 2 + dy_t ** 2)

            if dist_to_target <= max_step:
                new_world = (tx, ty)
                move_dist = dist_to_target
                self._drone_targets[i] = None
            else:
                scale = max_step / dist_to_target
                new_world = (wx + scale * dx_t, wy + scale * dy_t)
                move_dist = max_step

            self.total_distance += move_dist
            self.drone_positions[i] = new_world
            self.drone_paths[i].append(new_world)

    def _exact_path_reward(
        self,
        path: list[tuple[int, int]],
        current_map: np.ndarray,
    ) -> float:
        """Evaluate a candidate by the union of its FOV cells."""
        counted: set[tuple[int, int]] = set()
        reward = 0.0
        for row, col in path:
            visible = self._visible_cells_from_grid(row, col)
            new_cells = visible - self.globally_observed_cells - counted
            old_cells = (visible & self.globally_observed_cells) - counted
            reward += sum(current_map[r, c] for r, c in new_cells)
            reward += self.revisit_weight * sum(current_map[r, c] for r, c in old_cells)
            counted.update(visible)
        return float(reward)

    def _visible_cells_from_grid(self, row: int, col: int) -> set[tuple[int, int]]:
        return self._get_visible_cells_world(*self._grid_to_world(row, col))
    def _enumerate_exact_tree_paths(
        self,
        row: int,
        col: int,
        current_map: np.ndarray,
        first_action: tuple[int, int] | None = None,
    ) -> list[tuple[list[tuple[int, int]], float]]:
        """Enumerate valid N-step paths without clipping or implicit waits."""
        paths: list[tuple[list[tuple[int, int]], float]] = []

        def visit(
            current_row: int,
            current_col: int,
            remaining: int,
            path: list[tuple[int, int]],
        ):
            if remaining == 0:
                paths.append((path, self._exact_path_reward(path, current_map)))
                return
            neighbors = self._get_valid_neighbors(current_row, current_col)
            if not neighbors:
                return
            if not path and first_action is not None:
                neighbors = [position for position in neighbors if position == first_action]
            for next_row, next_col in neighbors:
                visit(
                    int(next_row),
                    int(next_col),
                    remaining - 1,
                    [*path, (int(next_row), int(next_col))],
                )

        visit(int(row), int(col), self._lookahead_steps, [])
        return paths

    def _shadow_tree_diagnostics(
        self,
        row: int,
        col: int,
        current_map: np.ndarray,
        bellman_action: tuple[int, int],
    ) -> dict:
        """Evaluate Bellman's first action against the exact tree at one state."""
        candidates = self._enumerate_exact_tree_paths(row, col, current_map)
        if not candidates:
            return {
                "action_agreement": False,
                "exact_optimal_reward": 0.0,
                "exact_bellman_action_reward": 0.0,
                "exact_regret": 0.0,
            }
        optimal_path, optimal_reward = max(candidates, key=lambda item: item[1])
        constrained = self._enumerate_exact_tree_paths(
            row,
            col,
            current_map,
            first_action=(int(bellman_action[0]), int(bellman_action[1])),
        )
        constrained_reward = max((reward for _, reward in constrained), default=0.0)
        return {
            "action_agreement": bool(optimal_path[0] == tuple(map(int, bellman_action))),
            "exact_optimal_reward": float(optimal_reward),
            "exact_bellman_action_reward": float(constrained_reward),
            "exact_regret": float(optimal_reward - constrained_reward),
        }

    def _move_drones_tree_nstep(self, dt: float):
        """Exhaustive exact FOV-union tree search with receding horizon."""
        max_step = self.drone_speed * dt
        needs_reeval = any(
            self._is_active(i) and self._drone_targets[i] is None
            for i in range(self.num_drones)
        )

        if needs_reeval:
            planning_start = _time.perf_counter()
            self._begin_replan()
            current_map = self._get_planning_map()
            for i in range(self.num_drones):
                if not self._is_active(i):
                    continue
                if self._drone_targets[i] is not None:
                    continue
                wx, wy = self.drone_positions[i]
                row, col = self._world_to_grid(wx, wy)
                candidates = self._enumerate_exact_tree_paths(row, col, current_map)
                if self.audit_fixes.target_reservation:
                    available = [
                        candidate for candidate in candidates
                        if candidate[0][0] not in self._reserved_target_cells
                    ]
                    self._number_of_target_conflicts += len(candidates) - len(available)
                    candidates = available or candidates
                if not candidates:
                    continue
                best_path, best_reward = max(candidates, key=lambda item: item[1])
                nr, nc = best_path[0]
                self._register_target(i, nr, nc, best_reward, row, col)
                self._planning_trace[-1]["exact_reward"] = best_reward
                self._planning_trace[-1]["candidate_path"] = best_path
                self._drone_targets[i] = self._grid_to_world(nr, nc)
            self._planner_wall_time += _time.perf_counter() - planning_start
            if all(target is None for target in self._drone_targets):
                self._done = True
                return

        for i in range(self.num_drones):
            if not self._is_active(i):
                continue
            target = self._drone_targets[i]
            if target is None:
                continue
            wx, wy = self.drone_positions[i]
            tx, ty = target
            dx_t = tx - wx
            dy_t = ty - wy
            dist_to_target = np.sqrt(dx_t ** 2 + dy_t ** 2)

            if dist_to_target <= max_step:
                new_world = (tx, ty)
                move_dist = dist_to_target
                self._drone_targets[i] = None
            else:
                scale = max_step / dist_to_target
                new_world = (wx + scale * dx_t, wy + scale * dy_t)
                move_dist = max_step

            self.total_distance += move_dist
            self.drone_positions[i] = new_world
            self.drone_paths[i].append(new_world)

    def _move_drones_static(self, dt: float):
        """Follow pre-computed static paths by interpolating along LineStrings.

        Hold policy (pizza_replan only): a survivor whose assigned task is
        finished is search-inactive - it does not move, does not append
        positions and does not sense. Search-inactive drones are skipped before
        any progress/movement either way.
        """
        step_distance = self.drone_speed * dt

        for i in range(self.num_drones):
            if not self._is_active(i) or i in self._search_inactive_uav_ids:
                continue
            if i >= len(self.static_paths):
                continue
            path = self.static_paths[i]
            if path.is_empty or path.length == 0:
                continue

            wx, wy = self.drone_positions[i]
            self._path_progress[i] += step_distance

            # Clamp to path length — drone stops at end of path
            progress = min(self._path_progress[i], path.length)
            pt = path.interpolate(progress)
            new_world = (pt.x, pt.y)

            move_dist = np.sqrt((new_world[0] - wx) ** 2 + (new_world[1] - wy) ** 2)
            self.total_distance += move_dist
            self.drone_positions[i] = new_world
            self.drone_paths[i].append(new_world)

            if (
                self._base_mode == "pizza_replan"
                and self._midflight_triggered
                and self._path_progress[i] >= path.length - 1e-9
                and i not in self._search_inactive_uav_ids
                and i not in self._pending_search_inactive_uav_ids
            ):
                # Finish -> final observation (this timestep) -> hold: the
                # promotion to search-inactive happens right after sensing.
                self._pending_search_inactive_uav_ids.append(i)

    # ------------------------------------------------------------------
    # C7 mid-mission failure and recovery
    # ------------------------------------------------------------------

    def _trigger_midflight_failure(self):
        """Mark the configured UAVs as failed; pizza_replan repartitions now.

        Immediate-awareness semantics: the survivor targets are cleared once so
        every online planner performs ONE explicit replanning right after the
        event (the dynamic replan at the same cadence Pizza receives its
        repartition); afterwards the normal target-consumption cadence resumes
        (needs_reeval considers only active UAVs).
        """
        self._failed_uav_ids = list(self._midflight_failed_uav_ids)
        self._midflight_triggered = True
        for i in range(self.num_drones):
            if self._is_active(i):
                self._drone_targets[i] = None
        self._midflight_info = {
            "failure_fraction": self._midflight_failure_fraction,
            "B_fail": float(self._midflight_B_fail),
            "failure_distance": float(self.total_distance),
            "failure_time": float(self.sim_time),
            "failed_uav_ids": list(self._failed_uav_ids),
            "trigger_step": len(self.drone_paths[0]) - 1,
            "pizza_replan": None,
        }
        if self._base_mode == "pizza_replan":
            self._replan_pizza_midflight()

    def activate_midflight_failure(self):
        """Public: activate the configured failure immediately.

        Used by the C7B state-matched branches (failure at the end of the
        captured prefix) and by the pre-step activation test. A second call
        after the trigger has fired is a no-op.
        """
        if not self._midflight_triggered:
            self._trigger_midflight_failure()

    def _replan_pizza_midflight(self):
        """Fault-triggered geometric repartitioning (pizza_replan only).

        Pure geometry: no probability source is ever read. Uses only survivor
        positions, the valid-domain mask, the observed set, the FOV geometry,
        and the static-path machinery.
        """
        from sarenv.analytics.midflight import (
            build_transit_path,
            min_entry_cost_assignment,
            route_length,
            split_route,
            trim_covered_tails,
            waypoint_has_unseen_cell,
        )
        from sarenv.analytics.paths import generate_pizza_zigzag_path

        survivors = [i for i in range(self.num_drones) if self._is_active(i)]
        K = len(survivors)
        diag = {
            "generator": (
                "generate_pizza_zigzag_path(overlap=0.0, "
                "path_point_spacing_m=10.0, border_gap_m=0.0)"
            ),
            "num_sectors": K,
            "survivor_ids": list(survivors),
            "search_inactive_uav_ids": [],
            "task_splits": 0,
            "empty_sectors": 0,
            "prefix_removed_total": 0,
            "suffix_removed_total": 0,
            "deadhead_per_drone": {},
            "deadhead_total": 0.0,
            "deadhead_fraction_remaining_budget": None,
            "route_length_after_replan": {},
            "assignment_cost": None,
            "orientations_per_sector": {},
            "assignment": None,
        }
        self._midflight_info["pizza_replan"] = diag
        self._search_inactive_uav_ids = []
        if K == 0:
            self.static_paths = [LineString() for _ in range(self.num_drones)]
            self._path_progress = [0.0] * self.num_drones
            return
        raw_paths = generate_pizza_zigzag_path(
            center_x=self.center_x,
            center_y=self.center_y,
            max_radius=self.max_radius,
            num_drones=K,
            fov_deg=self.fov_deg,
            altitude=self.altitude,
            overlap=0.0,
            path_point_spacing_m=10.0,
            border_gap_m=0.0,
        )
        tasks = []
        for path in raw_paths:
            coords = [tuple(float(v) for v in point) for point in path.coords]
            useful = [
                waypoint_has_unseen_cell(
                    wx,
                    wy,
                    visible_fn=self._get_visible_cells_world,
                    valid_mask=self._valid_domain_mask,
                    observed=self.globally_observed_cells,
                    shape=self.heatmap_shape,
                )
                for wx, wy in coords
            ]
            remaining, prefix_removed, suffix_removed = trim_covered_tails(
                coords, useful
            )
            diag["prefix_removed_total"] += int(prefix_removed)
            diag["suffix_removed_total"] += int(suffix_removed)
            if not remaining:
                diag["empty_sectors"] += 1
                continue
            tasks.append(remaining)

        # Mandatory splitting: give every survivor a task. A split is performed
        # ONLY if it yields two children >= 20 m each (a parent route needs at
        # least ~40 m); otherwise the route is kept whole and the leftover
        # survivors become search-inactive (hold policy). Children below the
        # 20 m threshold are therefore never produced.
        min_split_length = 20.0
        while tasks and len(tasks) < K:
            longest_index = max(
                range(len(tasks)), key=lambda j: route_length(tasks[j])
            )
            halves = split_route(tasks[longest_index], 2)
            if len(halves) < 2:
                break
            if min(route_length(h) for h in halves) < min_split_length:
                break
            before = len(tasks)
            tasks[longest_index:longest_index + 1] = halves
            if len(tasks) <= before:
                break
            diag["task_splits"] += 1

        new_static = [LineString() for _ in range(self.num_drones)]
        if tasks:
            # Rectangular assignment: the surivor-index -> (task, orientation)
            # mapping also CHOOSES which survivors get a task; the unassigned
            # survivors become search-inactive (hold policy).
            positions_list = [self.drone_positions[i] for i in survivors]
            assignment, cost = min_entry_cost_assignment(positions_list, tasks)
            assigned_survivors = [survivors[idx] for idx in sorted(assignment.keys())]
            search_inactive = [i for i in survivors if i not in assigned_survivors]
            self._search_inactive_uav_ids = list(search_inactive)
            diag["search_inactive_uav_ids"] = list(search_inactive)
            diag["assignment_cost"] = float(cost)
            diag["assignment"] = {
                str(survivors[index]): [int(task_index), orientation]
                for index, (task_index, orientation) in assignment.items()
            }
            deadhead_total = 0.0
            for pos_index, (task_index, orientation) in assignment.items():
                drone_id = survivors[pos_index]
                task_coords = tasks[task_index]
                chosen = (
                    list(task_coords)
                    if orientation == "forward"
                    else list(reversed(task_coords))
                )
                position = self.drone_positions[drone_id]
                entry = chosen[0]
                deadhead = float(
                    np.hypot(entry[0] - position[0], entry[1] - position[1])
                )
                diag["deadhead_per_drone"][int(drone_id)] = deadhead
                diag["route_length_after_replan"][int(drone_id)] = route_length(
                    chosen
                )
                diag["orientations_per_sector"][int(drone_id)] = orientation
                deadhead_total += deadhead
                new_static[drone_id] = build_transit_path(
                    list(position), task_coords, orientation
                )
            diag["deadhead_total"] = float(deadhead_total)
            remaining_budget = float(self.budget - self.total_distance)
            if remaining_budget > 1e-9:
                diag["deadhead_fraction_remaining_budget"] = float(
                    deadhead_total / remaining_budget
                )
        # Failed and search-inactive drones get an empty path (no move).
        self.static_paths = new_static
        self._path_progress = [0.0] * self.num_drones

    def inject_state(self, *, positions, observed_cells, belief_state, time, distance):
        """C7B-only: restore a captured checkpoint on an already-setup sim.

        Precondition: ``setup()`` has run (belief exists). ``positions`` are the
        per-drone world positions at the checkpoint; ``belief_state`` carries the
        EvidenceBelief absolute attributes captured at the checkpoint.
        """
        if len(positions) != self.num_drones:
            raise ValueError(
                f"positions length {len(positions)} != num_drones {self.num_drones}"
            )
        if self.belief_model != "evidence":
            raise NotImplementedError(
                "inject_state requires belief_model='evidence' (C7 uses evidence)"
            )
        self.drone_positions = [tuple(float(v) for v in pos) for pos in positions]
        self.drone_paths = [[pos] for pos in self.drone_positions]
        self.globally_observed_cells = set(
            (int(r), int(c)) for r, c in observed_cells
        )
        self._rebuild_observed_mask()
        self.total_distance = float(distance)
        self.sim_time = float(time)

        belief = self.dynamic_heatmap
        belief.ell_anchor = np.asarray(
            belief_state["ell_anchor"], dtype=np.float64
        ).copy()
        belief.last_evidence_time = np.asarray(
            belief_state["last_evidence_time"], dtype=np.float64
        ).copy()
        belief.observed_mask = np.asarray(
            belief_state["observed_mask"], dtype=bool
        ).copy()
        belief._current_time = float(time)
        belief._num_updates = int(belief_state.get("_num_updates", 0))
        belief._total_cells_observed = int(
            belief_state.get("_total_cells_observed", 0)
        )
        belief.posterior = belief.get_current_map()

        self._done = False
        self._drone_targets = [None] * self.num_drones
        self._cached_V = None
        self._search_inactive_uav_ids = []
        self._pending_search_inactive_uav_ids = []
        if self.static_paths is not None:
            self._path_progress = [0.0] * len(self.static_paths)
        self.snapshots = []
        self._number_of_replans = 0
        self._number_of_target_conflicts = 0
        self._reservation_blocked = 0
        self._first_actions = []
        self._planning_trace = []
        self._target_trace = []
        self._current_replan_id = 0
        self._record_snapshot(store_heatmap=True)

    def _record_snapshot(self, store_heatmap: bool = False):
        """Record current state."""
        current_map = self.dynamic_heatmap.get_current_map()
        # Likelihood: sum of PRIOR probability for all observed cells
        # Using prior (not posterior) because:
        # 1. Victims are sampled from the prior distribution
        # 2. Posterior is artificially inflated by renormalization after observations
        # 3. This measures actual coverage of original probability mass
        likelihood = sum(
            self.dynamic_heatmap.prior[r, c] for r, c in self.globally_observed_cells
        ) if self.globally_observed_cells else 0.0

        snap = SimulationSnapshot(
            time=self.sim_time,
            drone_positions=[p for p in self.drone_positions],
            victim_positions=[(v.x, v.y) for v in self.victim_positions],
            victims_found=list(self.victims_found),
            likelihood_score=likelihood,
            cells_observed=len(self.globally_observed_cells),
            heatmap=current_map if store_heatmap else None,
        )
        self.snapshots.append(snap)

    def run(self, dt: float = 1.0, snapshot_interval: int = 10, heatmap_interval: int = 50,
            visualizer=None, path_update_interval: int = 20) -> SimulationResult:
        """
        Run the full simulation loop.

        Args:
            dt: Timestep in seconds.
            snapshot_interval: Record a snapshot every N steps.
            heatmap_interval: Store the full heatmap every N steps (heavier).
            visualizer: Optional SimulationVisualizer for live rendering.
            path_update_interval: Update drone path lines every N steps.

        Returns:
            SimulationResult with all recorded data.
        """
        self.setup()
        return self.run_from_state(
            dt=dt,
            snapshot_interval=snapshot_interval,
            heatmap_interval=heatmap_interval,
            visualizer=visualizer,
            path_update_interval=path_update_interval
        )

    def run_from_state(self, dt: float = 1.0, snapshot_interval: int = 10, heatmap_interval: int = 50,
                       visualizer=None, path_update_interval: int = 20) -> SimulationResult:
        """
        Run simulation from current state WITHOUT calling setup().
        
        Use this when you've manually configured initial positions/state.
        For normal use, call run() instead.

        Args:
            dt: Timestep in seconds.
            snapshot_interval: Record a snapshot every N steps.
            heatmap_interval: Store the full heatmap every N steps (heavier).
            visualizer: Optional SimulationVisualizer for live rendering.
            path_update_interval: Update drone path lines every N steps.

        Returns:
            SimulationResult with all recorded data.
        """
        if visualizer is not None:
            visualizer.setup(self.dynamic_heatmap.get_current_map())
            visualizer.update(self.snapshots[0])

        wall_start = _time.perf_counter()
        step_count = 0

        while self.step(dt):
            step_count += 1
            if step_count % snapshot_interval == 0:
                store_hm = (step_count % heatmap_interval == 0)
                self._record_snapshot(store_heatmap=store_hm)
                if visualizer is not None:
                    visualizer.update(self.snapshots[-1])
                    if step_count % path_update_interval == 0:
                        visualizer.update_paths(self.drone_paths)

        # Final snapshot always includes heatmap
        self._record_snapshot(store_heatmap=True)
        if visualizer is not None:
            visualizer.update(self.snapshots[-1])
            visualizer.update_paths(self.drone_paths)

        wall_time = _time.perf_counter() - wall_start
        victims_found_count = sum(self.victims_found)
        final_likelihood = self.snapshots[-1].likelihood_score
        audit_record = make_audit_record(
            dataset=self.dataset_id,
            seed=self.seed,
            planner=self.planning_mode,
            implementation_version=self.implementation_version,
            num_drones=self.num_drones,
            num_victims=self.num_victims,
            budget=self.budget,
            pd=self.detection_probability,
            tau=self.decay_tau,
            w=self.revisit_weight,
            likelihood=final_likelihood,
            cells_observed=len(self.globally_observed_cells),
            victims_found=victims_found_count,
            total_time_sim=self.sim_time,
            wall_time=wall_time,
            planner_wall_time=self._planner_wall_time,
            number_of_replans=self._number_of_replans,
            number_of_target_conflicts=self._number_of_target_conflicts,
            reservation_blocked=self._reservation_blocked,
            first_actions=self._first_actions,
            fix_profile=self.fix_profile,
            belief_model=self.belief_model,
        ).to_dict()

        log.info(
            f"Simulation complete: {step_count} steps, "
            f"{self.sim_time:.0f}s sim time, "
            f"{self.total_distance:.0f}m traveled, "
            f"{victims_found_count}/{self.num_victims} victims found, "
            f"wall time {wall_time:.1f}s"
        )

        return SimulationResult(
            params=self._get_params_dict(),
            snapshots=self.snapshots,
            final_drone_paths=self.drone_paths,
            total_time=self.sim_time,
            total_distance=self.total_distance,
            victims_found_count=victims_found_count,
            final_likelihood=final_likelihood,
            final_heatmap=self.dynamic_heatmap.get_current_map(),
            wall_time=wall_time,
            planner_wall_time=self._planner_wall_time,
            shadow_wall_time=self._shadow_wall_time,
            number_of_replans=self._number_of_replans,
            number_of_target_conflicts=self._number_of_target_conflicts,
            reservation_blocked=self._reservation_blocked,
            first_action_sequence_hash=stable_hash(self._first_actions),
            planning_trace=self._planning_trace,
            target_trace=self._target_trace,
            audit_record=audit_record,
        )

    def _get_params_dict(self) -> dict:
        return {
            "dataset_center": self.dataset.center_point,
            "dataset_size": self.dataset.size,
            "environment_type": self.dataset.environment_type,
            "environment_climate": self.dataset.environment_climate,
            "num_drones": self.num_drones,
            "num_victims": self.num_victims,
            "fov_deg": self.fov_deg,
            "altitude": self.altitude,
            "detection_probability": self.detection_probability,
            "decay_tau": self.decay_tau,
            "victim_speed": self.victim_speed,
            "victim_model": type(self._victim_model).__name__,
            "budget": self.budget,
            "init_strategy": self.init_strategy,
            "planning_mode": self.planning_mode,
            "revisit_weight": self.revisit_weight,
            "seed": self.seed,
            "detection_radius": self.detection_radius,
            "dataset_id": self.dataset_id,
            "implementation_version": self.implementation_version,
            "fix_profile": self.fix_profile,
            "belief_model": self.belief_model,
            "shadow_diagnostics": self.shadow_diagnostics,
        }

    def cell_exposure_counts(self) -> np.ndarray:
        """Total independent sensor exposures per cell (sum over drone×timestep)."""
        counts = np.zeros(self.heatmap_shape, dtype=np.int64)
        for path in self.drone_paths:
            for wx, wy in path:
                for r, c in self._get_visible_cells_world(wx, wy):
                    counts[r, c] += 1
        return counts
