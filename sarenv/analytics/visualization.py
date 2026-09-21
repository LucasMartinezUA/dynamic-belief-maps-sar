# sarenv/analytics/visualization.py
"""
Real-time and offline visualization for SAR simulations.

Provides:
  - Live matplotlib window (--live flag)
  - MP4 video generation (always)
  - Final state PNG snapshot
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from sarenv.utils.logging_setup import get_logger

log = get_logger()


class SimulationVisualizer:
    """
    Renders simulation state as interactive display and/or MP4 video.

    Layout:
      Left half:  Main search area with heatmap + drones + victims + FOV + paths
      Right top:  Current probability heatmap
      Right middle: Three time-series subplots (likelihood, victims found, cells observed)
    """

    DRONE_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#7f7f7f",
                    "#d62728", "#17becf", "#bcbd22", "#e377c2", "#8c564b"]

    def __init__(self, bounds: tuple, heatmap_shape: tuple, detection_radius: float,
                 num_drones: int, num_victims: int, live: bool = False):
        self.bounds = bounds
        self.heatmap_shape = heatmap_shape
        self.detection_radius = detection_radius
        self.num_drones = num_drones
        self.num_victims = num_victims
        self.live = live

        # State for time-series
        self._times: list[float] = []
        self._likelihoods: list[float] = []
        self._victims_found: list[int] = []
        self._cells_observed: list[int] = []

        # Matplotlib objects
        self.fig = None
        self.ax_map = None
        self.ax_probability = None
        self.ax_likelihood = None
        self.ax_victims = None
        self.ax_cells = None
        self._heatmap_im = None
        self._probability_im = None
        self._drone_scatters = []
        self._victim_scatter_unfound = None
        self._victim_scatter_found = None
        self._fov_circles = []
        self._path_lines = []
        self._initialized = False

    def setup(self, initial_heatmap: np.ndarray):
        """Create the figure and initial plot elements."""
        if self.live:
            matplotlib.use("TkAgg")
            plt.ion()
        else:
            matplotlib.use("Agg")

        self.fig = plt.figure(figsize=(22, 12), dpi=100)
        gs = self.fig.add_gridspec(4, 2, width_ratios=[2, 1], hspace=0.25, wspace=0.3,
                                   left=0.05, right=0.98, top=0.95, bottom=0.05)

        # Left: Main search visualization
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        
        # Right column: Probability map + 3 time-series
        self.ax_probability = self.fig.add_subplot(gs[0, 1])
        self.ax_likelihood = self.fig.add_subplot(gs[1, 1])
        self.ax_victims = self.fig.add_subplot(gs[2, 1])
        self.ax_cells = self.fig.add_subplot(gs[3, 1])

        minx, miny, maxx, maxy = self.bounds
        extent = [minx, maxx, miny, maxy]

        # Main search area heatmap (semi-transparent for overlays)
        self._heatmap_im = self.ax_map.imshow(
            initial_heatmap, cmap="hot", interpolation="nearest",
            origin="lower", extent=extent, alpha=0.6,
        )
        self.ax_map.set_xlim(minx, maxx)
        self.ax_map.set_ylim(miny, maxy)
        self.ax_map.set_aspect('equal', adjustable='box')
        self.ax_map.set_title("SAR Dynamic Simulation — Search Area", fontsize=14, fontweight="bold")
        self.ax_map.set_xlabel("X (m)", fontsize=11)
        self.ax_map.set_ylabel("Y (m)", fontsize=11)

        # Probability heatmap (full opacity, dedicated view)
        self._probability_im = self.ax_probability.imshow(
            initial_heatmap, cmap="hot", interpolation="nearest",
            origin="lower", aspect='auto',
        )
        self.ax_probability.set_title("Probability Map (Dynamic)", fontsize=10, fontweight="bold")
        self.ax_probability.set_xlabel("X (grid)", fontsize=9)
        self.ax_probability.set_ylabel("Y (grid)", fontsize=9)
        plt.colorbar(self._probability_im, ax=self.ax_probability, fraction=0.046, pad=0.04)

        # Drone path lines (one per drone)
        for i in range(self.num_drones):
            color = self.DRONE_COLORS[i % len(self.DRONE_COLORS)]
            line, = self.ax_map.plot([], [], color=color, linewidth=1.2, alpha=0.6)
            self._path_lines.append(line)

        # Drone position markers
        for i in range(self.num_drones):
            color = self.DRONE_COLORS[i % len(self.DRONE_COLORS)]
            sc = self.ax_map.scatter([], [], c=color, s=80, marker="^",
                                     edgecolors="white", linewidths=1.5, zorder=10,
                                     label=f"Drone {i+1}")
            self._drone_scatters.append(sc)

        # FOV circles
        for i in range(self.num_drones):
            color = self.DRONE_COLORS[i % len(self.DRONE_COLORS)]
            circle = Circle((0, 0), self.detection_radius, fill=False,
                           linestyle="--", linewidth=0.8, edgecolor=color, alpha=0.5)
            self.ax_map.add_patch(circle)
            self._fov_circles.append(circle)

        # Victim markers
        self._victim_scatter_unfound = self.ax_map.scatter(
            [], [], c="red", s=120, marker="*", edgecolors="darkred",
            linewidths=0.8, zorder=11, label="Victim (unfound)")
        self._victim_scatter_found = self.ax_map.scatter(
            [], [], c="lime", s=120, marker="*", edgecolors="darkgreen",
            linewidths=0.8, zorder=11, label="Victim (found)")

        self.ax_map.legend(loc="upper right", fontsize=7, ncol=2)

        # Time-series axes
        self.ax_likelihood.set_ylabel("Likelihood")
        self.ax_likelihood.set_title("Cumulative Likelihood", fontsize=9)
        self.ax_victims.set_ylabel("Count")
        self.ax_victims.set_title("Victims Found", fontsize=9)
        self.ax_cells.set_ylabel("Cells")
        self.ax_cells.set_xlabel("Time (s)")
        self.ax_cells.set_title("Cells Observed", fontsize=9)

        # Note: tight_layout is already handled by GridSpec's hspace/wspace
        self._initialized = True

        if self.live:
            self.fig.show()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def update(self, snapshot):
        """Update all plot elements with new simulation snapshot."""
        if not self._initialized:
            return

        # Update both heatmaps if available
        if snapshot.heatmap is not None:
            self._heatmap_im.set_data(snapshot.heatmap)
            self._probability_im.set_data(snapshot.heatmap)
            vmax = snapshot.heatmap.max()
            if vmax > 0:
                self._heatmap_im.set_clim(0, vmax)
                self._probability_im.set_clim(0, vmax)

        # Update drone positions + paths + FOV
        for i in range(min(self.num_drones, len(snapshot.drone_positions))):
            dx, dy = snapshot.drone_positions[i]
            self._drone_scatters[i].set_offsets([[dx, dy]])
            self._fov_circles[i].center = (dx, dy)

        # Update victim positions
        unfound_x, unfound_y = [], []
        found_x, found_y = [], []
        for vi in range(min(self.num_victims, len(snapshot.victim_positions))):
            vx, vy = snapshot.victim_positions[vi]
            if snapshot.victims_found[vi]:
                found_x.append(vx)
                found_y.append(vy)
            else:
                unfound_x.append(vx)
                unfound_y.append(vy)

        self._victim_scatter_unfound.set_offsets(
            np.column_stack([unfound_x, unfound_y]) if unfound_x else np.empty((0, 2))
        )
        self._victim_scatter_found.set_offsets(
            np.column_stack([found_x, found_y]) if found_x else np.empty((0, 2))
        )

        # Time-series
        self._times.append(snapshot.time)
        self._likelihoods.append(snapshot.likelihood_score)
        self._victims_found.append(sum(snapshot.victims_found))
        self._cells_observed.append(snapshot.cells_observed)

        self.ax_likelihood.clear()
        self.ax_likelihood.plot(self._times, self._likelihoods, color="orange", linewidth=1.5)
        self.ax_likelihood.set_ylabel("Likelihood")
        self.ax_likelihood.set_title("Cumulative Likelihood", fontsize=9)

        self.ax_victims.clear()
        self.ax_victims.plot(self._times, self._victims_found, color="green", linewidth=1.5)
        self.ax_victims.set_ylabel("Count")
        self.ax_victims.set_title("Victims Found", fontsize=9)

        self.ax_cells.clear()
        self.ax_cells.plot(self._times, self._cells_observed, color="blue", linewidth=1.5)
        self.ax_cells.set_ylabel("Cells")
        self.ax_cells.set_xlabel("Time (s)")
        self.ax_cells.set_title("Cells Observed", fontsize=9)

        # Title with time
        self.ax_map.set_title(
            f"SAR Dynamic Simulation — t={snapshot.time:.0f}s | "
            f"Found: {sum(snapshot.victims_found)}/{self.num_victims}",
            fontsize=12, fontweight="bold"
        )

        if self.live:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def update_paths(self, drone_paths: list[list[tuple[float, float]]]):
        """Update drone path lines (called less frequently for performance)."""
        if not self._initialized:
            return
        for i in range(min(self.num_drones, len(drone_paths))):
            path = drone_paths[i]
            if len(path) > 1:
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                self._path_lines[i].set_data(xs, ys)

    def save_frame(self) -> np.ndarray | None:
        """Render current figure to numpy RGB array for video encoding."""
        if not self._initialized:
            return None
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        # Convert RGBA to BGR for OpenCV
        return img[:, :, [2, 1, 0]].copy()

    def save_snapshot(self, output_path: str):
        """Save current figure to PNG."""
        if self._initialized:
            self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved snapshot: {output_path}")

    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self._initialized = False


def generate_simulation_video(
    snapshots,
    drone_paths: list[list[tuple[float, float]]],
    bounds: tuple,
    heatmap_shape: tuple,
    detection_radius: float,
    num_drones: int,
    num_victims: int,
    output_path: str,
    fps: int = 24,
    max_frames: int | None = None,
):
    """
    Generate an MP4 video from collected simulation snapshots.

    Args:
        snapshots: List of SimulationSnapshot objects (must have heatmap for first+last).
        drone_paths: Full drone paths for overlay.
        bounds: (minx, miny, maxx, maxy).
        heatmap_shape: (height, width).
        detection_radius: Drone FOV radius in meters.
        num_drones: Number of drones.
        num_victims: Number of victims.
        output_path: Path to write the .mp4 file.
        fps: Frames per second.
        max_frames: Optional max frames to limit video length.
    """
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed. Skipping video generation. Install with: pip install opencv-python")
        return

    if not snapshots:
        log.warning("No snapshots to render.")
        return

    viz = SimulationVisualizer(
        bounds=bounds,
        heatmap_shape=heatmap_shape,
        detection_radius=detection_radius,
        num_drones=num_drones,
        num_victims=num_victims,
        live=False,
    )

    # Use first snapshot's heatmap for setup (or prior)
    initial_hm = snapshots[0].heatmap if snapshots[0].heatmap is not None else np.zeros(heatmap_shape)
    viz.setup(initial_hm)

    # Determine frame dimensions
    test_frame = viz.save_frame()
    if test_frame is None:
        log.error("Failed to render test frame.")
        viz.close()
        return

    h, w = test_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        log.error(f"Failed to open video writer: {output_path}")
        viz.close()
        return

    # Subsample snapshots if too many
    total = len(snapshots)
    if max_frames and total > max_frames:
        indices = np.linspace(0, total - 1, max_frames, dtype=int)
    else:
        indices = range(total)

    log.info(f"Rendering {len(list(indices))} frames to {output_path}...")

    # Build incremental paths
    for frame_i, snap_idx in enumerate(indices):
        snap = snapshots[snap_idx]

        # Build partial paths up to this snapshot's time
        partial_paths = []
        for dp in drone_paths:
            cutoff = min(snap_idx + 1, len(dp))
            partial_paths.append(dp[:cutoff])

        viz.update(snap)
        viz.update_paths(partial_paths)

        frame = viz.save_frame()
        if frame is not None:
            writer.write(frame)

        if frame_i % 100 == 0:
            log.info(f"  Frame {frame_i}/{len(list(indices))}")

    writer.release()
    viz.close()
    log.info(f"Video saved: {output_path}")
