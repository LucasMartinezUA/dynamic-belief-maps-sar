"""Shared helpers for teaser figure variants.

Reads from data/ — capture_teaser_data.py must have been run first
(copies heatmap.npy + generates teaser cell data).
"""
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DATA_DIR = Path(__file__).resolve().parent / ".." / "data"
HEATMAP_PATH = DATA_DIR / "heatmap.npy"

MARGIN = 0.30

DRONE_COLORS = np.array([
    [0.06, 0.55, 0.46],   # drone 0: dark teal
    [0.18, 0.68, 0.40],   # drone 1: green
    [0.40, 0.78, 0.36],   # drone 2: lime
    [0.10, 0.62, 0.62],   # drone 3: teal
])
COVERED_SOLID = np.array([0.13, 0.62, 0.47])
WARM_VOID = np.array([0.55, 0.18, 0.12])

ALPHA_OBS = 0.62
ALPHA_DIM = 0.32

cmap_bg = LinearSegmentedColormap.from_list(
    "bg", ["#f3f4f6", "#cfe0ef", "#7bb0da", "#2f7fc1", "#0b3a6b"])


def load_all():
    """Load heatmap and captured teaser cell data."""
    heatmap_full = np.load(HEATMAP_PATH)
    cp = np.load(DATA_DIR / "teaser_cells_pizza_fail1_seed43.npy")
    cd = np.load(DATA_DIR / "teaser_cells_dyn3_fail1_seed49.npy")
    pdz = np.load(DATA_DIR / "teaser_perdrone_pizza_fail1_seed43.npz")
    ddz = np.load(DATA_DIR / "teaser_perdrone_dyn3_fail1_seed49.npz")
    return heatmap_full, cp, cd, pdz, ddz


def crop_bounds(H, W):
    r0, r1 = int(H * MARGIN), int(H * (1 - MARGIN))
    c0, c1 = int(W * MARGIN), int(W * (1 - MARGIN))
    N = min(r1 - r0, c1 - c0)
    return r0, r0 + N, c0, c0 + N, N


def crop_cells(cells, r0, r1, c0, c1):
    m = (cells[:, 0] >= r0) & (cells[:, 0] < r1) & \
        (cells[:, 1] >= c0) & (cells[:, 1] < c1)
    out = cells[m].copy()
    out[:, 0] -= r0
    out[:, 1] -= c0
    return out


def make_mask(cells, N):
    m = np.zeros((N, N), dtype=bool)
    if len(cells):
        m[cells[:, 0], cells[:, 1]] = True
    return m


def drone_id_map(npz, r0, r1, c0, c1, N):
    did = np.full((N, N), -1, dtype=np.int8)
    for d in range(4):
        k = f"drone_{d}"
        if k in npz:
            cc = crop_cells(npz[k], r0, r1, c0, c1)
            if len(cc):
                did[cc[:, 0], cc[:, 1]] = d
    return did


def bg_rgb(heatmap):
    img = np.log1p(heatmap * 1e6)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return cmap_bg(img)[:, :, :3]


def composite(bg, obs_mask, drone_id, per_drone=True, void_mask=None):
    out = bg.copy()
    if per_drone:
        for d in range(4):
            m = (drone_id == d)
            if m.any():
                out[m] = bg[m] * (1 - ALPHA_OBS) + DRONE_COLORS[d] * ALPHA_OBS
    else:
        out[obs_mask] = bg[obs_mask] * (1 - ALPHA_OBS) + COVERED_SOLID * ALPHA_OBS
    unobs = ~obs_mask
    if void_mask is not None:
        out[unobs & ~void_mask] = bg[unobs & ~void_mask] * (1 - ALPHA_DIM)
        out[unobs & void_mask] = bg[unobs & void_mask] * 0.30 + WARM_VOID * 0.30
    else:
        out[unobs] = bg[unobs] * (1 - ALPHA_DIM)
    return out


def wedge_mask(N, lo=144, hi=216):
    rows, cols = np.mgrid[0:N, 0:N]
    cx = cy = N / 2
    ang = np.degrees(np.arctan2(-(rows - cy), cols - cx)) % 360
    return (ang >= lo) & (ang <= hi)


def compute_likelihood(heatmap_full, heatmap_cropped, mask_cropped, r0, r1, c0, c1):
    """Compute L from cropped region, normalising by full heatmap sum (paper Eq.8)."""
    total = np.sum(heatmap_full)
    # reconstruct full-grid mask from cropped coordinates
    full_mask = np.zeros(heatmap_full.shape, dtype=bool)
    rows_c, cols_c = np.where(mask_cropped)
    full_mask[rows_c + r0, cols_c + c0] = True
    return np.sum(heatmap_full[full_mask]) / total
