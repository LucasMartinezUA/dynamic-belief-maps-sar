#!/usr/bin/env python3
"""Teaser figure — diagonal split: pizza (hatched orphan) vs dynamic_3step (online replan).

Upper-left  : pizza  (slice lines, hatched orphaned wedge, labels).
Lower-right : dynamic_3step (4 surviving drones replan online).
Clean vector diagonal divider; hero Delta-L at centre; compact legend.

Requires captured cell data + heatmap in data/ (run capture_teaser_data.py first).
Output: output/fig_teaser.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
from matplotlib.lines import Line2D
import teaser_common as T

plt.rcParams.update({"font.family": "serif", "font.size": 8})

# ── Load & crop ───────────────────────────────────────────────────────────────
hm_full, cp, cd, pdz, ddz = T.load_all()
H, W = hm_full.shape
r0, r1, c0, c1, N = T.crop_bounds(H, W)
hm = hm_full[r0:r1, c0:c1]
bg = T.bg_rgb(hm)

mp = T.make_mask(T.crop_cells(cp, r0, r1, c0, c1), N)
md = T.make_mask(T.crop_cells(cd, r0, r1, c0, c1), N)
idp = T.drone_id_map(pdz, r0, r1, c0, c1, N)
idd = T.drone_id_map(ddz, r0, r1, c0, c1, N)
orph = T.wedge_mask(N)

Lp_show = T.compute_likelihood(hm_full, hm, mp, r0, r1, c0, c1)
Ld_show = T.compute_likelihood(hm_full, hm, md, r0, r1, c0, c1)
dL = Ld_show - Lp_show

# ── Composites ────────────────────────────────────────────────────────────────
comp_p = T.composite(bg, mp, idp, per_drone=True, void_mask=orph)
comp_d = T.composite(bg, md, idd, per_drone=True)

rows, cols = np.mgrid[0:N, 0:N]
upper_left = cols < (N - 1 - rows)
final = np.where(upper_left[..., None], comp_p, comp_d)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
ax.imshow(final, aspect="equal", interpolation="bilinear")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_linewidth(0.6)

# clean vector diagonal divider
ax.add_line(Line2D([-0.5, N - 0.5], [N - 0.5, -0.5], color="white",
                   lw=1.6, alpha=0.95, zorder=7))

# ── Pizza slice lines (only in upper-left half) ───────────────────────────────
cx = cy = N / 2
R = N * 0.46

def sect(a):
    rad = np.radians(a)
    return cx + np.cos(rad) * R, cy - np.sin(rad) * R

for a in (72, 144, 216, 288, 360):
    ex, ey = sect(a)
    if (ex < (N - 1 - ey)) or a in (144, 216):
        ls = (0, (3, 3)) if a in (144, 216) else (0, (1, 2))
        ax.add_line(Line2D([cx, ex], [cy, ey], color="white", lw=0.9,
                           linestyle=ls, alpha=0.8, zorder=5))

# hatched orphaned wedge (144°–216°), inner half only
w = Wedge((cx, cy), R * 0.5 + 2, 144, 216, facecolor="none", edgecolor="white",
          hatch="////", lw=0.0, alpha=0.9, zorder=4)
ax.add_patch(w)

# ── Labels ────────────────────────────────────────────────────────────────────
bb = lambda c, a: dict(boxstyle="round,pad=0.22", fc=c, alpha=a)

ax.text(N * 0.04, N * 0.07, "pizza", ha="left", va="top", fontsize=9,
        fontweight="bold", color="white", bbox=bb("black", 0.55))
ax.text(N * 0.04, N * 0.15, f"$L = {Lp_show:.3f}$", ha="left", va="top",
        fontsize=9, color="white", bbox=bb("black", 0.45))
ax.text(N * 0.10, N * 0.45, "orphaned", ha="center", va="center", fontsize=6.5,
        fontweight="bold", color="white", zorder=6, bbox=bb("#b3382c", 0.85))

ax.text(N * 0.96, N * 0.93, "dynamic_3step", ha="right", va="bottom",
        fontsize=9, fontweight="bold", color="white", bbox=bb("black", 0.55))
ax.text(N * 0.96, N * 0.85, f"$L = {Ld_show:.3f}$", ha="right", va="bottom",
        fontsize=9, color="white", bbox=bb("black", 0.45))

# hero Delta-L at centre
ax.text(cx, cy, f"$\\Delta L = +{dL:.3f}$", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#10331f", zorder=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9,
                  ec="#1f7a4d", lw=1.2))

# compact legend (bottom-left, inside pizza half)
items = [
    Patch(fc=T.cmap_bg(0.75)[:3], label="prior"),
    Patch(fc=T.COVERED_SOLID, label="covered"),
    Patch(fc="white", ec="#b3382c", hatch="////", label="orphaned"),
]
ax.legend(handles=items, loc="lower left", frameon=True, fontsize=6,
          handlelength=1.2, borderpad=0.3, labelspacing=0.25,
          framealpha=0.75).set_zorder(9)

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = T.DATA_DIR.parent / "figures" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

outpath = OUTPUT_DIR / "fig_teaser.pdf"
fig.savefig(outpath, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "fig_teaser.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {outpath}")
print(f"  pizza:        L = {Lp_show:.3f}")
print(f"  dynamic_3step: L = {Ld_show:.3f}")
print(f"  Delta L:       +{dL:.3f}")
