#!/usr/bin/env python3
"""Combined (a) violin + (b) forest plot, single columnwidth for IEEE RAL.

Panel (a): likelihood violin distributions across five planning modes.
Panel (b): Cliff's delta forest plot for all ten pairwise comparisons.
Statistical significance from paired Wilcoxon signed-rank tests (p capped at 1e-16).

Output: output/fig_pairwise_combined.pdf
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=0.85)
plt.rcParams.update({
    "font.family": "serif",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "font.weight": "normal",
})

# ── Shared constants ─────────────────────────────────────────────────────────
FIVE_MODES = ["pizza", "static", "dynamic", "static_3step", "dynamic_3step"]
MODE_LABELS = ["pizza", "static", "dynamic", "static\n3step", "dynamic\n3step"]
COLORS = {
    "pizza": "#2ecc71",
    "static": "#95a5a6",
    "dynamic": "#e67e22",
    "static_3step": "#3498db",
    "dynamic_3step": "#e74c3c",
}
COLOR_SIG = "#e74c3c"
COLOR_NS = "#95a5a6"

# ── Load data ────────────────────────────────────────────────────────────────
df_main = pd.read_csv(DATA_DIR / "study_main_20260531_095441.csv")
df_cliff = pd.read_csv(DATA_DIR / "cliff_delta_pairwise_likelihood.csv")

df = df_main[df_main["planning_mode"].isin(FIVE_MODES)].copy()
mode_order = df.groupby("planning_mode")["likelihood"].mean().sort_values(ascending=False).index.tolist()
palette_violin = [COLORS[m] for m in mode_order]
# Force Pandas to use that strict order
df["planning_mode"] = pd.Categorical(df["planning_mode"], categories=mode_order, ordered=True)


# ── Create figure ────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(3.45, 4.0),
    gridspec_kw={"height_ratios": [1.6, 2.0], "hspace": 0.0},
)

# Panel (a): Margen izquierdo pequeño para aprovechar todo el ancho
ax_a.set_position([0.07, 0.53, 0.87, 0.43]) 

# Panel (b): Margen izquierdo amplio (0.32) para que quepan los textos de "mode_a vs mode_b"
ax_b.set_position([0.32, 0.0, 0.63, 0.38])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL (a): Violin plot
# ══════════════════════════════════════════════════════════════════════════════
ax_a.set_title("(a) Likelihood by Planning Mode", loc="left", pad=2)

vp = sns.violinplot(
    data=df, x="planning_mode", y="likelihood", order=mode_order,
    palette=palette_violin, ax=ax_a, inner="quartile", cut=0, linewidth=0.8,
    density_norm="count", common_norm=False,
)
ax_a.set_xlabel("")
ax_a.set_ylabel("Likelihood $L$")
ax_a.set_xticklabels([MODE_LABELS[FIVE_MODES.index(m)] for m in mode_order])
for lab in ax_a.get_xticklabels():
    lab.set_weight("bold")
ax_a.tick_params(axis='x', pad=1.5)

# Significance brackets
ymax = df["likelihood"].max()
xpos = {m: i for i, m in enumerate(mode_order)}
y_offset = ymax * 0.03
bracket_y = ymax + y_offset * 1.2

pairs_to_annotate = [
    ("dynamic_3step", "pizza"),
    ("dynamic_3step", "static"),
    ("dynamic_3step", "dynamic"),
    ("dynamic_3step", "static_3step"),
    ("static_3step", "static"),
]
pair_info = {}
for _, row in df_cliff.iterrows():
    a, b = row["mode_a"], row["mode_b"]
    pair_info[tuple(sorted([a, b]))] = {
        "p": row["wilcoxon_p"], "d": row["cliff_delta"], "sig": row["sig_holm"],
    }

# Sort by bracket width (draw widest first = topmost)
bracket_specs = []
for a, b in pairs_to_annotate:
    key = tuple(sorted([a, b]))
    if key in pair_info:
        bracket_specs.append((abs(xpos[a] - xpos[b]), a, b, pair_info[key]))

bracket_specs.sort(key=lambda t: t[0], reverse=False)

for _, a, b, info in bracket_specs:
    if a not in xpos or b not in xpos:
        continue
    x1, x2 = xpos[a], xpos[b]
    y = bracket_y
    ax_a.plot([x1, x1, x2, x2], [y, y + y_offset * 0.3, y + y_offset * 0.3, y],
              "k-", linewidth=0.8, color="#333333")
    p = info["p"]
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    label = f"$\\delta={info['d']:.2f}$ {stars}"
    ax_a.text((x1 + x2) / 2, y + y_offset * 0.1, label, ha="center", va="bottom", fontsize=6,
              color="#333333", fontweight="bold" if stars != "ns" else "normal")
    bracket_y += y_offset * 1.5

ax_a.set_ylim(top=bracket_y + y_offset)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.grid(True, alpha=0.25, axis="y")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL (b): Forest plot (Cliff's δ)
# ══════════════════════════════════════════════════════════════════════════════
ax_b.set_title("(b) Pairwise Cliff's $\\delta$", loc="left", pad=2)

comparisons = []
for _, row in df_cliff.iterrows():
    ma = row["mode_a"]
    mb = row["mode_b"]
    delta = row["cliff_delta"]
    ci_lo = row["cliff_ci_lo"]
    ci_hi = row["cliff_ci_hi"]
    
    # Si delta es negativo, se invierte la pareja y todos sus estadísticos
    if delta < 0:
        label = f"{mb} vs {ma}"
        delta = -delta
        # Al multiplicar por -1, el límite inferior pasa a ser el superior y viceversa
        ci_lo_new = -ci_hi
        ci_hi_new = -ci_lo
        ci_lo, ci_hi = ci_lo_new, ci_hi_new
    else:
        label = f"{ma} vs {mb}"

    comparisons.append({
        "label": label,
        "delta": delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "sig": row["sig_holm"],
        "p": row["wilcoxon_p"],
    })

comparisons.sort(key=lambda c: c["delta"])  # ascending δ
n = len(comparisons)
#y_pos = np.arange(n)
y_pos = np.array([i * 0.5 for i in range(n)])
labels = [c["label"] for c in comparisons]
d_vals = np.array([c["delta"] for c in comparisons])
ci_lo = np.array([c["ci_lo"] for c in comparisons])
ci_hi = np.array([c["ci_hi"] for c in comparisons])
colors_b = [COLOR_SIG if c["sig"] else COLOR_NS for c in comparisons]

# Draw error bars
for i in range(n):
    ax_b.errorbar(d_vals[i], y_pos[i],
                  xerr=[[d_vals[i] - ci_lo[i]], [ci_hi[i] - d_vals[i]]],
                  fmt="o", capsize=2.5, capthick=0.8, elinewidth=0.8,
                  color=colors_b[i], markersize=3, zorder=3)

ax_b.axvline(x=0, color="black", linewidth=0.6, linestyle="--", alpha=0.5, zorder=1)

# Effect-size regions
ax_b.axvspan(-0.147, 0.147, alpha=0.05, color="gray", zorder=0)
ax_b.axvspan(-0.33, -0.147, alpha=0.05, color="#3498db", zorder=0)
ax_b.axvspan(0.147, 0.33, alpha=0.05, color="#3498db", zorder=0)
ax_b.axvspan(-0.474, -0.33, alpha=0.05, color="#f39c12", zorder=0)
ax_b.axvspan(0.33, 0.474, alpha=0.05, color="#f39c12", zorder=0)
ax_b.axvspan(-1.05, -0.474, alpha=0.05, color="#e67e22", zorder=0)
ax_b.axvspan(0.474, 1.05, alpha=0.05, color="#e67e22", zorder=0)

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(labels, fontsize=6, family="serif")
ax_b.set_ylim(min(y_pos) - 0.3, max(y_pos) + 0.3)

ax_b.set_xlabel("Cliff's $\\delta$")
ax_b.set_xlim(-0.6, 1.1)

ax_b.xaxis.set_ticks_position('bottom')
ax_b.tick_params(axis='x', which='both', bottom=True, length=3, width=0.8,
                 colors='#cccccc', labelcolor='black')

# Significance markers
for i, (y, d, c) in enumerate(zip(y_pos, d_vals, comparisons)):
    p = c["p"]
    mkr = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    x_pos = c["ci_lo"] - 0.04
    ax_b.text(x_pos, y, mkr, ha="right", va="center", fontsize=6,
              fontweight="bold", color=colors_b[i])

# Compact legend
legend_el = [
    Patch(facecolor="gray", alpha=0.12, label="negligible"),
    Patch(facecolor="#3498db", alpha=0.12, label="small"),
    Patch(facecolor="#f39c12", alpha=0.12, label="medium"),
    Patch(facecolor="#e67e22", alpha=0.12, label="large"),
]
ax_b.legend(handles=legend_el, loc="upper left", fontsize=5.5,
            framealpha=0.85, ncol=1, borderpad=0.3, handlelength=1.0)

ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.grid(True, alpha=0.2, axis="x")

outpath = OUTPUT_DIR / "fig_pairwise_combined.pdf"
plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()
print(f"✓ Saved: {outpath}")