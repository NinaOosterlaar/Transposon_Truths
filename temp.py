import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Fake data to mimic the figure
# -----------------------------
np.random.seed(7)

x1 = np.arange(0, 250)      # Non-essential
y1 = np.clip(np.random.normal(10, 3, len(x1)), 0, 22)

x2 = np.arange(250, 400)    # Semi-essential
y2 = np.clip(np.random.normal(4, 1.0, len(x2)), 0, 8)

x3 = np.arange(400, 500)    # Essential
y3 = np.clip(np.random.exponential(0.05, len(x3)), 0, 0.4)

x4 = np.arange(500, 600)    # Non-essential
y4 = np.clip(np.random.normal(10, 3, len(x4)), 0, 22)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10.6, 4.85))

# White background
ax.set_facecolor("white")

# Colors
green = "#2ecc71"
dark_yellow = "#c49a00"
orange = "#e67e22"

# Thin vertical lines
ax.vlines(x1, 0, y1, color=green, alpha=0.75, linewidth=1.1)
ax.vlines(x2, 0, y2, color=dark_yellow, alpha=0.85, linewidth=1.1)
ax.vlines(x3, 0, y3, color=orange, alpha=0.9, linewidth=1.0)
ax.vlines(x4, 0, y4, color=green, alpha=0.75, linewidth=1.1)

# Vertical dashed separators
for xpos in [250, 400, 500]:
    ax.axvline(x=xpos, color="gray", linestyle="--", linewidth=1.5, alpha=0.9)

# Labels
ax.set_xlabel("Position")
ax.set_ylabel("Count")

# Limits and ticks
ax.set_xlim(0, 600)
ax.set_ylim(0, 22.5)
ax.set_xticks(np.arange(0, 601, 50))

# Light grid
ax.grid(axis="y", color="gray", alpha=0.15)

# Region labels
ax.text(125, 20.2, "Non-essential",
        ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#9bd3ae", edgecolor="gray", alpha=0.95))

ax.text(325, 20.2, "Semi-essential",
        ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#d8c27a", edgecolor="gray", alpha=0.95))

ax.text(450, 20.2, "Essential",
        ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#e6c6a5", edgecolor="gray", alpha=0.95))

ax.text(550, 20.2, "Non-essential",
        ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#9bd3ae", edgecolor="gray", alpha=0.95))

plt.tight_layout()
plt.show()