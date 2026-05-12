import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Arbitrary x-axis
n = 1000
x = np.arange(n)

# Change points
change_points = [170, 330, 500, 660, 820]
boundaries = [0] + change_points + [n]

# Segment distributions
# Differences are visible, but not overly dramatic
segment_params = [
    {"mean": 0.0,  "std": 0.9},
    {"mean": 1.1,  "std": 0.8},
    {"mean": 0.8,  "std": 1.5},
    {"mean": 4.6,  "std": 1.0},
    {"mean": 4.2,  "std": 1.8},
    {"mean": -2.0, "std": 1.0},
]

# Generate signal
signal = np.zeros(n)

for i in range(len(boundaries) - 1):
    start = boundaries[i]
    end = boundaries[i + 1]

    mean = segment_params[i]["mean"]
    std = segment_params[i]["std"]

    signal[start:end] = np.random.normal(
        loc=mean,
        scale=std,
        size=end - start
    )

# Add a few spikes to keep the noisy original feel
spike_indices = np.random.choice(n, size=15, replace=False)
signal[spike_indices] += np.random.normal(loc=0, scale=2.5, size=len(spike_indices))

# Plot
plt.figure(figsize=(12, 5))

plt.plot(x, signal, label="signal", linewidth=1.5)

for cp in change_points:
    plt.axvline(cp, color="red", linewidth=2, alpha=0.8)

plt.title("Time Series with Abrupt Changes")
plt.xlabel("Position")
plt.ylabel("Signal")
plt.legend()
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()