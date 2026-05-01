import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the estimation results
results_path = 'Signal_processing/results/ZINB_estimates/zinb_estimates_windows_size2000.csv'
df = pd.read_csv(results_path)

# Remove all thetas bigger than 5
df = df[df['theta'] <= 5]

# Print the mean and median of theta, pi and, mu
print(f"Theta (θ) - Mean: {df['theta'].mean():.4f}, Median: {df['theta'].median():.4f}")
print(f"Pi (π) - Mean: {df['pi'].mean():.4f}, Median: {df['pi'].median():.4f}")
print(f"Mu (μ) - Mean: {df['mu'].mean():.4f}, Median: {df['mu'].median():.4f}")

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df['theta'], df['pi'], alpha=0.6, s=50)
plt.xlabel('Theta (θ)', fontsize=12)
plt.ylabel('Pi (π)', fontsize=12)
plt.title('Pi vs Theta Values from ZINB Estimation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_dir = 'Signal_processing/results/ZINB_estimates/plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'pi_vs_theta.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to {output_dir}/pi_vs_theta.png")
print(f"\nData summary:")
print(f"Number of data points: {len(df)}")
print(f"Theta range: [{df['theta'].min():.4f}, {df['theta'].max():.4f}]")
print(f"Pi range: [{df['pi'].min():.4f}, {df['pi'].max():.4f}]")