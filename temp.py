import pandas as pd
# Calculate mean of mean_density from specific files
file1 = "Data_exploration/results/densities/centromere/combined_All_Boolean_True_bin_10000_absolute/ALL_combined_centromere_density_Boolean_True_bin_10000_absolute.csv"
df1 = pd.read_csv(file1)
mean1 = df1['mean_density'].mean()

file2 = "Data_exploration/results/densities/nucleosome_new/combined_All_Boolean_True/ALL_combined_Boolean_True_nucleosome_density.csv"
df2 = pd.read_csv(file2)
mean2 = df2['mean_density'].mean()

print(f"Centromere file mean of mean_density: {mean1}")
print(f"Nucleosome file mean of mean_density: {mean2}")