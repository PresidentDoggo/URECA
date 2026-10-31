import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load and clean the data
fileName = input("Enter filename without .csv: ")
df = pd.read_csv("csv/" + fileName + ".csv")

# Drop columns that have 0 counts across the entire kernel to save space
df = df.loc[:, (df != 0).any(axis=0)]
instructions = df.columns

# Set up the figure style for academic presentation
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# Visualization 1: Heatmap
# ==========================================
plt.figure(figsize=(14, 6))
# Transpose the dataframe so time (bins) is on the X-axis and instructions on the Y-axis
sns.heatmap(df.T, cmap="YlGnBu", cbar_kws={'label': 'Instruction Count'})
plt.title('SASS Instruction Density per Execution Bin (Heatmap)', fontsize=14, pad=15)
plt.xlabel('Execution Bin (Chronological)', fontsize=12)
plt.ylabel('SASS Instruction', fontsize=12)
plt.tight_layout()
plt.savefig("png/" + fileName + ".png", dpi=300)
plt.show()

# ==========================================
# Visualization 2: Stacked Area Chart
# ==========================================
plt.figure(figsize=(14, 6))
plt.stackplot(df.index, df.T, labels=instructions, alpha=0.8, colormap='tab20')
plt.title('Workload Composition over Kernel Execution (Stacked Area)', fontsize=14, pad=15)
plt.xlabel('Execution Bin (Chronological)', fontsize=12)
plt.ylabel('Instruction Count per Bin', fontsize=12)
# Place legend outside the plot so it doesn't cover data
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Instructions")
plt.margins(x=0, y=0)
plt.tight_layout()
plt.savefig('stacked_area.png', dpi=300)
plt.show()

# ==========================================
# Visualization 3: 3D Surface Plot
# ==========================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for X (Bins) and Y (Instructions)
X, Y = np.meshgrid(range(df.shape[0]), range(df.shape[1]))
Z = df.T.values

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Instruction Count')

ax.set_title('3D Surface Plot of Instruction Counts', fontsize=14, pad=15)
ax.set_xlabel('Execution Bin')
ax.set_ylabel('Instruction Index')
ax.set_zlabel('Count')
# Set Y-ticks to match instruction names
ax.set_yticks(range(len(instructions)))
ax.set_yticklabels(instructions, rotation=-45, ha='left')

plt.tight_layout()
plt.savefig('surface_plot.png', dpi=300)
plt.show()