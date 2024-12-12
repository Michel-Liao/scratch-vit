import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv(
    "/home/michel/repos/scratch_vit/gd_grd_2/results_20241212_081159.csv"
)

# Create a pivot table for the heatmap
pivot_data = data.pivot(
    index="learning_rate", columns="patch_size", values="test_accuracy"
)

# Create figure and axis with specified size
plt.figure(figsize=(10, 6))

# Create heatmap
sns.heatmap(
    pivot_data,
    annot=True,  # Show values in cells
    fmt=".3f",  # Format to 3 decimal places
    cmap="viridis",  # Use viridis colormap (better for continuous data)
    cbar_kws={"label": "Test Accuracy"},
    vmin=0.7,  # Set minimum value for better color contrast
    vmax=1.0,
)  # Set maximum value for better color contrast

# Customize the plot
plt.title("Test Accuracy by Learning Rate and Patch Size")
plt.xlabel("Patch Size")
plt.ylabel("Learning Rate")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig("accuracy_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
