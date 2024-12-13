import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
df = pd.read_csv("vit_grid_search_results/results_20241212_102421.csv")

# Convert learning_rate to numeric and create readable format
df['learning_rate'] = pd.to_numeric(df['learning_rate'])
df['lr_exp'] = df['learning_rate'].apply(lambda x: f'1e{int(np.log10(x))}')

# Create accuracy heatmap
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
pivot_acc = df.pivot(index='patch_size', columns='lr_exp', values='test_accuracy')
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='viridis')
plt.title('Grid Search Results: Test Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Patch Size')

# Create inference time heatmap
plt.subplot(2, 1, 2)
pivot_time = df.pivot(index='patch_size', columns='lr_exp', values='inference_time_ms')
sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='rocket_r')
plt.title('Grid Search Results: Inference Time (ms)')
plt.xlabel('Learning Rate')
plt.ylabel('Patch Size')

plt.tight_layout()
plt.savefig("vit_grid_search_results/grid_search_results.png")
