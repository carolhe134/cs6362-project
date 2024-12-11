import pandas as pd
import matplotlib.pyplot as plt

# File paths
file_paths = [
    "training_logs_50k_optim/kl_divergence.csv",
    "training_logs_50k_optim/reconstruction_loss.csv",
    "training_logs_50k_optim/training_loss.csv",
    "training_logs_50k_optim/validation_loss.csv"
]

# Titles for the plots
titles = [
    "KL Divergence",
    "Reconstruction Loss",
    "Training Loss",
    "Validation Loss"
]

# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten to easily iterate over axes

for i, (file_path, title) in enumerate(zip(file_paths, titles)):
    # Read the tab-delimited file
    data = pd.read_csv(file_path, sep='\t')
    
    # Plot the data
    axes[i].plot(data['Epoch'], data['Value'], label=title)
    axes[i].set_title(title)
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("Value")
    axes[i].legend()
    axes[i].grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Save the entire grid as a single PNG file
grid_output_file = "Training_Logs_Grid_50k_optim.png"
plt.savefig(grid_output_file, dpi=300)

print(f"2x2 grid of plots saved as {grid_output_file}!")
