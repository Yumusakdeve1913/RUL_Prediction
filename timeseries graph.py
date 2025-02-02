import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r"/Users/emekcapoglu/Desktop/EmekUtku_Capoglu/Datasets/Final Database.csv"
data = pd.read_csv(file_path)

# Select the first 250 samples
data = data.iloc[:250]

# Columns to plot
columns_to_plot = [
    'Discharge Time (s)',
    'Decrement 3.6-3.4V (s)',
    'Max. Voltage Dischar. (V)',
    'Min. Voltage Charg. (V)',
    'Time at 4.15V (s)',
    'Time constant current (s)',
    'Charging time (s)',
    'Total time (s)'
]

# Define the grid layout
n_cols = 4  # Number of columns
n_rows = -(-len(columns_to_plot) // n_cols)  # Ceiling division to get rows

# Create subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 10))
axes = axes.flatten()  # Flatten the axes array

# Loop through each column and plot
for i, column in enumerate(columns_to_plot):
    axes[i].plot(data['Cycle_Index'], data[column], color='b', linewidth=2)
    axes[i].set_xlabel('Cycle_Index', fontsize=14)
    axes[i].set_ylabel(column, fontsize=14)
    axes[i].tick_params(axis='both', labelsize=12)
    axes[i].grid(True)

    # Add numbered labels (1), (2), ...
    axes[i].text(0.05, 0.85, f"({i+1})", transform=axes[i].transAxes, fontsize=16, fontweight='bold')

# Remove any unused subplots
for j in range(len(columns_to_plot), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()
