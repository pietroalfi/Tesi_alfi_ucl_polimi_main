import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for saving the plot as a file
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to the CSV file
file_path = '/g100/home/userexternal/palfi000/ray_results/dqn_with_wandb_and_callbacks/DQN_intc-basic-v0_204b5_00000_0_2025-03-24_20-28-14/progress.csv'  # Replace with the correct path
#DQN_markets-mm-basic-v0-HF_5465d_00000
# Select the column to analyze
column_name = 'custom_metrics/action_distribution'

# Number of last rows to analyze or specific row index (set one of these)
n_last = 1
specific_index = None  # Set to an integer for a specific row, or None to use n_last

# Load the data
data = pd.read_csv(file_path)

# Ensure the column exists
if column_name not in data.columns:
    raise ValueError(f"Column '{column_name}' not found in the data.")

# Extract the data from the column and handle the string format issue
raw_data = data[column_name].dropna().apply(
    lambda x: np.array([float(i.strip()) for i in x.strip('[]').split(',')])
)

# Determine the rows to analyze
if specific_index is not None:
    if specific_index >= len(raw_data):
        raise IndexError(f"Row index {specific_index} is out of range.")
    rows_to_analyze = raw_data.iloc[specific_index:specific_index+1]
else:
    if len(raw_data) < n_last:
        raise ValueError(f"Not enough data to extract the last {n_last} rows.")
    rows_to_analyze = raw_data.iloc[-n_last:]

# Compute the cumulative sum across the selected rows
cumulative_sum = np.sum(rows_to_analyze.to_list(), axis=0)

# Calculate percentages relative to the total number of actions
total_actions = np.sum(cumulative_sum)
percentages = (cumulative_sum / total_actions) * 100

# Create a bar plot of the action percentages
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(percentages) + 1), percentages, tick_label=[i + 1 for i in range(len(percentages))])
plt.xlabel("Action")
plt.ylabel("Percentage (%)")
plt.title("Percentage Distribution of Total Actions")
plt.grid(axis='y')

# Save the plot as a PNG file
output_file = 'action_distribution_hf_percentage_prova.png'
plt.savefig(output_file)
print(f"Plot saved as {output_file}")
