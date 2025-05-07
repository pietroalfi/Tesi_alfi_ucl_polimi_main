import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend to save the plot as a file
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval

# Specify the path to the CSV file
file_path = '/g100/home/userexternal/palfi000/ray_results/dqn_with_wandb_and_callbacks/DQN_intc-basic-v0_f4ca6_00000_0_2025-04-02_14-06-49/progress.csv'
# Column to analyze
column_name = 'custom_metrics/action_distribution'

# Number of last rows to analyze or specific row index (set one of these)
n_last = 25
specific_index = None  # Set an integer for a specific row, or None to use n_last

# Load the data
data = pd.read_csv(file_path)

# Ensure the column exists
if column_name not in data.columns:
    raise ValueError(f"Column '{column_name}' not found in the data.")

# Convert the string into a list of floats using literal_eval
raw_data = data[column_name].dropna().apply(lambda x: np.array(literal_eval(x)))

# Select the rows to analyze
if specific_index is not None:
    if specific_index >= len(raw_data):
        raise IndexError(f"Row index {specific_index} is out of range.")
    rows_to_analyze = raw_data.iloc[specific_index:specific_index+1]
else:
    if len(raw_data) < n_last:
        raise ValueError(f"Not enough data to extract the last {n_last} rows.")
    rows_to_analyze = raw_data.iloc[-n_last:]

# Compute the cumulative sum of the selected rows
cumulative_sum = np.sum(rows_to_analyze.to_list(), axis=0)

# Consider only the first 4 actions (ignore the last 3)
cumulative_sum = cumulative_sum[:4]

# Calculate the percentage relative to the total number of actions (only the first 4)
total_actions = np.sum(cumulative_sum)
percentages = (cumulative_sum / total_actions) * 100

# Define labels for the actions
action_labels = ["Best Bid & Best Ask", "Best Bid", "Best Ask", "Do Nothing"]

# Create the bar chart for the percentage distribution of actions
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(percentages) + 1), percentages, tick_label=action_labels)
plt.xlabel("Action Type") 
plt.ylabel("Percentage (%)") 
plt.title("Percentage Distribution of Actions")
plt.grid(axis='y')

# Save the plot as a PNG file
output_file = 'action_distribution_hf_att_intc_old_params_lr_2e-4_eps=0.7_config_std(overall 35 it).png'
plt.savefig(output_file)

print(f"Plot saved as {output_file}")
