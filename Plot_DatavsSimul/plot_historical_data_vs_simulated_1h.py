import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load simulation data DQN_fx-basic-v0_19d27_00000_0_2025-02-11_23-11-11 # EXE PNL
simulation_file_path = "/g100/home/userexternal/palfi000/ray_results/dqn_with_wandb_and_callbacks/DQN_intc-basic-v0_eed90_00000_0_2025-03-27_15-01-09/training_data.csv"  
df = pd.read_csv(simulation_file_path)
# Convert current_time from nanoseconds to datetime
df['current_time'] = pd.to_datetime(df['current_time'], unit='ns')

# Identify distinct episodes by detecting resets in time
df['time_diff'] = df['current_time'].diff().dt.total_seconds()
df['episode'] = (df['time_diff'] < 0).cumsum()

# Get unique episodes for color mapping
unique_episodes = df['episode'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_episodes)))

# Load historical data
historical_file_path = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Merged_data/INTC_merged_mid_price_train.pkl"  # Update with actual path
historical_df = pd.read_pickle(historical_file_path)

# Reset index and rename columns
historical_df = historical_df.reset_index().rename(columns={'index': 'current_time', 'FundamentalValue': 'mid_price'})

# Convert current_time to datetime format
historical_df['current_time'] = pd.to_datetime(historical_df['Date'])

# Determine simulation time range
simulation_start = df['current_time'].min()
simulation_end = df['current_time'].max()

# Extract only historical data matching the simulation time range
simulation_hour = df['current_time'].dt.hour.unique()[0]
historical_hour_data = historical_df[historical_df['current_time'].dt.hour == simulation_hour]
historical_hour_data = historical_hour_data[
    (historical_hour_data['current_time'] >= simulation_start) & 
    (historical_hour_data['current_time'] <= simulation_end)
]

# Plot simulation data with colors for each episode
plt.figure(figsize=(12, 6))
for i, episode in enumerate(unique_episodes):
    mask = df['episode'] == episode
    plt.plot(df['current_time'][mask], df['mid_price'][mask], label=f'Episode {episode}', color=colors[i])

# Plot historical data as a black dashed line
plt.plot(historical_hour_data['current_time'], historical_hour_data['mid_price'],
         linestyle='dashed', color='black', linewidth=1.5, label='Historical Data')

plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.title('Price Trend Over Simulation Time Window (With Historical Data)')
plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)
plt.xlim([simulation_start, simulation_end])  # Set x-axis limits to match simulation time window

# Save the plot as a PNG file
plt.savefig("simulation_vs_historical_1h.png", dpi=300, bbox_inches='tight')
