import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load simulation data
simulation_file_path = "/g100/home/userexternal/palfi000/ray_results/dqn_with_wandb_and_callbacks/DQN_intc-basic-v0_eed90_00000_0_2025-03-27_15-01-09/training_data.csv"  
df = pd.read_csv(simulation_file_path)

# Convert current_time from nanoseconds to datetime
df['current_time'] = pd.to_datetime(df['current_time'], unit='ns')

# Identify distinct episodes by detecting resets in time
df['time_diff'] = df['current_time'].diff().dt.total_seconds()
df['episode'] = (df['time_diff'] < 0).cumsum()

# Get unique episodes and select the first
unique_episodes = df['episode'].unique()
first_episode = unique_episodes[0]
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_episodes)))

# Filter only the first episode
mask = df['episode'] == first_episode
first_episode_df = df[mask]
episode_start = first_episode_df['current_time'].min()
episode_end = first_episode_df['current_time'].max()

# Load historical data
historical_file_path = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Merged_data/INTC_merged_mid_price_train.pkl"
historical_df = pd.read_pickle(historical_file_path)

# Reset index and rename columns
historical_df = historical_df.reset_index().rename(columns={'index': 'current_time', 'FundamentalValue': 'mid_price'})

# Convert current_time to datetime format
historical_df['current_time'] = pd.to_datetime(historical_df['Date'])

# Filter historical data by date (same day as the first timestamp in the episode)
episode_date = episode_start.date()
historical_filtered = historical_df[
    historical_df['current_time'].dt.date == episode_date
]

# Plot first episode
plt.figure(figsize=(12, 6))
plt.plot(first_episode_df['current_time'], first_episode_df['mid_price'], label=f'Episode {first_episode}', color=colors[0])

# Plot filtered historical data
plt.plot(historical_filtered['current_time'], historical_filtered['mid_price'],
         linestyle='dashed', color='black', linewidth=1.5, label='Historical Data')

plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.title('Price Trend of First Episode (With Matched Historical Data)')
plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)
plt.xlim([episode_start, episode_end])  # Set x-axis limits to match episode time

# Save the plot
plt.savefig("simulation_vs_historical_first_episode.png", dpi=300, bbox_inches='tight')
