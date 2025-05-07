import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Define the CSV file path
CSV_PATH = r"/g100/home/userexternal/palfi000/ray_results/dqn_with_wandb_and_callbacks/DQN_intc-basic-v0_204b5_00000_0_2025-03-24_20-28-14/progress.csv"

def parse_array_string(array_str):
    """
    Convert a string representation of an array into a NumPy array.
    Handles lists in the format "[0.1807, 0.1445, ...]"
    """
    try:
        return np.fromstring(array_str.strip("[]"), sep=" ", dtype=np.float32)
    except (SyntaxError, ValueError):
        return np.zeros(15, dtype=np.float32)  # Default to zero array if parsing fails

def parse_sparsity_list(sparsity_str):
    """
    Convert a string representation of a list into a list of integers.
    """
    try:
        return ast.literal_eval(sparsity_str) if sparsity_str else []
    except (SyntaxError, ValueError, TypeError):
        return []

def generate_plots(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    df = df.head(16)  # Limit to the first 16 rows

    # Process Sparsity Columns
    bid_sparsity_flat = sum(df['custom_metrics/Bid Sparsity'].dropna().apply(parse_sparsity_list), [])
    ask_sparsity_flat = sum(df['custom_metrics/Ask Sparsity'].dropna().apply(parse_sparsity_list), [])

    bid_sparsity_counts = {i: bid_sparsity_flat.count(i) for i in range(10)}
    ask_sparsity_counts = {i: ask_sparsity_flat.count(i) for i in range(10)}

    # Process Volume Profile Columns
    bid_volume_profiles = df['custom_metrics/Bid Volume Profile'].dropna().apply(parse_array_string)
    ask_volume_profiles = df['custom_metrics/Ask Volume Profile'].dropna().apply(parse_array_string)

    if not bid_volume_profiles.empty:
        bid_volume_profiles = np.vstack(bid_volume_profiles.to_numpy())
        mean_bid_vol_profile = np.mean(bid_volume_profiles, axis=0)[:15]
    else:
        mean_bid_vol_profile = np.zeros(15)

    if not ask_volume_profiles.empty:
        ask_volume_profiles = np.vstack(ask_volume_profiles.to_numpy())
        mean_ask_vol_profile = np.mean(ask_volume_profiles, axis=0)[:15]
    else:
        mean_ask_vol_profile = np.zeros(15)

    mean_bid_vol_profile *= 100
    mean_ask_vol_profile *= 100

    # Plot 1: Frequency Distribution of Bid Sparsity
    plt.figure(figsize=(8, 5))
    plt.bar(bid_sparsity_counts.keys(), bid_sparsity_counts.values(), color='red', alpha=0.7)
    plt.xlabel("Number of non-empty ticks form best bid")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution of Bid Sparsity")
    plt.xticks(range(10))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("bid_sparsity_distribution_lamb_1e-9.png")
    plt.close()

    # Plot 2: Frequency Distribution of Ask Sparsity
    plt.figure(figsize=(8, 5))
    plt.bar(ask_sparsity_counts.keys(), ask_sparsity_counts.values(), color='blue', alpha=0.7)
    plt.xlabel("Number of non-empty ticks form best bid")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution of Ask Sparsity")
    plt.xticks(range(10))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("ask_sparsity_distribution_lamb_1e-9.png")
    plt.close()

    # Plot 3: Average Bid Volume Profile
    plt.figure(figsize=(8, 5))
    plt.bar(range(15), mean_bid_vol_profile, color='red', alpha=0.7)
    plt.xlabel("Ticks from Best Bid (0 = Best Bid)")
    plt.ylabel("Volume Percentage (%)")
    plt.title("Average Volume Profile - Bid Side")
    plt.xticks(range(15))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("bid_volume_profile_lamb_1e-9.png")
    plt.close()

    # Plot 4: Average Ask Volume Profile
    plt.figure(figsize=(8, 5))
    plt.bar(range(15), mean_ask_vol_profile, color='blue', alpha=0.7)
    plt.xlabel("Ticks from Best Ask (0 = Best Ask)")
    plt.ylabel("Volume Percentage (%)")
    plt.title("Average Volume Profile - Ask Side")
    plt.xticks(range(15))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("ask_volume_profile_lamb_1e-9.png")
    plt.close()

    # Process Executed Volume Columns
    bid_executed_column = [col for col in df.columns if "Bid Executed Volume" in col]
    ask_executed_column = [col for col in df.columns if "Ask Executed Volume" in col]

    if bid_executed_column:
        bid_executed_vectors = df[bid_executed_column[0]].dropna().apply(parse_array_string)
        if not bid_executed_vectors.empty:
            bid_executed_matrix = np.vstack(bid_executed_vectors.to_numpy())
            mean_bid_executed_vector = np.mean(bid_executed_matrix, axis=0)[:15]
        else:
            mean_bid_executed_vector = np.zeros(15)
        mean_bid_executed_vector *= 100

        plt.figure(figsize=(8, 5))
        plt.bar(range(15), mean_bid_executed_vector, color='red', alpha=0.7)
        plt.xlabel("Ticks from Best Bid (0 = Best Bid)")
        plt.ylabel("Volume Percentage (%)")
        plt.title("Average Bid Executed Volume Profile")
        plt.xticks(range(15))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig("bid_executed_volume_profile_lamb_1e-9.png")
        plt.close()

    if ask_executed_column:
        ask_executed_vectors = df[ask_executed_column[0]].dropna().apply(parse_array_string)
        if not ask_executed_vectors.empty:
            ask_executed_matrix = np.vstack(ask_executed_vectors.to_numpy())
            mean_ask_executed_vector = np.mean(ask_executed_matrix, axis=0)[:15]
        else:
            mean_ask_executed_vector = np.zeros(15)
        mean_ask_executed_vector *= 100

        plt.figure(figsize=(8, 5))
        plt.bar(range(15), mean_ask_executed_vector, color='blue', alpha=0.7)
        plt.xlabel("Ticks from Best Ask (0 = Best Ask)")
        plt.ylabel("Volume Percentage (%)")
        plt.title("Average Ask Executed Volume Profile")
        plt.xticks(range(15))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig("ask_executed_volume_profile_lamb_1e-9.png")
        plt.close()

if __name__ == "__main__":
    generate_plots(CSV_PATH)
