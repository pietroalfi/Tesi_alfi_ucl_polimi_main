import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Path to the input pickle file (modify as needed)
INPUT_PKL_PATH = r"/home/pietro/ray_results/dqn_without_wandb/DQN_mkt-data-v0_6aead_00000_0_2025-03-17_10-30-09/training_data.pkl"
OUTPUT_PKL_PATH = "Log_return_distribution.pkl"  # Output file

def plot_log_return_distribution(input_pkl_path, output_png_path, num_bins=50):
    """
    Loads a preprocessed pickle file, creates a log return distribution plot,
    and saves it as a PNG file.

    Parameters:
    - input_pkl_path (str): Path to the input .pkl file containing log return data.
    - output_png_path (str): Path to save the output PNG file.
    - num_bins (int): Number of bins for the histogram.
    """
    # Load preprocessed log return data from pickle
    with open(input_pkl_path, "rb") as f:
        df_log_returns = pickle.load(f)

    # Estimate mean (mu) and standard deviation (sigma) for both log return series
    mu_fundamental, sigma_fundamental = np.mean(df_log_returns["log_ret_fundamental"]), np.std(df_log_returns["log_ret_fundamental"])
    mu_mid, sigma_mid = np.mean(df_log_returns["log_ret_mid"]), np.std(df_log_returns["log_ret_mid"])

    # Define bins
    bins = np.linspace(df_log_returns["log_ret_fundamental"].min(), df_log_returns["log_ret_fundamental"].max(), num_bins)

    # Compute histograms ensuring no empty bins
    hist_fundamental, bin_edges = np.histogram(df_log_returns["log_ret_fundamental"], bins=bins, density=False)
    hist_mid, _ = np.histogram(df_log_returns["log_ret_mid"], bins=bins, density=False)

    # **Fill Empty Bins with Small Values to Avoid Gaps**
    hist_fundamental = np.where(hist_fundamental == 0, 1, hist_fundamental)
    hist_mid = np.where(hist_mid == 0, 1, hist_mid)

    # Normalize histogram counts to ensure they sum to dataset size
    hist_fundamental_normalized = hist_fundamental * (len(df_log_returns) / hist_fundamental.sum())
    hist_mid_normalized = hist_mid * (len(df_log_returns) / hist_mid.sum())

    # Create the histogram plot
    plt.figure(figsize=(10, 6))

    # Plot step histograms for clear visualization
    plt.step(bin_edges[:-1], hist_fundamental_normalized, where="mid", color="blue", label="Historical Data (Normalized)", linewidth=2)
    plt.step(bin_edges[:-1], hist_mid_normalized, where="mid", color="orange", label="Simulated Data (Normalized)", linewidth=2)

    # Overlay normal distributions (scaled correctly)
    x_vals = np.linspace(df_log_returns["log_ret_fundamental"].min(), df_log_returns["log_ret_fundamental"].max(), 1000)
    scaling_factor = max(hist_fundamental_normalized) / max(norm.pdf(x_vals, mu_fundamental, sigma_fundamental))

    plt.plot(x_vals, scaling_factor * norm.pdf(x_vals, mu_fundamental, sigma_fundamental),
             linestyle="dashed", color="blue", label=f"N(μ=0, σ={sigma_fundamental:.1e})")
    plt.plot(x_vals, scaling_factor * norm.pdf(x_vals, mu_mid, sigma_mid),
             linestyle="dashed", color="orange", label=f"N(μ=0, σ={sigma_mid:.1e})")

    # Formatting for clarity
    plt.xlabel("Log Returns")
    plt.ylabel("Frequency (Normalized)")
    plt.legend()
    plt.title("Log Return Distribution (No Gaps)")
    plt.grid()

    # Save the plot as a PNG file
    plt.savefig(output_png_path)
    plt.close()

    print(f"Plot saved as {output_png_path}")

# Example usage:
# plot_log_return_distribution("path/to/input.pkl", "path/to/output.png")
if __name__ == "__main__":
    plot_log_return_distribution(INPUT_PKL_PATH, OUTPUT_PKL_PATH)

