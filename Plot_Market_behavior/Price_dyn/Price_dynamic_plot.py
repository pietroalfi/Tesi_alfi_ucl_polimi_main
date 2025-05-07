import pickle
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = r"/home/pietro/ray_results/dqn_without_wandb/DQN_mkt-data-v0_6aead_00000_0_2025-03-17_10-30-09/training_data.pkl"


def plot_fundamental_vs_bid_ask(file_path, output_path="fundamental_vs_bid_ask.png"):
    """
    Reads a pickle file containing financial market data, resamples it to 1-minute intervals,
    and saves a PNG plot of Fundamental Value, Best Bid, and Best Ask over time.

    Parameters:
        file_path (str): Path to the pickle file.
        output_path (str): Path to save the output PNG image.
    """
    # Load the pickle file
    data = []
    with open(file_path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break

    # Extract and flatten the nested structure if needed
    flattened_data = []
    for row in data:
        if isinstance(row, list):
            for item in row:
                if isinstance(item, dict):
                    flattened_data.append(item)
        elif isinstance(row, dict):
            flattened_data.append(row)

    # Convert extracted data into a DataFrame
    df = pd.DataFrame(flattened_data)

    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["current_time"], unit="ns")

    # Sort values by time
    df = df.sort_values("datetime")

    # Resample data to one observation per minute GROUPING FOR 1MIN SLOT
    df_resampled = df.set_index("datetime").resample("1T").first().dropna().reset_index()

    # Plot fundamental value, best bid, and best ask over time with resampled data
    plt.figure(figsize=(12, 6))
    plt.plot(df_resampled["datetime"], df_resampled["fundamental_value"], label="Fundamental Value", linestyle="solid", color="black")
    plt.plot(df_resampled["datetime"], df_resampled["bid"], label="Best Bid", linestyle="dashed", color="blue")
    plt.plot(df_resampled["datetime"], df_resampled["ask"], label="Best Ask", linestyle="dashed", color="red")

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Fundamental Value, Best Bid, and Best Ask over Time (1 Observation per Minute)")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)

    # Save the plot as a PNG file
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved successfully at: {output_path}")

# Example Usage:
# plot_fundamental_vs_bid_ask("path/to/your/training_data.pkl", "output_plot.png")
if __name__ == "__main__":
    plot_fundamental_vs_bid_ask(CSV_PATH, "Price_dyn_base_setting.png")