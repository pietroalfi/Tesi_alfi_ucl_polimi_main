import pickle
import numpy as np
import pandas as pd

# Path to the input pickle file (modify as needed)
INPUT_PKL_PATH = r"/home/pietro/ray_results/dqn_without_wandb/DQN_mkt-data-v0_589d6_00000_0_2025-02-26_22-47-59/training_data.pkl"
OUTPUT_PKL_PATH = "Log_return_data.pkl"  # Output file

def preprocess_pkl(input_file_path, output_file_path, time_horizon="10S"):
    """
    Processes a raw pickle file, computes log returns at the specified time horizon,
    and saves the processed data into a new pickle file.

    Parameters:
    - input_file_path (str): Path to the input .pkl file containing raw data.
    - output_file_path (str): Path to save the processed .pkl file.
    - time_horizon (str): Time interval for return calculation (e.g., "10S" for 10 seconds, "1T" for 1 minute).
    """
    # Load raw data
    data = []
    with open(input_file_path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break

    # Flatten the structure if needed
    flattened_data = []
    for row in data:
        if isinstance(row, list):
            for item in row:
                if isinstance(item, dict):
                    flattened_data.append(item)
        elif isinstance(row, dict):
            flattened_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)

    # Convert timestamps to datetime format
    df["datetime"] = pd.to_datetime(df["current_time"], unit="ns")

    # Resample data to specified time horizon
    df_resampled = df.set_index("datetime").resample(time_horizon).first().dropna().reset_index()

    # Compute log returns
    df_processed = pd.DataFrame({
        "datetime": df_resampled["datetime"],
        "log_ret_fundamental": np.log(df_resampled["fundamental_value"] / df_resampled["fundamental_value"].shift(1)),
        "log_ret_mid": np.log(df_resampled["mid"] / df_resampled["mid"].shift(1))
    }).dropna()

    # Save processed data as a pickle file
    with open(output_file_path, "wb") as f:
        pickle.dump(df_processed, f)

    print(f"Processed data saved to {output_file_path}")

if __name__ == "__main__":
    preprocess_pkl(INPUT_PKL_PATH, OUTPUT_PKL_PATH, time_horizon="10S")
