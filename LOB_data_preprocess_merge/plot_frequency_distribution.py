import pandas as pd

def analyze_intra_day_intervals(file_path):
    # Load the time series data
    df = pd.read_pickle(file_path)

    # Group by day (date only)
    df_by_day = df.groupby(df.index.date)

    # Calculate intra-day time differences in nanoseconds
    intra_day_deltas_ns = []
    for _, group in df_by_day:
        delta_ns = pd.Series(group.index).diff().dropna().view('int64')
        intra_day_deltas_ns.extend(delta_ns)

    # Convert to milliseconds (1 ms = 1_000_000 ns)
    intra_day_deltas_ms = pd.Series(intra_day_deltas_ns) / 1_000_000

    # Compute basic statistics
    stats = {
        "Number of intervals": len(intra_day_deltas_ms),
        "Mean (ms)": intra_day_deltas_ms.mean(),
        "Minimum (ms)": intra_day_deltas_ms.min(),
        "25th percentile (ms)": intra_day_deltas_ms.quantile(0.25),
        "50th percentile / Median (ms)": intra_day_deltas_ms.median(),
        "75th percentile (ms)": intra_day_deltas_ms.quantile(0.75),
        "Maximum (ms)": intra_day_deltas_ms.max()
    }

    # Print results
    print("\n📊 Intra-day time interval statistics (in milliseconds):\n")
    for label, value in stats.items():
        if isinstance(value, float):
            print(f"{label:35}: {value:,.6f}")
        else:
            print(f"{label:35}: {value}")

if __name__ == "__main__":
    file_path = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Merged_data/INTC_merged_mid_price_train_aligned.pkl"
    analyze_intra_day_intervals(file_path)