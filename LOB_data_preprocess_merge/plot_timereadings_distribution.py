import pandas as pd
import matplotlib.pyplot as plt

def plot_time_step_distribution_high_res():
    input_file = '/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered/INTC_merged_mid_price_train.pkl'
    output_file = '/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered/time_readings_histogram_LOBSTER_data.png'

    # Carica la series
    series = pd.read_pickle(input_file)
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    # Calcola il time delta tra letture consecutive (in secondi)
    time_deltas = series.index.to_series().diff().dt.total_seconds()

    # Rimuove i salti tra giorni diversi
    same_day = series.index.to_series().dt.date == series.index.to_series().dt.date.shift(1)
    filtered_deltas = time_deltas[same_day].dropna()

    # Plot con molti più bin
    plt.figure(figsize=(12, 6))
    plt.hist(filtered_deltas, bins=1000, color='royalblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Time Between Price Readings (ms)')
    plt.ylabel('Frequency')
    plt.title('High-Resolution Distribution of Time Steps (Same Day Only)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ Grafico salvato con più bin: {output_file}")

if __name__ == '__main__':
    plot_time_step_distribution_high_res()
