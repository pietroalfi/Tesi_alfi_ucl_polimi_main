import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_spread_analysis(file_path, output_dir):
    spread_series = pd.read_pickle(file_path)
    spread_series.index = pd.to_datetime(spread_series.index)
    spread_series = spread_series.between_time("09:00", "16:30")
    os.makedirs(output_dir, exist_ok=True)
    """
    # Plot 1: Heatmap classica
    df = spread_series.to_frame(name="spread")
    df['date'] = df.index.date
    df['time'] = df.index.time
    df_resampled = df.resample("5min").mean(numeric_only=True)
    df_resampled['date'] = df_resampled.index.date
    df_resampled['time'] = df_resampled.index.time
    pivot = df_resampled.pivot_table(index='time', columns='date', values='spread')

    plt.figure(figsize=(16, 6))
    sns.heatmap(pivot, cmap="viridis", linewidths=0.1)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Time", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_spread.png"))
    plt.close()
    print("✅ Plot 1: Heatmap classica salvata")

    # Plot 2: Media giornaliera con barre di errore
    daily_stats = spread_series.groupby(spread_series.index.date).agg(['mean', 'std'])
    daily_stats.index = pd.to_datetime(daily_stats.index)

    plt.figure(figsize=(12, 6))
    plt.errorbar(daily_stats.index, daily_stats['mean'], yerr=daily_stats['std'],
                 fmt='-o', ecolor='gray', capsize=3)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Daily Average Spread', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_mean_std_errorbar.png'))
    plt.close()
    print("✅ Plot 2: Media giornaliera con barre di errore salvata")
    """
    # Plot 3: Media intraday con CI95%
    df_resampled = spread_series.resample("5min").mean(numeric_only=True).to_frame(name="spread")
    df_resampled['time'] = df_resampled.index.time
    df_resampled['date'] = df_resampled.index.date
    pivot_ci = df_resampled.pivot(index='date', columns='time', values='spread')

    start_time = pd.to_datetime("09:00").time()
    end_time = pd.to_datetime("16:30").time()
    filtered_columns = [col for col in pivot_ci.columns if start_time <= col <= end_time]
    pivot_ci = pivot_ci[filtered_columns]

    mean_spread = pivot_ci.mean()
    std_spread = pivot_ci.std()
    n = pivot_ci.count()
    ci95 = 1.96 * (std_spread / np.sqrt(n))

    time_labels = [t.strftime('%H:%M') for t in mean_spread.index]

    plt.figure(figsize=(14, 6))
    plt.plot(time_labels, mean_spread, label='Average Spread', color='blue')
    plt.fill_between(time_labels,
                     mean_spread - ci95,
                     mean_spread + ci95,
                     color='skyblue', alpha=0.4, label='CI 95%')

    tick_interval = 3
    plt.xticks(
        ticks=np.arange(0, len(time_labels), tick_interval),
        labels=[time_labels[i] for i in range(0, len(time_labels), tick_interval)],
        rotation=45,
        fontsize=12
    )
    plt.yticks(fontsize=14)

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Spread", fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=19)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_spread_5min_ci95_grid.png'))
    plt.close()
    print("✅ Plot 3: Spread medio 5min con CI 95% salvato")

if __name__ == '__main__':
    file_path = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered_spreads/merged_spread.pkl"
    output_dir = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Spread_analysis"

    plot_spread_analysis(file_path, output_dir)
    print(f"\n🎉 Analisi completata. Grafici salvati in: {output_dir}")
