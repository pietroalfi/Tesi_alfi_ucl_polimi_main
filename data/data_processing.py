import pandas as pd
import matplotlib.pyplot as plt


def convert_to_pickle(file_name: str, multiplier: int = 1000):
    """
    Converts a csv file with bid/ask data to a pickle file with the midpoint price multiplied by a given multiplier. The output price data will be interpreted as price in cents. For example, a bid price of 1.234567 will be 123.4567 cents, and a multiplier of 1000 will convert it to 123,456.7 cents.
    :param file_name: The name of the csv file to convert.
    :param multiplier: The multiplier to apply to the midpoint.
    """
    cols = ["Date", "Bid Quote", "Ask Quote", "Volume"]
    df = pd.read_csv(file_name, names=cols)

    df.drop(columns="Volume", inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d %H%M%S%f')
    df.set_index("Date", inplace=True)
    df = df.sort_index()

    df["Midpoint"] = (df["Ask Quote"] + df["Bid Quote"]) / 2
    series = df["Midpoint"]
    series = (series * 100 * multiplier).round().astype(int)

    output_file = file_name.replace(".csv", ".pkl").replace("bidask", f"midpoint_{multiplier}")

    series.to_pickle(output_file)


def combine_months(file_names: list, output_file: str):
    """
    Combines multiple CSV files into a single CSV file.
    :param file_names: A list of file names to combine.
    :param output_file: The name of the output file.
    """
    df = pd.concat([pd.read_csv(file_name, header=None) for file_name in file_names])

    df.to_csv(output_file, header=False, index=False)


def drop_weekends(df: pd.DataFrame):
    """
    Drop weekends from the data. The market weekend is defined as Friday 17:00 to Sunday 17:00.

    :param df: DataFrame with a DatetimeIndex
    """
    df = df.copy()
    df = df[df.index.dayofweek != 5]
    df = df[(df.index.dayofweek != 4) | (df.index.hour < 17)]
    df = df[(df.index.dayofweek != 6) | (df.index.hour >= 17)]
    
    return df


def only_busy_hours(df: pd.DataFrame, start_time: str = '08:00', end_time: str = '12:00'):
    """
    Only keep data between start_time and end_time, which default to 08:00 and 12:00.

    :param df: DataFrame with a DatetimeIndex
    :param start_time: Start time of the period
    :param end_time: End time of the period
    """
    df = df.copy()
    df = df.between_time(start_time, end_time)

    return df


def load_and_preprocess_data(file_name: str):
    """
    Load the data from a pickle file and preprocess it by resampling to 1 minute intervals and adding some features.

    :param file_name: Name of the pickle file to load.
    """
    df = pd.read_pickle(file_name)
    df = pd.DataFrame(df)

    minute_df = df.resample("1min").last()
    minute_df['returns'] = minute_df['Midpoint'].pct_change(fill_method=None)
    minute_df['volatility'] = minute_df['returns'].rolling(60).std()
    minute_df['range'] = minute_df['Midpoint'].rolling(60).max() - minute_df['Midpoint'].rolling(60).min()
    minute_df['trend'] = minute_df['returns'].rolling(60).mean() / minute_df['volatility']
    minute_df['jump'] = minute_df['Midpoint'].pct_change(periods=2, fill_method=None)

    return minute_df

def find_non_overlapping_periods(df: pd.DataFrame, metric: str, max_val: bool, n_periods: int):
    """
    Find n_periods non-overlapping periods with the highest or lowest values of a given metric.

    :param df: DataFrame with a DatetimeIndex
    :param metric: Metric to find the periods for
    :param max_val: Boolean indicating whether to find the highest or lowest values
    :param n_periods: Number of periods to return

    :return: List of DatetimeIndex objects representing the start of the periods
    """
    periods = []
    df_copy = df.copy()

    for _ in range(n_periods):
        idx = df_copy[metric].idxmax() if max_val else df_copy[metric].idxmin()
        periods.append(idx - pd.Timedelta(hours=1))
        df_copy = df_copy[(df_copy.index < idx - pd.Timedelta(hours=1)) | 
                          (df_copy.index > idx + pd.Timedelta(hours=1))]
        
    return sorted(periods)

def plot_period_type(df: pd.DataFrame, periods: dict, period_type: str):
    """
    Plots the periods of a given type in a grid of subplots.

    :param df: DataFrame with a DatetimeIndex
    :param periods: Dictionary with the periods of different types
    :param period_type: Type of period to plot
    """
    fig, axes = plt.subplots(6, 5, figsize=(18, 15))
    axes = axes.flatten()

    period_list = periods[period_type]

    for i, period in enumerate(period_list):
        ax = axes[i]
        if period_type == "jump_up" or period_type == "jump_down":
            plot_data = df[(df.index > period - pd.Timedelta(minutes=30)) & (df.index <= period + pd.Timedelta(minutes=30))]
        else:
            plot_data = df[(df.index > period) & (df.index <= period + pd.Timedelta(hours=1))]
        ax.plot(plot_data.index, plot_data['Midpoint'])
        ax.set_title(period.strftime("%Y-%m-%d %H:%M:%S"))
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add figure title
    fig.suptitle(f"Periods of type {period_type}", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"{period_type}.png")

def plot_trading_periods(file_name: str):
    """
    Plots the trading periods of a given pickle file with the midpoint price data.

    :param file_name: The name of the file to plot.
    """
    df = pd.read_pickle(file_name)
    df = df.sort_index()

    # Limit to relevant hours
    df = only_busy_hours(df)

    # Get unique days
    days = df.index.date
    days = sorted(list(set(days)))

    fig, axes = plt.subplots(6, 4, figsize=(18, 13))
    axes = axes.flatten()

    for i, day in enumerate(days):
        # Filter dataframe for the specific day
        df_day = df[df.index.date == day]

        # Plot the data on the corresponding subplot
        ax = axes[i]
        ax.plot(df_day.index, df_day)
        ax.set_title(day)
        ax.set_xlim([df_day.index.min(), df_day.index.max()])
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    plt.tight_layout()
    fig.savefig("price_plots.png", dpi=300)

if __name__ == "__main__":
    # combine_months(
    #     [
    #         "data/EURUSD_2024_01_bidask.csv", 
    #         "data/EURUSD_2024_02_bidask.csv", 
    #         "data/EURUSD_2024_03_bidask.csv", 
    #         "data/EURUSD_2024_04_bidask.csv",
    #         "data/EURUSD_2024_05_bidask.csv",
    #         "data/EURUSD_2024_06_bidask.csv"
    #     ], 
    #     "data/EURUSD_2024_H1_bidask.csv"
    # )
    # convert_to_pickle("data/EURUSD_2024_H1_bidask.csv", multiplier=1000)

    df = load_and_preprocess_data("data/EURUSD_2024_H1_midpoint_1000.pkl")
    period_types = [
        ("high_vol", "volatility", True),
        ("low_vol", "volatility", False),
        ("side_ways", "range", False),
        ("trend_up", "trend", True),
        ("trend_down", "trend", False),
        ("jump_up", "jump", True),
        ("jump_down", "jump", False)
    ]

    periods = {label: find_non_overlapping_periods(df, metric, max_val, 30)
               for label, metric, max_val in period_types}
    plot_period_type(df, periods, "trend_up")
    
    # Print periods
    for period_type in period_types:
        print(period_type[0], [period.strftime("%Y%m%d %H%M%S") for period in periods[period_type[0]]])
