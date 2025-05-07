"""
General purpose utility functions for the simulator, attached to no particular class.
Available to any agent or other module/utility.  Should not require references to
any simulator object (kernel, agent, etc).
"""
import inspect
import hashlib
import os
import random
import pickle
from typing import List, Dict, Any, Callable, Optional

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import NanosecondTime


def subdict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Returns a dictionnary with only the keys defined in the keys list
    Arguments:
        - d: original dictionnary
        - keys: list of keys to keep
    Returns:
        - dictionnary with only the subset of keys
    """
    return {k: v for k, v in d.items() if k in keys}


def restrictdict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Returns a dictionnary with only the intersections of the keys defined in the keys list and the keys in the o
    Arguments:
        - d: original dictionnary
        - keys: list of keys to keep
    Returns:
        - dictionnary with only the subset of keys
    """
    inter = [k for k in d.keys() if k in keys]
    return subdict(d, inter)


def custom_eq(a: Any, b: Any) -> bool:
    """returns a==b or True if both a and b are null"""
    return (a == b) | ((a != a) & (b != b))


# Utility function to get agent wake up times to follow a U-quadratic distribution.
def get_wake_time(open_time, close_time, a=0, b=1):
    """
    Draw a time U-quadratically distributed between open_time and close_time.

    For details on U-quadtratic distribution see https://en.wikipedia.org/wiki/U-quadratic_distribution.
    """

    def cubic_pow(n: float) -> float:
        """Helper function: returns *real* cube root of a float."""

        if n < 0:
            return -((-n) ** (1.0 / 3.0))
        else:
            return n ** (1.0 / 3.0)

    #  Use inverse transform sampling to obtain variable sampled from U-quadratic
    def u_quadratic_inverse_cdf(y):
        alpha = 12 / ((b - a) ** 3)
        beta = (b + a) / 2
        result = cubic_pow((3 / alpha) * y - (beta - a) ** 3) + beta
        return result

    uniform_0_1 = np.random.rand()
    random_multiplier = u_quadratic_inverse_cdf(uniform_0_1)
    wake_time = open_time + uniform_0_1 * (close_time - open_time)

    return wake_time


def fmt_ts(timestamp: NanosecondTime) -> str:
    """
    Converts a timestamp stored as nanoseconds into a human readable string.
    """
    return pd.Timestamp(timestamp, unit="ns").strftime("%Y-%m-%d %H:%M:%S")


def str_to_ns(string: str) -> NanosecondTime:
    """
    Converts a human readable time-delta string into nanoseconds.

    Arguments:
        string: String to convert into nanoseconds. Uses Pandas to do this.

    Examples:
        - "1s" -> 1e9 ns
        - "1min" -> 6e10 ns
        - "00:00:30" -> 3e10 ns
    """
    return pd.to_timedelta(string).to_timedelta64().astype(int)


def datetime_str_to_ns(string: str) -> NanosecondTime:
    """
    Takes a datetime written as a string and returns in nanosecond unix timestamp.

    Arguments:
        string: String to convert into nanoseconds. Uses Pandas to do this.
    """
    return pd.Timestamp(string).value


def ns_date(ns_datetime: NanosecondTime) -> NanosecondTime:
    """
    Takes a datetime in nanoseconds unix timestamp and rounds it to that day at 00:00.

    Arguments:
        ns_datetime: Nanosecond time value to round.
    """
    return ns_datetime - (ns_datetime % (24 * 3600 * int(1e9)))


def parse_logs_df(end_state: dict) -> pd.DataFrame:
    """
    Takes the end_state dictionnary returned by an ABIDES simulation goes through all
    the agents, extracts their log, and un-nest them returns a single dataframe with the
    logs from all the agents warning: this is meant to be used for debugging and
    exploration.
    """
    agents = end_state["agents"]
    dfs = []
    for agent in agents:
        messages = []
        for m in agent.log:
            m = {
                "EventTime": m[0] if isinstance(m[0], (int, np.int64)) else 0,
                "EventType": m[1],
                "Event": m[2],
            }
            event = m.get("Event", None)
            if event == None:
                event = {"EmptyEvent": True}
            elif not isinstance(event, dict):
                event = {"ScalarEventValue": event}
            else:
                pass
            try:
                del m["Event"]
            except:
                pass
            m.update(event)
            if m.get("agent_id") == None:
                m["agent_id"] = agent.id
            m["agent_type"] = agent.type
            messages.append(m)
        dfs.append(pd.DataFrame(messages))

    return pd.concat(dfs)


# caching utils: not used by abides but useful to have
def input_sha_wrapper(func: Callable) -> Callable:
    """
    compute a sha for the function call by looking at function name and inputs for the call
    """

    def inner(*args, **kvargs):
        argspec = inspect.getfullargspec(func)
        index_first_kv = len(argspec.args) - (
            len(argspec.defaults) if argspec.defaults != None else 0
        )
        if len(argspec.args) > 0:
            total_kvargs = dict(
                (k, v) for k, v in zip(argspec.args[index_first_kv:], argspec.defaults)
            )
        else:
            total_kvargs = {}
        total_kvargs.update(kvargs)
        input_sha = (
            func.__name__
            + "_"
            + hashlib.sha1(str.encode(str((args, total_kvargs)))).hexdigest()
        )
        return {"input_sha": input_sha}

    return inner


def cache_wrapper(
    func: Callable, cache_dir="cache/", force_recompute=False
) -> Callable:
    """
    local caching decorator
    checks the functional call sha is only there is specified directory
    """

    def inner(*args, **kvargs):
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        sha_call = input_sha_wrapper(func)(*args, **kvargs)
        cache_path = cache_dir + sha_call["input_sha"] + ".pkl"
        if os.path.isfile(cache_path) and not force_recompute:
            with open(cache_path, "rb") as handle:
                result = pickle.load(handle)
            return result
        else:
            result = func(*args, **kvargs)
            with open(cache_path, "wb") as handle:
                pickle.dump(result, handle)
            return result

    return inner


def random_weekday(start_date: datetime, end_date: datetime) -> Optional[str]:
    """
    Function which returns a weekday for use as a start date in the abides framework.
    """
    # Create a list of all dates between start_date and end_date
    all_dates = pd.date_range(start=start_date, end=end_date)

    # Filter out weekends
    weekdays = [date for date in all_dates if date.weekday() < 5]  # weekday() < 5 means Monday to Friday

    # Return None if there are no weekdays in the range
    if not weekdays:
        return None
    
    # Select a random weekday
    random_date = random.choice(weekdays)
    return random_date.strftime('%Y%m%d')

def plot_state_hist(data, time_index, sim_date):
    cols = ["T - t", "Cash", "Holdings", "Bid", "Ask", "Bid Vol", "Ask Vol", "Agent Bid Vol",
         "Agent Ask Vol", "Vol Imbalance", "Momentum", "d_down", "d_mid", "d_up", "Bid1", "Ask1", "Bid Vol1", "Ask Vol1", "Agent Bid Vol1",
         "Agent Ask Vol1", "Fundamental"]
    cols = ["T - t", "Cash", "Holdings", "Bid", "Ask", "Bid Vol", "Ask Vol", "Agent Bid Vol",
         "Agent Ask Vol", "Vol Imbalance", "Momentum", "d_down", "d_mid", "d_up", "Fundamental"]
    df = pd.DataFrame(data, columns=cols)
    fig, axes = plt.subplots(2, 3, figsize=(15,8), tight_layout=True)
    time_index_pd = pd.to_datetime(time_index)

    axes[0,0].hist(df["Momentum"], bins=50)
    axes[0,0].set_xlabel("Value")
    axes[0,0].set_ylabel("Frequency")

    axes[0,1].plot(time_index_pd, df["Cash"])
    axes[0,1].set_xlabel("Time")
    axes[0,1].set_ylabel("Cash")
    
    axes[0,2].plot(time_index_pd, df["Holdings"])
    axes[0,2].set_xlabel("Time")
    axes[0,2].set_ylabel("Holdings")

    axes[1,0].plot(time_index_pd, df["Bid"])
    axes[1,0].plot(time_index_pd, df["Ask"])
    axes[1,0].plot(time_index_pd, df["Fundamental"], label="Fundamental")
    axes2 = axes[1,0].twinx()
    axes2.plot(time_index_pd, df["Ask"] - df["Bid"], alpha=0.4, color="black", label="Spread")
    lines, labels = axes[1,0].get_legend_handles_labels()
    lines2, labels2 = axes2.get_legend_handles_labels()
    lines += lines2
    labels += labels2
    axes2.set_ylim([0,100])
    axes[1,0].legend(lines, labels)
    axes[1,0].set_xlabel("Time")
    axes[1,0].set_ylabel("Price")

    axes[1,1].plot(time_index_pd, df["Bid Vol"], label="Bid")
    axes[1,1].plot(time_index_pd, df["Ask Vol"], label="Ask")
    axes[1,1].legend()
    axes[1,1].set_xlabel("Time")
    axes[1,1].set_ylabel("Volume")

    axes[1,2].hist(df["Vol Imbalance"], label="Volume Imbalance", color="black", bins=50)
    axes[1,2].set_xlabel("Value")
    axes[1,2].set_ylabel("Frequency")

    titles = ["Momentum", "Cash Balance", "Holdings", "Bid/Ask Price", 
                "Bid/Ask Volume", "Volume Imbalance"]
    for ax, v in zip(axes.flatten(), titles):
        ax.set_title(v)
    fig.savefig(f"results/sim_results_{sim_date}.png", dpi=200)
    plt.close()

def save_pickle(file_name, object):
    # Open the file in binary write mode
    with open(file_name, 'wb') as file:
        # Use pickle.dump() to export the dictionary to the file
        pickle.dump(object, file)

    print("Dictionary exported successfully.")

def load_pickle(file_name):
    # Importing the dictionary back from the pickle file
    with open(file_name, 'rb') as file:
        loaded_dict = pickle.load(file)
        
    return loaded_dict