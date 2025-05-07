import ray
from ray import tune
import json
import pandas as pd
import matplotlib.pyplot as plt


def get_data(experiment_dir: str, experiment_name: str):
    """
    Get the data from the longest running trial of an experiment.

    :param experiment_dir: The directory where the experiment results are stored.
    :param experiment_name: The name of the experiment to get the data from.
    """
    analysis = tune.Analysis(f"{experiment_dir}/{experiment_name}")
    dir_path = analysis.get_best_logdir(metric='timesteps_total', mode='max')  # Get the longest running trial 
    progress_file = f"{dir_path}/progress.csv"
    params_file = f"{dir_path}/params.json"

    df = pd.read_csv(progress_file)
    
    with open(params_file, 'r') as file:
        params = json.load(file)

    return df, params


def plot_experiment(df: pd.DataFrame, params: dict):
    """
    Plot the data of an experiment.

    :param df: The dataframe containing the experiment data.
    :param params: The parameters of the experiment.
    """
    fig, axes = plt.subplots(5, 1, figsize=(10,12), tight_layout=True, sharex=True)

    # Plot reward
    axes[0].plot(df["timesteps_total"], df["evaluation/episode_reward_mean"].rolling(20, min_periods=1).mean(), label="Eval Reward (MA-20)")
    axes[0].plot(df["timesteps_total"], df["episode_reward_mean"].rolling(20, min_periods=1).mean(), label="Mean Reward (MA-20)")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    # Plot TD error
    axes[1].plot(df["timesteps_total"], df["info/learner/default_policy/mean_td_error"], label="Mean TD Error")
    axes[1].plot(df["timesteps_total"], df["info/learner/default_policy/mean_td_error"].rolling(20, min_periods=1).mean(), label="MA-20")
    axes[1].set_ylabel("Mean TD Error")
    axes[1].legend()

    # Plot Q value entropy
    axes[2].plot(df["timesteps_total"], df["evaluation/custom_metrics/q_value_entropy_mean_mean"], label="Q Value Entropy")
    axes[2].plot(df["timesteps_total"], df["evaluation/custom_metrics/q_value_entropy_mean_mean"].rolling(20, min_periods=1).mean(), label="MA-20")
    axes[2].set_ylabel("Entropy")
    axes[2].legend()

    # Plot Q values
    axes[3].plot(df["timesteps_total"], df["evaluation/custom_metrics/q_value_max_mean"], label="Max Q Value")
    axes[3].plot(df["timesteps_total"], df["evaluation/custom_metrics/q_value_mean_mean"], label="Mean Q Value")
    axes[3].plot(df["timesteps_total"], df["evaluation/custom_metrics/q_value_min_mean"], label="Min Q Value")
    axes[3].set_ylabel("Q Value")
    axes[3].legend()

    # Plot learning rate
    axes[4].plot(df["timesteps_total"], df["info/learner/default_policy/cur_lr"], label="Learning Rate")
    axes[4].set_ylabel("Rate")
    axes[4].set_xlabel("Timesteps")
    axes[4].set_ylim(0, max(df["info/learner/default_policy/cur_lr"]) * 1.1)
    axes[4].legend()

    fig.savefig(f"{experiment_name}_results.png", dpi=300)


if __name__ == "__main__":
    experiment_dir = "/cs/student/projects3/cf/2023/fdobber/ray-results"
    experiment_name = "new_reduction_factor"

    df, params = get_data(experiment_dir, experiment_name)
    plot_experiment(df, params)

    plt.show()