import ray
from ray import tune
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.dqn.apex as apex
from ray.rllib.utils import merge_dicts
import abides_gym
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
from sklearn.metrics import confusion_matrix
import seaborn as sns
from ray.tune.registry import register_env
from abides_gym.envs.markets_fx_environment_basic import (
    SubGymFxEnv_basic
)

ENV_NAME = "fx-basic-v0"

register_env(
    ENV_NAME,
    lambda config: SubGymFxEnv_basic(**config),
)

class RLAgent:
    def __init__(self, experiment_dir, experiment_name):
        self.name = 'RL'

        analysis = tune.Analysis(f"{experiment_dir}/{experiment_name}")
        best_trial_path = analysis.get_best_logdir(metric='timesteps_total', mode='max')  # Only 1 trial, always the same
        best_config = analysis.get_best_config(metric='episode_reward_mean', mode='max')  # Only 1 config, always the same

        df = analysis.trial_dataframes[best_trial_path]
        best_checkpoint_number = df.loc[df['episode_reward_mean'].idxmax()].training_iteration
        best_checkpoint = f"{best_trial_path}/checkpoint_{best_checkpoint_number:06d}/checkpoint-{best_checkpoint_number}"

        config = merge_dicts(apex.APEX_DEFAULT_CONFIG.copy(), best_config)
        config["num_gpus"] = 0
        config["explore"] = True
        config["callbacks"] = None

        self.agent = dqn.ApexTrainer(config=config, env=ENV_NAME)
        self.agent.env_creator
        self.agent.restore(best_checkpoint)
        self.config = config
        
    def compute_action(self, state):
        return self.agent.compute_single_action(state)

class RandomAgent:
    def __init__(self):
        self.name = 'random'
        
    def compute_action(self, state):
        return np.random.randint(0, 7)
    
class BuyAndHoldAgent:
    def __init__(self):
        self.name = 'buy-and-hold'
        
    def compute_action(self, state):
        return 1  # Buy limit order at mid-price

class AggresiveAgent:
    def __init__(self):
        self.name = 'aggresive'
        
    def compute_action(self, state):
        up = state[-1]
        mid = state[-2]
        down = state[-3]
        max_val = max(up, mid, down) 

        if up == max_val:
            return 2  # LMT buy order at best ask
        elif down == max_val:
            return 3  # LMT sell order at best bid
        else:
            return 6  # Do nothing

class DoNothingAgent:
    def __init__(self):
        self.name = 'do-nothing'
        
    def compute_action(self, state):
        return 6

def eval_agent(agent, env: gym.Env, timesteps: int, save_data: bool = False):
    info_log = []
    obs_history = []

    env.seed(0)
    obs = env.reset()
    done = False
    episode_length = 0

    while not done and episode_length <= timesteps:
        action = agent.compute_action(obs)  # Choose agent here

        obs, _, done, info = env.step(action)
        info_log.append(list(info.values())[1:])
        obs_history.append(obs.flatten())

        episode_length += 1

        if episode_length % (timesteps // 10) == 0:
            print(f"Episode length: {episode_length}")

    if done:
        print("Reached end of episode")

    info_df = pd.DataFrame(info_log, columns=list(info.keys())[1:])
    obs_df = pd.DataFrame(
        obs_history,
        columns=[
            'time_to_close', 
            'cash', 
            'holdings', 
            'bid', 
            'ask', 
            'bid_volume', 
            'ask_volume', 
            'agent_bid_volume', 
            'agent_ask_volume',
            'volume_imbalance',
            'momentum',
            'directional_signal_0', 
            'directional_signal_1', 
            'directional_signal_2'
    ])

    if save_data:
        # Agents 0 is the exchange agent, which has access to the order book
        order_book = env.gym_agent.kernel.agents[0].order_books['EURUSD']
        L1 = order_book.get_L1_snapshots()
        best_bids = pd.DataFrame(L1["best_bids"], columns=["time","price","qty"])
        best_asks = pd.DataFrame(L1["best_asks"], columns=["time","price","qty"])
        best_bids["time"] = best_bids["time"].apply(lambda x: pd.Timestamp(x).strftime("%Y%m%d %H%M%S%f"))
        best_asks["time"] = best_asks["time"].apply(lambda x: pd.Timestamp(x).strftime("%Y%m%d %H%M%S%f"))
        bid_ask_df = pd.concat([best_bids,best_asks], axis=1)
        bid_ask_df = bid_ask_df.drop("qty", axis=1)
        bid_ask_df.columns = ["time","bid","best_ask_time","ask"]
        bid_ask_df = bid_ask_df.drop("best_ask_time", axis=1)
        bid_ask_df.to_csv("data/simulation_bidask.csv", index=False)

    return info_df, obs_df


def prep_data(info_df: pd.DataFrame, obs_df: pd.DataFrame, env: gym.Env):
    """
    Calculate additional metrics and convert columns to the correct data types.
    """
    info_df['current_time'] = pd.to_datetime(info_df['current_time'])
    info_df['cum_reward'] = info_df['total_reward'].cumsum()
    info_df['agent_return'] = ((info_df['cash_balance'] + (info_df['holdings'] * info_df['mid'])) / env.starting_cash) - 1
    info_df['signal_direction'] = info_df[['directional_signal_0', 'directional_signal_1', 'directional_signal_2']].idxmax(axis=1).apply(lambda x: int(x[-1]) - 1)
    info_df['spread'] = info_df['ask'] - info_df['bid']
    info_df['price_direction'] = info_df['mid'].diff().apply(np.sign).fillna(0)
    info_df['signal_price_agreement'] = info_df['signal_direction'] == info_df['price_direction']

    return info_df, obs_df

def load_and_preprocess_hist_data(file_name: str):
    """
    Load and preprocess data from HistData. Must be in the format:
    date, bid, ask, volume
    
    :param file_name: Name of the file to load
    """
    cols = ["date", "bid", "ask", "volume"]
    df = pd.read_csv(file_name, names=cols)

    df.drop(columns="volume", inplace=True)

    df["date"] = pd.to_datetime(df["date"], format='%Y%m%d %H%M%S%f')
    df.set_index("date", inplace=True)
    df = df.sort_index()

    df['mid'] = (df['bid'] + df['ask']) / 2
    df['returns'] = df['mid'].pct_change()

    return df

def load_and_preprocess_sim_data(file_name: str):
    """
    Load and preprocess data from a simulation. Must be in the format:
    date, bid, ask
    
    :param file_name: Name of the file to load
    """
    cols = ["date", "bid", "ask"]
    df = pd.read_csv(file_name, header=1, names=cols)

    df["date"] = pd.to_datetime(df["date"], format='%Y%m%d %H%M%S%f')
    df.set_index("date", inplace=True)
    df = df.sort_index()

    df['mid'] = df[['bid', 'ask']].mean(axis=1)
    df = df.dropna(subset=['mid'])
    df['returns'] = df['mid'].pct_change()

    return df

def drop_weekends(df):
    """
    Drop weekends from the data.
    The market is closed from Friday 17:00 to Sunday 17:00.

    :param df: DataFrame with a DatetimeIndex
    """
    df = df.copy()
    df = df[df.index.dayofweek != 5]
    df = df[(df.index.dayofweek != 4) | (df.index.hour < 17)]
    df = df[(df.index.dayofweek != 6) | (df.index.hour >= 17)]
    return df

def only_busy_hours(df):
    """
    Only keep data between 08:00 and 12:00 using between_time()

    :param df: DataFrame with a DatetimeIndex
    """
    df = df.copy()
    df = df.between_time("08:00", "12:00")

    return df

def plot_info_data(agent, info_df: pd.DataFrame, env: gym.Env):
    fig, axes = plt.subplots(6, 1, figsize=(10,12), tight_layout=True)

    # Bid, ask, and fundamental, with signal direction at the bottom
    axes[0].plot(info_df['current_time'], info_df["bid"], label="Bid")
    axes[0].plot(info_df['current_time'], info_df["ask"], label="Ask")
    axes[0].plot(info_df['current_time'], info_df["fundamental_value"], label="Fundamental")
    axes[0].scatter(info_df['current_time'], [info_df["bid"].min()] * len(info_df['current_time']), c=info_df["signal_direction"], cmap='RdYlGn', label="Signal Direction", marker='|', s=20)
    axes[0].legend()

    # Holdings
    axes[1].plot(info_df['current_time'], info_df["holdings"], label="Holdings")
    axes[1].axhline(y=0, color='black', linestyle='--', label="Zero")
    axes[1].legend()

    # Portfolio value
    axes[2].plot(info_df['current_time'], info_df["portfolio_value"], label="Portfolio Value")
    axes[2].axhline(y=env.starting_cash, color='black', linestyle='--', label="Starting Cash")
    axes[2].legend()

    # Reward
    axes[3].plot(info_df['current_time'], info_df["pnl_reward"], alpha=0.5, label="PnL")
    axes[3].plot(info_df['current_time'], info_df["directional_reward"], alpha=0.5, label="Directional")
    axes[3].plot(info_df['current_time'], info_df["total_reward"], alpha=0.5, label="Total")
    axes[3].legend()

    # Action distribution
    total_actions = sum(env.custom_metrics_tracker.action_counter.values())
    action_distribution = {action: count / total_actions for action, count in env.custom_metrics_tracker.action_counter.items()}
    _, probabilities = zip(*sorted(action_distribution.items()))

    actions = ['buy@bid', 'buy@mid', 'buy@ask', 'sell@bid', 'sell@mid', 'sell@ask', 'do_nothing']

    axes[4].bar(actions, probabilities)
    axes[4].set_xlabel('Action')
    axes[4].set_ylabel('Probability')

    # Assorted metrics
    axes[-1].axis('off')

    stock_return = (info_df['mid'].iat[-1] - info_df['mid'].iat[0]) / info_df['mid'].iat[0]
    agent_return = info_df['agent_return'].iat[-1]
    volatility = info_df['mid'].pct_change().std()
    avg_spread = info_df['spread'].mean()
    num_zeros = (info_df['pnl_reward'] == 0).sum() / len(info_df)

    # Column 1
    axes[-1].text(0, 1, f"Date: {pd.Timestamp(env.kernelStartTime).date()}", fontsize=12)
    axes[-1].text(0, 0.8, f"Cumulative Reward: {round(info_df['cum_reward'].iat[-1],2)}", fontsize=12)
    axes[-1].text(0, 0.6, f"PnL / Directional: {round(info_df['pnl_reward'].sum(),2)} / {round(info_df['directional_reward'].sum(),2)}", fontsize=12)
    axes[-1].text(0, 0.4, f"PnL = 0: {round(num_zeros, 3) * 100}%", fontsize=12)

    axes[-1].text(1/3, 1.0, f"Volatility: {round(volatility, 5)}", fontsize=12)
    axes[-1].text(1/3, 0.8, f"Avg. Spread: {round(avg_spread, 5)}", fontsize=12)
    axes[-1].text(1/3, 0.4, f"Agent Return: {round(agent_return * 100,2)}%", fontsize=12)
    axes[-1].text(1/3, 0.2, f"Stock Return: {round(stock_return * 100,2)}%", fontsize=12)
    axes[-1].text(1/3, 0.0, f"Outperformance: {round((agent_return - stock_return) * 100,2)}%", fontsize=12)

    # Column 2
    axes[-1].text(2/3, 1.0, f"Agent type: {agent.name}", fontsize=12)
    if agent.name == 'RL':
        axes[-1].text(2/3, 0.8, f"lr: {agent.config.get('lr', 'N/A')}", fontsize=12)
        axes[-1].text(2/3, 0.6, f"n_step: {agent.config.get('n_step', 'N/A')}", fontsize=12)
        axes[-1].text(2/3, 0.4, f"dirichlet_a_h: {agent.config.get('env_config', {}).get('dirichlet_a_h', 'N/A')}", fontsize=12)
        axes[-1].text(2/3, 0.2, f"reduction_factor: {agent.config.get('env_config', {}).get('reduction_factor', 'N/A')}", fontsize=12)

    fig.savefig(f"agent_eval.png", dpi=300)


def plot_observation_histograms(obs_df: pd.DataFrame):
    """
    Plot histograms of the observation columns of the dataframe.

    :param obs_df: The dataframe containing the observations.
    """
    fig, axes = plt.subplots(5, 3, figsize=(15, 15), tight_layout=True)

    for i in range(len(obs_df.columns)):
        ax = axes[i // 3, i % 3]
        ax.hist(obs_df.iloc[:, i], bins=50)
        ax.set_title(obs_df.columns[i])

    fig.savefig('observation_histograms.png', dpi=300)


def plot_reward_histograms(info_df: pd.DataFrame):
    """
    Plot histograms of the reward columns of the dataframe.

    :param info_df: The dataframe containing the reward columns.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

    axes[0].hist(info_df['pnl_reward'], bins=50)
    axes[0].set_title('PnL Reward')

    axes[1].hist(info_df['directional_reward'], bins=50)
    axes[1].set_title('Directional Reward')

    axes[2].hist(info_df['total_reward'], bins=50)
    axes[2].set_title('Total Reward')

    fig.savefig('reward_histograms.png', dpi=300)

def plot_confusion_matrix(info_df: pd.DataFrame):
    """
    Plot a confusion matrix of the signal direction vs. the price direction.

    :param info_df: The dataframe containing the signal and price direction columns.
    """
    cm = confusion_matrix(info_df['price_direction'], info_df['signal_direction'])

    # Create labels for the plot
    labels = [-1, 0, 1]

    # Create a pyplot figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use seaborn to create a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)

    # Set labels and title
    plt.xlabel('Predicted (Signal Direction)')
    plt.ylabel('Actual (Price Direction)')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.tight_layout()
    plt.show()

    fig.savefig('confusion_matrix.png', dpi=300)

def plot_order_frequency(hist_df: pd.DataFrame, sim_df: pd.DataFrame, config: dict):
    """
    Analyze and plot the order frequency.
    
    :param dhist_dff: DataFrame with a DatetimeIndex
    :param sim_df: DataFrame with a DatetimeIndex
    :param config: Configuration dictionary
    """
    hist_df_1min = hist_df.resample('1min').count()
    hist_df_1min = drop_weekends(hist_df_1min)
    hist_df_1min = only_busy_hours(hist_df_1min)

    sim_df_1min = sim_df.resample('1min').count()

    fig, axes = plt.subplots(2, 1, figsize=(10,6), tight_layout=True)

    hist_df_1min['mid'].hist(bins=100, ax=axes[0], histtype='step', density=True, label='Historical data')
    sim_df_1min['mid'].hist(bins=100, ax=axes[0], histtype='step', density=True, label='Simulation')

    max_value = max(hist_df_1min['mid'].max().round(-2), sim_df_1min['mid'].max().round(-2))

    x = np.linspace(0, max_value, 1000)

    alpha, loc, beta = gamma.fit(hist_df_1min['mid'])
    y = gamma.pdf(x, alpha, loc=loc, scale=beta)
    axes[0].plot(x, y, lw=1, alpha=0.8, label='Historical data fit')

    alpha, loc, beta = gamma.fit(sim_df_1min['mid'])
    y = gamma.pdf(x, alpha, loc=loc, scale=beta)
    axes[0].plot(x, y, lw=1, alpha=0.8, label='Simulation fit')

    axes[0].set_title("Distribution of price changes per minute")
    axes[0].set_xlabel("Price change")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[-1].axis('off')

    axes[-1].text(0, 1, f"Value agents: {config['background_config_extra_kvargs']['num_value_agents']}, {config['background_config_extra_kvargs']['val_wake_up_freq']}", fontsize=12)
    axes[-1].text(0, 0.8, f"Momentum agents: {config['background_config_extra_kvargs']['num_momentum_agents']}, {config['background_config_extra_kvargs']['m_wake_up_freq']}", fontsize=12)
    axes[-1].text(0, 0.6, f"Market maker: {config['background_config_extra_kvargs']['num_mm']}, {config['background_config_extra_kvargs']['mm_wake_up_freq']}", fontsize=12)

    fig.savefig("order_frequency.png", dpi=300)

if __name__ == "__main__":
    experiment_dir = "/cs/student/projects3/cf/2023/fdobber/ray-results"
    experiment_name = "new_reduction_factor_2"
    timesteps = 20_000

    # _, config = get_data(experiment_dir, experiment_name)
    config = {
        "env_config": {
            "background_config": "fx_basic",
            "timestep_duration": "0.2s",
            "state_timesteps_back": 0,
            "dirichlet_steps_ahead": 8,
            "dirichlet_k": 0.00005,  # 0.0002
            "dirichlet_a_h": 5.0,
            "dirichlet_a_l": 1.0,
            "starting_cash": 2_000_000,
            "max_holdings": 10,
            "order_fixed_size": 1,
            "state_history_length": 2,
            "market_data_buffer_length": 5,
            "w_direc": 1,
            "reduction_factor":  0.99996,
            "kappa": 4,
            "debug_mode": False,
            "first_interval": "00:00:30",
            "background_config_extra_kvargs": {
                "ticker": "EURUSD",
                "period_starts": ['20240105 082700', '20240105 095300', '20240111 082900', '20240116 104900', '20240125 083000', '20240131 135900', '20240131 150000', '20240202 082400', '20240213 082900', '20240216 082600', '20240222 031400', '20240305 095800', '20240308 082600', '20240312 072900', '20240320 125900', '20240403 095700', '20240405 082500', '20240410 082600', '20240411 081300', '20240415 082500', '20240423 030200', '20240423 093300', '20240425 082900', '20240430 082700', '20240501 135900', '20240503 082900', '20240503 093100', '20240515 082300', '20240606 081400', '20240607 082800'],
                "fundamental_file_path": "/cs/student/projects3/cf/2023/fdobber/ucl-thesis/data/EURUSD_2024_H1_midpoint_1000.pkl",
                "exchange_log_orders": False,
                "log_orders": False,
                "book_logging": False,  # True for eval
                "book_log_depth": 10,
                "starting_cash": 2_000_000,
                "seed": 1,
                "num_noise_agents": 5000,
                "num_momentum_agents": 40,
                "m_wake_up_freq": "0.5s",
                "num_value_agents": 200,
                "val_wake_up_freq": "60s",
                "val_lambda_a": 1e-11,
                "num_mm": 2,
                "mm_wake_up_freq": "0.1s",
                "mm_window_size": "adaptive",
                "mm_pov": 0.025,
                "mm_num_ticks": 5,
                "mm_min_order_size": 3,
                "mm_skew_beta": 0,
                "mm_price_skew": 4,
                "mm_level_spacing": 5,
                "mm_spread_alpha": 0.75,
                "mm_backstop_quantity": 0,
                "mm_cancel_limit_delay": 50,
                "val_kappa": 1.67e-15,
                "val_vol": 1e-8,
                "computation_delay": 0,
            }
        }
    }

    # agent = RLAgent(experiment_dir, experiment_name)
    agent = AggresiveAgent()

    env = gym.make(ENV_NAME, **config["env_config"])

    info_metrics, obs_history = eval_agent(agent, env, timesteps, save_data=False)
    info_df, obs_df = prep_data(info_metrics, obs_history, env)
    plot_info_data(agent, info_df, env)
    plot_observation_histograms(obs_df)
    plot_reward_histograms(info_df)
    plot_confusion_matrix(info_df)

    # hist_df = load_and_preprocess_hist_data("data/EURUSD_2024_01_bidask.csv")
    # sim_df = load_and_preprocess_sim_data("data/simulation_bidask.csv")
    # plot_order_frequency(hist_df, sim_df, config['env_config'])