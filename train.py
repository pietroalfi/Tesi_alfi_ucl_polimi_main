import ray
from ray.tune.registry import register_env
from abides_gym.envs.markets_fx_environment_basic import (
    SubGymFxEnv_basic
)
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from typing import Dict
import numpy as np

ray.shutdown()  # Shut down any currently running experiments
ray.init()  # Set up a new local Ray server

# Register a custom environment, so it can be used in `ray.tune.run()`
register_env(
    "fx-basic-v0",
    lambda config: SubGymFxEnv_basic(**config),
)

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.user_data["q_values"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        policy = next(iter(policies.values()))  # Get the first (and likely only) policy
        episode.user_data["q_values"].append(policy.q_values)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        q_values = np.array(episode.user_data["q_values"])
            
        episode.custom_metrics["q_value_mean"] = np.mean(q_values)
        episode.custom_metrics["q_value_max"] = np.max(q_values)
        episode.custom_metrics["q_value_min"] = np.min(q_values)

        q_probs = np.exp(q_values - np.max(q_values, axis=1, keepdims=True))
        q_probs = q_probs / np.sum(q_probs, axis=1, keepdims=True)
        entropy = -np.sum(q_probs * np.log(q_probs + 1e-10), axis=1)
        episode.custom_metrics["q_value_entropy_mean"] = np.mean(entropy)

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        pass

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass


# tensorboard --logdir=~/ray_results/[NAME] 

tune.run(
    "APEX",  # Algorithm to use
    name="sideways",  # Experiment name
    resume=False,  # Set to True if you want to continue a stopped experiment, requires name and local_dir
    checkpoint_at_end=True,  # Save at the end
    checkpoint_freq=1,  # Create checkpoint every n iterations
    local_dir="/g100/home/userexternal/palfi000/ray_results",
    stop={  # Stopping condition(s), OR clause
        "timesteps_total": 5_000_000,
    },
    config={
        # Environment
        "env": "fx-basic-v0",
        "env_config": {
            "background_config": "fx_basic",
            "timestep_duration": "0.2s",
            "state_timesteps_back": 0,
            "dirichlet_steps_ahead": 8,
            "dirichlet_k": 0.00005,
            "dirichlet_a_h": 5.0,
            "dirichlet_a_l": 1.0,
            "starting_cash": 2_000_000,
            "max_holdings": 10,
            "order_fixed_size": 1,
            "state_history_length": 2,
            "market_data_buffer_length": 5,
            "w_direc": 1,
            "reduction_factor":  0.99996,
            "kappa": 5,
            "debug_mode": False,
            "first_interval": "00:00:30",
            "background_config_extra_kvargs": {
                "ticker": "EURUSD",
                #"period_starts": ['20240117 160100', '20240124 171100', '20240208 231800', '20240212 220500', '20240213 001000', '20240216 001200', '20240220 172000', '20240226 155000', '20240305 170700', '20240311 161400', '20240312 160400', '20240314 224500', '20240317 231000', '20240318 160400', '20240320 223900', '20240321 164200', '20240326 161200', '20240327 231300', '20240329 002800', '20240329 120000', '20240329 130600', '20240401 155400', '20240424 170200', '20240425 160100', '20240425 171800', '20240507 160200', '20240512 175800', '20240521 004100', '20240521 171800', '20240605 171200'],
                "fundamental_file_path":"/home/pietro/ucl-thesis-main/ucl-thesis-main/data/EURUSD_2024_01_midpoint_1000.pkl",
                "exchange_log_orders": False,
                "log_orders": False,
                "book_logging": False,
                "book_log_depth": 10,
                "starting_cash": 2_000_000,
                "seed": 1,
                "num_noise_agents": 10000,
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
            },
        },
        # General
        "framework": "tfe",
        "seed": 1,
        # Logging
        "timesteps_per_iteration": 10_000,
        "min_iter_time_s": 1, 
        "callbacks": MyCallbacks,
        # Performance
        "num_cpus_for_driver": 2,  # One extra for evaluation
        "num_gpus": 1,
        "num_workers": 4,
        "num_envs_per_worker": 4,
        "num_cpus_per_worker": 4,
        # Model
        "dueling": True,
        "double_q": True,
        "prioritized_replay": False,  # Must set worker_side_prioritization to false as well
        "worker_side_prioritization": False,  # Must be false when not using prioritized_replay
        "n_step": 10,
        "num_atoms": 51,
        "v_min": -10,
        "v_max": 10,
        # "noisy": False,  # If true, turn off epsilon-greedy exploration
        # Neural net
        "hiddens": [32, 64],
        "model": {
            "fcnet_hiddens": [64, 128, 64],
            "fcnet_activation": "tanh",
        },
        # Learning parameters
        "gamma": 0.99,
        # "lr_schedule": [[0,2e-5], [2.5e5, 5e-6]],
        "lr": 5e-6,
        "observation_filter": "MeanStdFilter",  # Normalize observations
        # Training parameters
        "buffer_size": 200000,
        "learning_starts": 5000,
        "train_batch_size": 50,
        "rollout_fragment_length": 50,
        "target_network_update_freq": 5000,
        # Evaluation
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "explore": False,
        },
        "evaluation_num_workers": 0,
        "metrics_smoothing_episodes": 20,
    },
)