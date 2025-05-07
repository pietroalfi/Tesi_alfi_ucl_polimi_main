import ray
from ray.tune.registry import register_env
from  abides_gym.envs.markets_intc_apex_environment_basic import (
    SubGymApexIntcEnv_basic
)
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from abides_gym.envs.markets_apex_custom_metrics_checkpoint_modification import MyCallbacks
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
    "intc-apex-v0",
    lambda config: SubGymApexIntcEnv_basic(**config),
)
# tensorboard --logdir=~/ray_results/[NAME] 

tune.run(
    "APEX",  # Algorithm to use
    name="apex_training",  # Experiment name
    resume=True,  # Set to True if you want to continue a stopped experiment, requires name and local_dir
    checkpoint_at_end=True,  # Save at the end
    checkpoint_freq=1,  # Create checkpoint every n iterations
    local_dir="/g100/home/userexternal/palfi000/ray_results",
    stop={  # Stopping condition(s), OR clause
        "timesteps_total": 5_000_000,
    },
    config={
        # Environment
        "env": "intc-apex-v0",
        "env_config": {
            "background_config": "intel_basic",
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
                "ticker": "INTC",
                "period_starts": ['20241001 120000', '20241002 120000', '20241003 120000', '20241004 120000',
                                  '20241007 120000', '20241008 120000', '20241009 120000', '20241010 120000',
                                  '20241011 120000', '20241014 120000', '20241015 120000', '20241016 120000',
                                  '20241017 120000', '20241018 120000', '20241021 120000', '20241022 120000',
                                  '20241023 120000', '20241024 120000', '20241025 120000', '20241028 120000',
                                  '20241029 120000', '20241030 120000', '20241031 120000', '20241101 120000',
                                  '20241104 120000', '20241105 120000', '20241106 120000', '20241107 120000',
                                  '20241108 120000', '20241111 120000', '20241112 120000', '20241113 120000', 
                                  '20241114 120000', '20241115 120000', '20241118 120000'],
                "fundamental_file_path": "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Merged_data/INTC_merged_mid_price_train.pkl", 
                "log_orders": False,
                "book_logging": False,
                "book_log_depth": 10,
                "starting_cash": 200_000,
                "seed": 1,
                "num_noise_agents": 5000,
                "num_momentum_agents": 40,
                "m_wake_up_freq": "0.5s",
                "num_value_agents": 200,
                "val_wake_up_freq": "60s",
                "val_lambda_a": 7e-11, 
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
        "framework": "torch",
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
        "lr":  2e-4, #5e-6
        "observation_filter": "MeanStdFilter",  # Normalize observations
        # Reply buffer specification
        "local_replay_buffer": True,
        "replay_buffer_config": {
            "type": "MultiAgentReplayBuffer",
            "capacity": 200000,
            "storage_unit": "timesteps",
        }  
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