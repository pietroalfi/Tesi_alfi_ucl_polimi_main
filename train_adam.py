import ray
from ray.tune.registry import register_env
from abides_gym.envs.markets_execution_environment_v0 import (
    SubGymMarketsExecutionEnv_v0,
)
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from typing import Dict
import numpy as np
from collections import defaultdict

ray.shutdown()  # Shut down any currently running experiments
ray.init()  # Set up a new local Ray server

# Register a custom environment, so it can be used in `ray.tune.run()`
register_env(
    "markets-execution-v0",
    lambda config: SubGymMarketsExecutionEnv_v0(**config),
)

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.user_data = defaultdict(list)
        episode.user_data["q_values"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        policy = next(iter(policies.values()))  # Get the first (and likely only) policy
        episode.user_data["q_values"].append(policy.q_values)

        agent0_info = episode._agent_to_last_info["agent0"]
        for k, v in agent0_info.items():
            episode.user_data[k].append(v)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        q_values = np.array(episode.user_data["q_values"])
            
        episode.custom_metrics["q_value_mean"] = np.mean(q_values)
        episode.custom_metrics["q_value_max"] = np.max(q_values)
        episode.custom_metrics["q_value_min"] = np.min(q_values)

        # q_spread = np.max(q_values, axis=1) - np.min(q_values, axis=1)
        # episode.custom_metrics["q_value_spread_mean"] = np.mean(q_spread)

        q_probs = np.exp(q_values - np.max(q_values, axis=1, keepdims=True))
        q_probs = q_probs / np.sum(q_probs, axis=1, keepdims=True)
        entropy = -np.sum(q_probs * np.log(q_probs + 1e-10), axis=1)
        episode.custom_metrics["q_value_entropy_mean"] = np.mean(entropy)


        for metric in [
            "bid_volume", "ask_volume",
            "bid_volume_agent", "ask_volume_agent",
        ]:
            episode.custom_metrics[metric] = np.sum(episode.user_data[metric])

        episode.custom_metrics["avg_spread"] = np.mean(np.array(episode.user_data["ask"]) - np.array(episode.user_data["bid"]))
        
        for metric in [
            "cash_balance",
            "holdings"
        ]:
            episode.custom_metrics["final_"+metric] = episode.user_data[metric][-1]
            episode.custom_metrics[metric+"_vol"] = np.std(episode.user_data[metric])
            episode.custom_metrics[metric] = episode.user_data[metric]
        
        for i in range(0,3):
            episode.custom_metrics[f"d_signal_{i}"] = np.mean(episode.user_data[f"directional_signal_{i}"])
            episode.custom_metrics[f"d_signal_{i}_vol"] = np.std(episode.user_data[f"directional_signal_{i}"])

        action_counter = episode.user_data["action_counter"][-1]
        total_actions = sum(action_counter.values())
        for key, val in action_counter.items():
            episode.custom_metrics[f"pct_action_counter_{key}_{i}"] = val/total_actions

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass
        # policy = trainer.get_policy()
        # grad_norms = []
        # for param in policy.model.parameters():
        #     if param.grad is not None:
        #         grad_norms.append(param.grad.data.norm(2).item())
        # if grad_norms:
        #     result['custom_metrics']['grad_norm_mean'] = np.mean(grad_norms)
        #     result['custom_metrics']['grad_norm_max'] = np.max(grad_norms)


    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        pass
        # train_batch["actions"]
        # result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass


# tensorboard --logdir=~/ray_results/[NAME] 

tune.run(
    "APEX",  # Algorithm to use
    name="kappa_075_a5",  # Experiment name
    raise_on_failed_trial=False,
    resume=False,  # Set to True if you want to continue a stopped experiment, requires name and local_dir
    checkpoint_at_end=True,  # Save at the end
    checkpoint_freq=4,  # Create checkpoint every n iterations
    local_dir="/cs/student/projects3/cf/2023/adamkeys/ucl-thesis/results",
    stop={  # Stopping condition(s), OR clause
        # "training_iteration": 100,
        # "episode_reward_mean": 150.0,
        "timesteps_total": 10_000_000,
        # "time_total_s": 100,
    },
    config={
        # Environment
        "env": "markets-execution-v0",
        "env_config": {
            "background_config": "FX_test_execution",
            "starting_cash": 2_000_000,
            # Signal
            "dirichlet_steps_ahead": 8,
            "dirichlet_k": 0.0002,
            "dirichlet_a_h": 5.0,
            "dirichlet_a_l": 1.0,
            "w_direc": 1.0,
            "reduction_factor": 0.9999,
            "kappa": 1.0,
            # Reward function
            "background_config_extra_kvargs": {  # Overwrite fx_basic config
                "historical_date_start": "20240103",
                "historical_date_end": "20240128",
                "random_date": True,
                "start_time": "09:30:00",
                "end_time": "10:30:00",
                "fundamental_file_path": "/cs/student/msc/cf/2023/adamkeys/ucl-thesis/data/EURUSD_midpoint_x10.pkl",  # <-----------
                "starting_cash": 2_000_000,
                "num_momentum_agents": 40,
                "num_noise_agents": 5000,
                "num_value_agents": 200,
                "num_mm": 2,
            }
        },
        # General
        "framework": "tfe",
        "seed": 1,
        # Logging
        "timesteps_per_iteration": 10_000,
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
        #"noisy": False,
        # Neural net
        "hiddens": [32, 64],
        "model": {
            "fcnet_hiddens": [64, 128, 64],
            "fcnet_activation": "tanh"
        },
        # Learning parameters
        "gamma": 0.99,
        #"lr_schedule": [[0,2e-5], [1e6, 5e-6]],
        "lr": 5e-6,
        "observation_filter": "MeanStdFilter",  # Normalize observations (not sure if this works)
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